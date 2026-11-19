# SPDX-License-Identifier: Apache-2.0

"""Choose complete prefill context transfers before compiling model forwards."""

from functools import partial
from types import SimpleNamespace

import torch
from b12x.preparation import PreparedCall

from ignition.components.shared_pages import SharedPages
from ignition.preparation.capture import capture_forwards
from ignition.preparation.composition_tuning import PreparedComposition, plan_identity, race_compositions, trial_session
from ignition.preparation.execution import GraphSignature

from .dcp import PreparedContextGather


class _Forward:
    def __init__(self, fixture, implementation):
        self.fixture, self.implementation = fixture, implementation

    def __call__(self):
        f = self.fixture
        f.operation(f.scratch, f.table, f.end, implementation=self.implementation)


class _Fixture:
    # Bound integer oracle construction and validation, including simultaneous intermediates.
    oracle_rows = 4096
    temporary_bytes = 32 << 20

    def __init__(self, positions, window, context, session):
        device, world = context.shards.index.device, context.shards.world
        pages = positions // (256 * world)
        self.shared = SharedPages(context.coordinator.group, device, pages * 256 * 528)
        try:
            self._allocate(positions, window, pages, context, session)
        except BaseException:
            self.shared.close()
            raise

    def _allocate(self, positions, window, pages, context, session):
        device, world = context.shards.index.device, context.shards.world
        self.cache = self.shared.tensor.view(pages * 4, 64, 528)
        words = torch.arange(132, device=device)[None, :]
        cache = self.cache.view(torch.int32).view(-1, 132)
        for first in range(0, cache.shape[0], self.oracle_rows):
            physical = torch.arange(first, min(first + self.oracle_rows, cache.shape[0]), device=device)[:, None]
            cache[first:first + self.oracle_rows].copy_(context.shards.rank * 100000003 + physical * 132 + words)
        self.table = torch.arange(pages - 1, -1, -1, dtype=torch.int32, device=device)
        self.end = torch.zeros(1, dtype=torch.int64, device=device)
        self.scratch = SimpleNamespace(
            slab=torch.zeros((positions, 528), dtype=torch.uint8, device=device),
            records=torch.empty((world, window * 528), dtype=torch.uint8, device=device),
            record_counts=torch.zeros(world, dtype=torch.int64, device=device))
        state = SimpleNamespace(cache=self.cache, peers=self.shared.addresses)
        # The trial's own gather: `context` has no selection_tuning, so it registers nowhere.
        self.operation = PreparedContextGather(state, self.scratch, context, session=session, name="context_trial")
        self.operation.functional(session=session)
        self.forwards = {kind: _Forward(self, kind) for kind in PreparedContextGather.IMPLEMENTATIONS}
        # The race times three context lengths; the holdout length is one it never times.
        self.raced = (5, min(8191, positions), positions)
        self.lengths = (*self.raced, positions // 2 + 3)
        self.step = 0

    def produce(self, index):
        self.step = index
        self.end.fill_(self.lengths[index])

    def reset(self):
        self.scratch.slab.zero_()
        self.scratch.records.zero_()

    def validate(self):
        device, world = self.end.device, self.operation.context.shards.world
        words = torch.arange(132, device=device)[None, :]
        for first in range(0, self.scratch.slab.shape[0], self.oracle_rows):
            token = torch.arange(first, min(first + self.oracle_rows, self.scratch.slab.shape[0]), device=device)
            local = (token // (4 * world)) * 4 + token % 4
            physical = self.table[(local // 256).long()].long() * 256 + local % 256
            expected = ((token // 4 % world)[:, None] * 100000003 + physical[:, None] * 132 + words).int()
            expected.masked_fill_((token >= self.lengths[self.step])[:, None], 0)
            if not torch.equal(self.scratch.slab.view(torch.int32)[first:first + self.oracle_rows], expected):
                raise AssertionError("context transport changed latent cache bytes or slab padding")

    @property
    def nbytes(self):
        tensors = (self.cache, self.shared.addresses, self.table, self.end, *vars(self.scratch).values())
        return sum(tensor.nbytes for tensor in tensors) + self.temporary_bytes


class ContextGatherTuning:
    """Bind, for the prefill gathers of one preparation, the transport that measures fastest.

    Every prefill graph of a preparation shares one slab shape, so one private trial decides for all of
    them. The trial holds its own peer cache, slab and records and is released before the model's forwards
    are compiled. The race scores each transport by its mean latency over three context lengths (5,
    8,191 and the full slab positions), so the longest context dominates the choice.
    `context` is the context without `selection_tuning`, which the model's gathers then register with.
    """

    def __init__(self, context, *, cache_root=None):
        if context.selection_tuning is not None:
            raise ValueError("the trial context must not carry the tuning that registers the model's gathers")
        self.context, self.cache_root, self.gathers = context, cache_root, []

    def register(self, gather):
        if self.gathers and (gather.positions, gather.window) != (self.gathers[0].positions, self.gathers[0].window):
            raise ValueError("the prefill context gathers of one preparation share one slab shape")
        self.gathers.append(gather)

    def tune(self, session):
        """Race the transports, bind the winner to every registered gather, and return the race's record."""
        coordinator, device = self.context.coordinator, self.context.shards.index.device
        positions, window = self.gathers[0].positions, self.gathers[0].window
        with trial_session(session, device=device, coordinator=coordinator) as trials:
            fixture, graphs, compiled = None, {}, {}
            try:
                fixture = _Fixture(positions, window, self.context, trials)
                trials.tune(show_progress=False)
                for gather in self.gathers:
                    if self._identities(gather) != self._identities(fixture.operation):
                        raise ValueError("context trial plans differ from the model's prepared kernels")
                fixture.produce(0)
                fixture.reset()
                torch.cuda.synchronize(device)
                coordinator.barrier("context trial inputs", coordinator.ranks)
                signatures = {kind: GraphSignature("prefill", 1, positions) for kind in fixture.forwards}
                graphs, compiled = capture_forwards(fixture.forwards, signatures, session=trials,
                    device=device, stream=torch.cuda.Stream(device=device), pool=torch.cuda.graph_pool_handle(),
                    coordinator=coordinator, report_catalog=False, cache_root=self.cache_root, retain_forwards=True)
                chosen = self._race(fixture, compiled, trials)
                self._require_agreement(chosen["implementation"])
                # The winner must also reproduce the exact bytes at a length the race never timed.
                fixture.produce(len(fixture.raced))
                torch.cuda.synchronize(device)
                coordinator.barrier("context trial holdout", coordinator.ranks)
                fixture.reset()
                compiled[chosen["implementation"]]()
                torch.cuda.synchronize(device)
                fixture.validate()
                for gather in self.gathers:
                    gather.bind(chosen["implementation"])
            finally:
                for graph in graphs.values():
                    graph.reset()
                for forward in compiled.values():
                    forward.retire_after_capture()
                graphs.clear()
                compiled.clear()
                if fixture is not None:
                    fixture.shared.close()
                    fixture.forwards.clear()
        return dict(positions=positions, **chosen)

    def _require_agreement(self, implementation):
        """Every rank must run the same transport: ranks that differ would deadlock in the ring's gathers."""
        if not self.context.coordinator.same_on_every_rank(implementation):
            raise RuntimeError("ranks selected different prefill context transports")

    @staticmethod
    def _identities(gather):
        return tuple((role, plan_identity(plan)) for role, plan in gather.plans("ring"))

    def _race(self, fixture, compiled, trials):
        candidates = {}
        for kind, forward in compiled.items():
            producers = tuple(partial(fixture.produce, index) for index in range(len(fixture.raced)))
            call = PreparedCall(run=forward, output=fixture.scratch.slab,
                produce=producers[0], benchmark_producers=producers, reset=fixture.reset, validate=fixture.validate,
                owners=(fixture,), capture_safe=True, benchmark_repetitions=8)
            candidates[kind] = PreparedComposition(forward, fixture.operation.plans(kind), call)
        with trials._tuning_clocks():
            return race_compositions(candidates, operation="glm-prefill-context",
                geometry=dict(positions=fixture.scratch.slab.shape[0], window=fixture.operation.window,
                              world_size=self.context.shards.world),
                numerics="exact-cache-bytes", workload=dict(lengths=fixture.raced,
                    reversed_pages=True, rank_distinct=True), default="pull", session=trials,
                coordinator=self.context.coordinator, cache_policy="warm", resident_bytes=fixture.nbytes)
