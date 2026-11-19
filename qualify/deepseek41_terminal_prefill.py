# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare complete and deferred prefill through native graph replay at TP1.

Uses synthetic token sequences to test state preservation, not model quality or
serving performance. Requires the qualification-enabled Heliopause extension.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sm103-preparation', type=Path, required=True)
    parser.add_argument('--gpu', required=True, help='Assigned GPU UUID')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from types import SimpleNamespace
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply

    configuration = Configuration.load(args.config)
    apply(configuration, [SimpleNamespace(uuid=args.gpu)])
    import torch
    from ignition.preparation.core_storage import CoreStorage
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig, prepare_rank
    from ignition.recipes.deepseek41.sm103 import SM103Config

    config = PreparationConfig(devices=(0,), max_active=1, image_concurrency=1,
        prefill_chunk_size=8192, speculative_widths=tuple(range(1, 9)),
        speculative_token_capacity=128, kv_pages=2048, snapshot_budget_bytes=64 << 20,
        kernel_cache=configuration.directory('kernel_cache'), compile_workers=16,
        sm103=SM103Config.load(args.sm103_preparation))
    report = dict(scope='synthetic TP1 state-equivalence diagnostic; not serving performance',
                  snapshot=str(args.snapshot), preparation=asdict(config), cases=[], passed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, default=str, indent=2) + '\n')

    save()
    with torch.no_grad(), CoreStorage(world=1) as storage:
        model = prepare_rank(args.snapshot, config, rank=0, group=None, core_storage=storage,
                             progress=lambda message: print(message, flush=True))
    try:
        comparison = Comparison(model)
        # Both routes execute identical chunk sizes. The only changed work is
        # omission of decoder/state-finalization work in intermediate chunks.
        minimum = comparison.layout.terminal_prefill_min_tokens
        for chunks in ((128,), (minimum,), (128, minimum), (8192, 128, minimum),
                       (4096, 128, 256, minimum), (8064, minimum), (8192 - minimum, minimum),
                       (8192, 8192), (8192, 8192, minimum)):
            reference = comparison.run(chunks, defer=False)
            repeat = comparison.run(chunks, defer=False)
            reference_differences = differences(reference, repeat)
            candidate = comparison.run(chunks, defer=True)
            state_differences = differences(reference, candidate)
            case = dict(chunks=chunks, reference_repeat_differences=reference_differences,
                        deferred_differences=state_differences)
            report['cases'].append(case)
            save()
            print(json.dumps(case), flush=True)
        report['passed'] = all(not case['reference_repeat_differences']
                               and not case['deferred_differences'] for case in report['cases'])
        save()
        if not report['passed']:
            raise AssertionError('state comparison failed; raw result retains reference repeatability separately')
    finally:
        model.close()


class Comparison:
    def __init__(self, model):
        from ignition.direct.scheduler import Layout
        from ignition.components.paged_state import PagedStateStorage
        from ignition.components.sampling import DeviceRequestState
        from ignition.components.dspark import SpeculativePreparation
        from ignition.components.speculation import SpeculativeRngState
        from ignition.recipes.deepseek41.reference_speculation import ReferenceSpeculation

        self.model = model
        self.layout = Layout.of(self.model)
        self.pool = next(value for value in self.model.resources if isinstance(value, PagedStateStorage))
        self.requests = next(value for value in self.model.resources if isinstance(value, DeviceRequestState))
        self.speculation = next((value for value in self.model.resources
                                 if isinstance(value, SpeculativePreparation)), None)
        self.rng = (self.speculation.rng if self.speculation is not None
                    else next(value for value in self.model.resources if isinstance(value, SpeculativeRngState)))
        forwards = next((value for value in self.model.resources if isinstance(value, dict)), {})
        ref_spec = next((op for op in forwards.values() if isinstance(op, ReferenceSpeculation)), None)
        self.proposal_generation = (self.speculation.generation if self.speculation is not None
                                    else ref_spec.proposal_generation)
        self.proposal_offset = (self.speculation.offset if self.speculation is not None
                                else ref_spec.proposal_offset)
        self.stagers = tuple(stager for value in self.model.resources
                             for stager in getattr(value, 'expert_stagers', {}).values())
        manifest = self.model.manifest()
        self.indices = {graph['name']: i for i, graph in enumerate(manifest['graphs'])}
        # These executor inputs are separate from semantic text packing. This
        # diagnostic requests no generation logprobs, including on reused slots.
        self.logprob_uploads = [(name, 0, bytes(self.layout.capacity * 4))
                               for name in ('logprob_requested', 'logprob_token_id_counts')
                               if name in self.model.buffers]
        if self.model.embedding_catalog is not None:
            self.model.prepare_embedding_tables()
        self.executor = self.model.runtime = self.model.start_executor()
        self.sequence = self.generation = 0

    def execute(self, scheduler, step):
        print(f'replaying {step.graph}: {[lane.tokens for lane in step.lanes]}', flush=True)
        uploads = [*scheduler.pack(step), *self.logprob_uploads]
        fence = self.executor.submit(self.indices[step.graph], uploads)
        if fence is None:
            raise RuntimeError('native executor rejected an isolated diagnostic submission')
        deadline = time.monotonic() + 120
        while (result := self.executor.poll()) is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f'native graph {step.graph} did not complete')
            time.sleep(.0001)
        assert result.fence == fence and result.graph_index == self.indices[step.graph]
        step.sequence = scheduler.sequence = self.sequence = fence
        scheduler.complete(step, result.records, confidence=result.proposal_confidence)
        if any(lane.request.error is not None for lane in step.lanes):
            raise AssertionError(f'{step.graph} returned a device error')
        return result

    def state(self, request):
        import torch

        row = self.layout.state_row(request.slot)
        # State regions exclude alignment padding; include every declared history
        # and every assigned page, including the partial page and draft reserve.
        values = {region.name: self.pool.region(region.name)[row].cpu().clone()
                  for region in self.pool.layout.regions}
        indices = torch.tensor(request.pages, dtype=torch.int64, device=self.pool.backing.device)
        values.update({'page.' + name: tensor.index_select(0, indices).cpu()
                       for name, tensor in self.pool.pages.items()})
        values.update({'request.' + name: tensor[request.slot].cpu().clone()
                       for name, tensor in vars(self.requests).items()
                       if isinstance(tensor, torch.Tensor) and name != 'generation'})
        values['draft_rng'] = self.proposal_offset[request.slot].cpu().clone()
        values['target_rng'] = self.rng.target_offset[request.slot].cpu().clone()
        values['tokens'] = torch.tensor(request.generated)
        values['consumed'] = torch.tensor(request.consumed)
        return values

    def run(self, chunks, *, defer):
        import torch
        from ignition.direct.scheduler import Lane, Request, SamplingParameters, Scheduler

        scheduler = Scheduler(self.layout, prefix_cache=False)
        scheduler.sequence = self.sequence
        scheduler.generations[0] = self.generation
        # Initialize unused bytes equally before each isolated run. No graph is
        # outstanding here; resetting pages is diagnostic setup, never serving.
        rows, pages = (int(value) for value in self.pool.memory_capacity[:2].cpu())
        self.pool.backing[:rows].zero_()
        self.proposal_generation.zero_()
        self.proposal_offset.zero_()
        self.rng.generation.zero_()
        self.rng.target_offset.zero_()
        for stager in self.stagers:
            stager.decision.zero_()
        for tensor in self.pool.pages.values():
            tensor[:pages].zero_()
        # Native replay uses a nonblocking stream. Finish diagnostic setup on
        # PyTorch's stream before handing the same storage back to the executor.
        torch.cuda.synchronize()
        tokens = [100 + (position * 37) % 20000 for position in range(sum(chunks))]
        request = Request(1, tokens, 64, SamplingParameters(seed=17, top_k=1), ignore_eos=True)
        scheduler.submit(request)
        self.execute(scheduler, scheduler.plan())
        for i, count in enumerate(chunks):
            kind = 'prefill' if defer and i + 1 < len(chunks) else 'terminal_prefill'
            step = scheduler._step(kind, count, [Lane(request, request.consumed, count)])
            self.execute(scheduler, step)
            if i + 1 < len(chunks):
                assert not request.generated
        assert request.consumed == len(tokens) and len(request.generated) == 1
        state = {'prefill.' + key: value for key, value in self.state(request).items()}
        for _ in range(2):
            self.execute(scheduler, scheduler.plan())
        state.update({'continuation.' + key: value for key, value in self.state(request).items()})
        self.generation = request.generation
        torch.cuda.synchronize()
        return state


def differences(left, right):
    import torch

    if left.keys() != right.keys():
        raise AssertionError('compared states have different declared regions')
    result = {}
    for name, reference in left.items():
        candidate = right[name]
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            result[name] = dict(reference_shape=list(reference.shape), candidate_shape=list(candidate.shape),
                                reference_dtype=str(reference.dtype), candidate_dtype=str(candidate.dtype))
            continue
        count = int(torch.count_nonzero(reference.reshape(-1).view(torch.uint8)
                                        != candidate.reshape(-1).view(torch.uint8)))
        if count:
            result[name] = count
    return result


if __name__ == '__main__':
    main()
