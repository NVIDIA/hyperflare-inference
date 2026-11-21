# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Decoder suffix metadata preserves live positions, cache pages, and replay bounds."""

import pytest
import torch
from types import SimpleNamespace

from ignition.components.paged_batch import Metadata
from ignition.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix


@pytest.mark.parametrize("capacity", [256, 512, 4096])
@pytest.mark.parametrize("start", [0, 127, 4095])
def test_decoder_suffix_alternates_full_partial_and_inactive_chunks(capacity, start):
    source = Metadata.allocate(capacity, 1, 40, "cpu")
    packing = PreparedDecoderSuffix(source).functional()
    values = torch.arange(capacity * 12).reshape(capacity, 3, 4)
    output = torch.empty((128, 3, 4), dtype=values.dtype)
    pages = torch.arange(40).flip(0).view(1, -1)
    source.page_table.copy_(pages)
    source.starts.fill_(start)
    source.query_start_loc[0].zero_()
    source.token_ids.copy_(torch.arange(capacity) + 1000)
    for count in (capacity, 129, 128, 1, 127, 0, capacity - 1):
        source.lengths.fill_(count)
        source.query_start_loc[1] = count
        source.state_rows.fill_(7 if count else 0)
        source.live_counts.copy_(torch.tensor([count, 1 if count else 0]))
        source.sample_indices.fill_(max(0, count - 1))
        source.positions.fill_(-1)
        source.positions[:count] = torch.arange(start, start + count)
        output.fill_(-999)
        packed = packing.begin(source)
        output.copy_(packed.gather(values))
        target = packed.metadata
        retained = list(range(max(0, count - 128), count))
        kept = len(retained)
        assert target.positions.tolist() == [start + index for index in retained] + [-1] * (128 - kept)
        assert target.token_ids.tolist() == [1000 + index for index in retained] + [0] * (128 - kept)
        assert target.starts.item() == start + max(0, count - 128)
        assert target.lengths.item() == kept
        assert target.query_start_loc.tolist() == [0, kept]
        assert target.sample_indices.item() == max(0, kept - 1)
        assert target.state_rows.item() == (7 if count else 0)
        assert target.live_counts.tolist() == [kept, 1 if count else 0]
        assert target.requests.eq(0).all()
        assert torch.equal(target.page_table, pages)
        assert torch.equal(target.token_page_table, pages.expand(128, -1))
        expected = torch.zeros_like(output)
        if retained:
            expected[:kept] = values[retained]
        assert torch.equal(output, expected)
        assert packed.replay_start.item() == (start + retained[0] if count > 128 else 0)


@pytest.mark.parametrize("rows,tokens", [(2, 256), (1, 128)])
def test_decoder_suffix_rejects_unsupported_packing_geometry(rows, tokens):
    with pytest.raises(ValueError, match="one prefill chunk larger"):
        PreparedDecoderSuffix(Metadata.allocate(tokens, rows, 2, "cpu"))


def test_decoder_feedback_at_replay_boundary_is_rejected_before_component_construction(monkeypatch):
    from ignition.recipes.deepseek41 import decoder

    monkeypatch.setattr(decoder, "PreparedTableRequests", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder.Scratch, "allocate", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder, "PreparedEmbedding",
                        lambda *args, **kwargs: pytest.fail("unsupported feedback reached component construction"))
    weights = SimpleNamespace(config=SimpleNamespace(decoder_start=20,
        sources=(SimpleNamespace(layer=20),), draft=SimpleNamespace(target_layers=(20,))),
        draft=object(), layers=tuple(SimpleNamespace(engram=None) for _ in range(40)),
        shard=SimpleNamespace(world_size=4), embedding=None)
    scratch = SimpleNamespace(embedding=torch.zeros(1), hidden=torch.zeros((1, 5120)),
        local_logits=torch.zeros((1, 1)), logits=torch.zeros((1, 1)))
    with pytest.raises(ValueError, match="draft target layers after"):
        decoder.PreparedDecoder(weights, None, scratch, Metadata.allocate(256, 1, 2, "cpu"), None,
            mode="prefill", runtime=None, vocabulary_runtime=None, coordinator=None,
            session=None, name="boundary", workspace=None)


def test_prepared_prefill_rejects_decoder_without_terminal_keyword(monkeypatch):
    from ignition.components import paged_generation

    class LegacyDecoder:
        def __call__(self, metadata):
            return None

        def release_outputs(self):
            pass

    batch = SimpleNamespace(
        inputs=SimpleNamespace(row_slot_ids=torch.zeros(1), row_generations=torch.zeros(1), live_rows=torch.zeros(1)),
        rows=1,
        requests=object(),
        functional=lambda: None,
    )
    with pytest.raises(TypeError, match="terminal=False"):
        paged_generation.PreparedPrefill(LegacyDecoder(), batch, None, None, None, validation=lambda v: v)


def test_prepared_prefill_invokes_decoder_with_terminal_false_and_completes(monkeypatch):
    from ignition.components import paged_generation

    events = []

    class FakeCompletion:
        def __init__(self, *args):
            self.prepared = False

        def prepare(self):
            self.prepared = True

        def __call__(self):
            events.append("complete")

    monkeypatch.setattr(paged_generation, "PreparedCompletion", FakeCompletion)

    class BoundaryDecoder:
        def __init__(self):
            self.released = False

        def release_outputs(self):
            self.released = True

        def __call__(self, metadata, *, terminal=True):
            events.append(("decoder", metadata, terminal))

    class FakeBatch:
        def __init__(self):
            self.inputs = SimpleNamespace(
                row_slot_ids=torch.zeros(1), row_generations=torch.zeros(1), live_rows=torch.zeros(1)
            )
            self.rows = 1
            self.requests = object()
            self.functional_calls = 0

        def functional(self):
            self.functional_calls += 1

        def validate(self):
            events.append("validate")
            return "v", "p"

        def begin(self, totals, *, present):
            events.append(("begin", totals, present))
            return "meta"

        def finish(self, metadata):
            events.append(("finish", metadata))

    decoder = BoundaryDecoder()
    batch = FakeBatch()
    prefill = paged_generation.PreparedPrefill(
        decoder, batch, None, None, None, validation=lambda value: ("reduced", value)
    )
    assert decoder.released and batch.functional_calls == 1 and prefill.completion.prepared
    prefill()
    assert events == [
        "validate",
        ("begin", ("reduced", "v"), "p"),
        ("decoder", "meta", False),
        ("finish", "meta"),
        "complete",
    ]


@pytest.mark.parametrize("with_boundary", [True, False])
def test_deepseek_decoder_non_terminal_stops_at_global_kv_boundary(with_boundary):
    from ignition.recipes.deepseek41.decoder import PreparedDecoder

    events = []
    decoder = object.__new__(PreparedDecoder)
    decoder.mode = "prefill"
    decoder.scoring = None
    decoder.decoder_start = 1
    decoder.vision = None
    decoder.staging_shape = None
    decoder.auxiliary_shape = (4, 0)
    decoder.aux_layers = ()
    decoder.compact_head = False
    decoder.weights = SimpleNamespace(embedding=torch.zeros((16, 4)))

    class FakeTableRequests:
        def __call__(self, metadata):
            events.append("table_prepare")
            return "raw_history"

        def commit(self, lengths, raw_history, metadata):
            events.append(("table_commit", raw_history))

    decoder.table_requests = FakeTableRequests()
    decoder.embedding = lambda *, token_ids, num_rows: torch.ones((token_ids.numel(), 4), dtype=torch.bfloat16)

    class FakeIndices:
        def compressed(self, ratio):
            return ("compressed", ratio)

    decoder.window_metadata = lambda metadata: FakeIndices()

    class EncoderLayer:
        boundary = None

        def __call__(self, incoming, pre_mix, metadata, indices, selection, *, boundary=None, window_cache=None):
            events.append("encoder_layer")
            return incoming, pre_mix, "enc_rec", selection

    class BoundaryHelper:
        def produce_global_kv(self, incoming, pre_mix, metadata, indices, indexer):
            events.append(("boundary_global_kv", indexer, indices.compressed(1)))
            return None, pre_mix

    class FakeGlobalKV:
        ratio = 1

        def __call__(self, normalized, indexer, metadata, geometry):
            events.append(("layer_global_kv", indexer, geometry))

    class DecoderStartLayer:
        def __init__(self):
            self.boundary = BoundaryHelper() if with_boundary else None
            self.attention = SimpleNamespace(selectors=["indexer0"], global_kv=FakeGlobalKV())

        def attention_hc(self, *incoming, pre_mix, pre_out):
            events.append("layer_mhc")
            return None, None, None, "normalized"

        def __call__(self, *args, **kwargs):
            pytest.fail("non-terminal prefill must not execute decoder attention/FFN")

    class TrailingDecoderLayer:
        boundary = None

        def __call__(self, *args, **kwargs):
            pytest.fail("non-terminal prefill must not execute layers after decoder_start")

    decoder.layers = [EncoderLayer(), DecoderStartLayer(), TrailingDecoderLayer()]
    decoder.head = lambda *args, **kwargs: pytest.fail("non-terminal prefill must not project logits")
    metadata = Metadata.allocate(4, 1, 2, "cpu")
    hidden, logits, auxiliary, (_, records, raw) = decoder(metadata, terminal=False)
    assert (hidden, logits, auxiliary, records, raw) == (None, None, None, ("enc_rec",), "raw_history")
    expected_boundary = (
        [("boundary_global_kv", "indexer0", ("compressed", 1))]
        if with_boundary
        else ["layer_mhc", ("layer_global_kv", "indexer0", ("compressed", 1))]
    )
    assert events == ["table_prepare", "encoder_layer", *expected_boundary, ("table_commit", "raw_history")]


@pytest.mark.parametrize("mode,scoring,decoder_start", [
    ("decode", None, 20),
    ("prefill", object(), 20),
    ("prefill", None, None),
])
def test_deepseek_decoder_rejects_invalid_non_terminal_configurations(mode, scoring, decoder_start):
    from ignition.recipes.deepseek41.decoder import PreparedDecoder

    decoder = object.__new__(PreparedDecoder)
    decoder.mode = mode
    decoder.scoring = scoring
    decoder.decoder_start = decoder_start
    with pytest.raises(ValueError, match="non-terminal prefill"):
        decoder(Metadata.allocate(4, 1, 2, "cpu"), terminal=False)
