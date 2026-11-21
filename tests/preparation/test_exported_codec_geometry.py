# SPDX-License-Identifier: Apache-2.0
"""Cache metadata shared with the Helios runtime geometry tests. Regenerate with --write."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from google.protobuf.json_format import ParseDict

from ignition.preparation import runtime_core_pb2 as wire

FIXTURES = Path(__file__).parents[1] / "fixtures" / "codec_geometry"


class Storage:
    """Allocate CPU tensors and record the same sizing and view references as core storage."""

    def __init__(self):
        self.tensors = {}
        self.allocations = {}
        self.dynamic_regions = []
        self.bundle = wire.BundleIndex(architecture=[12, 0], context=wire.Context())

    def device_allocate(self, name, shape, dtype, *, group, maximum, units):
        tensor = torch.empty(shape, dtype=dtype)
        address = tensor.data_ptr()
        self.tensors[address] = tensor
        self.dynamic_regions.append(SimpleNamespace(address=address, maximum=maximum))
        self.allocations[address] = self.bundle.allocations.add(name=name, address=len(self.tensors) << 32, alignment=65536,
            nbytes=tensor.nbytes, sizing=wire.Sizing(capacity=wire.Capacity.Value(group),
                unit_bytes=tensor[0].nbytes, maximum=maximum))
        return tensor

    def bind(self, buffers):
        for name, tensor in buffers.items():
            base = tensor.untyped_storage().data_ptr()
            allocation = self.allocations[base]
            self.bundle.execution.buffers.add(name=name, nbytes=tensor.nbytes,
                reference=wire.Reference(allocation=allocation.name, offset=tensor.data_ptr() - base))


def recipe(name, storage):
    if name == "glm53-tp4-mtp5":
        from ignition.recipes.glm53 import contract
        from ignition.recipes.glm53.bootstrap import PreparationConfig
        from ignition.recipes.glm53.pool import ContinuationPool
        layers = ["indexed_attention" if i % 4 == 3 else "linear_attention" for i in range(45)]
        config = PreparationConfig(devices=(0, 1, 2, 3), context_parallel=True, mtp=True, speculative_fixed_width=6)
        ranks = len(config.devices)
        pool = ContinuationPool.allocate(layers, ranks, 2, 2, "cpu", mtp=config.mtp, storage=storage,
                                         maximum_pages=2, context_ranks=ranks)
        tokens, lag = config.page_tokens, 1
        buffers = {"state_pool": pool.backing}
        regions = contract.history_regions(buffers, pool, tokens)
    elif name == "deepseek41":
        from ignition.recipes.deepseek41 import contract
        from ignition.recipes.deepseek41.config import ModelConfig
        from ignition.recipes.deepseek41.pool import ContinuationPool, PAGE_TOKENS
        config = ModelConfig(dict(num_hidden_layers=40, hidden_size=5120, sliding_window=128,
            kv_source_layer_ids=[2, 8, 14, 20], compress_ratios=[0] * 2 + [2] * 18 + [1] * 20 + [0] * 3,
            num_nextn_predict_layers=3, dspark_block_size=5, dspark_noise_token_id=128799,
            dspark_target_layer_ids=[37, 38, 39], dspark_markov_rank=256, dspark_n_routed_experts=128,
            dspark_num_experts_per_tok=3, vocab_size=129280), {})
        pool = ContinuationPool.allocate(config, 2, 2, "cpu", storage=storage, maximum_pages=2, maximum_rows=2, dspark=True)
        buffers = {"state_pool": pool.backing}
        tokens, lag, regions = PAGE_TOKENS, 0, []
    else:
        from ignition.recipes.qwen38 import contract
        from ignition.recipes.qwen38.attention import QSAState
        from ignition.recipes.qwen38.state import HybridStatePool, QSA_LAYERS
        world = 1 if name == "qwen38-tp1" else 2
        pool = HybridStatePool.allocate(2, "cpu", world_size=world, speculative_tokens=7, storage=storage)
        state = QSAState.allocate(pages=2, slots=2, device="cpu", draft_tokens=7, kv_heads=2 // world)
        cache = {i: state.resize_pages_for_preparation(2, storage=storage, prefix=f"kv-{i}-", backed_pages=2)
                 for i in (*QSA_LAYERS, 48)}
        buffers = {"state_pool": pool.backing}
        tokens, lag = 64, 1
        regions = contract.history_regions(buffers, cache)
    model = contract.prepare({}, buffers, pool, page_tokens=tokens, draft_lag=lag, regions=regions)
    storage.bind(buffers)
    return wire.ServingContract(model_contract=ParseDict(model, wire.ModelContract()))


def exported_geometry():
    from b12x.sequence._shared import disk_table

    for name in ("glm53-tp4-mtp5", "deepseek41", "qwen38-tp1", "qwen38-tp2"):
        storage = Storage()
        # The recipes request a virtual view over reserved rows. Only the two backed rows are used.
        with patch.object(torch.cuda, "mem_get_info", return_value=(1 << 30, 1 << 30)), \
             patch.object(disk_table, "_tensor_from_pointer", side_effect=lambda ptr, **_: storage.tensors[ptr]):
            serving = recipe(name, storage)
        yield name + ".contract.pb", serving.SerializeToString(deterministic=True)
        yield name + ".rank.pb", storage.bundle.SerializeToString(deterministic=True)


def test_exported_codec_geometry_matches_runtime_fixtures():
    for name, data in exported_geometry():
        assert (FIXTURES / name).read_bytes() == data, name


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", required=True)
    parser.parse_args()
    FIXTURES.mkdir(exist_ok=True)
    for name, data in exported_geometry():
        (FIXTURES / name).write_bytes(data)
