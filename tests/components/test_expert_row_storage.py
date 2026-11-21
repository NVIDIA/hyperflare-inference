"""CPU checks of the prepared canonical row-storage declaration."""
import pytest
import torch
from google.protobuf.json_format import ParseDict

from ignition.components.expert_row_storage import megamoe_row_storage
from ignition.components.megamoe_tiers import MegaMoeHostTier, MegaMoeLayerWeights
from ignition.components.megamoe_weights import MegaMoeWeights
from ignition.preparation import runtime_core_pb2 as proto


def weights():
    def planes():
        return (torch.empty_strided((2, 4, 6), (24, 1, 4), dtype=torch.uint8),
                torch.empty((2, 8), dtype=torch.int32))
    hot, cold = MegaMoeWeights(planes(), planes()), MegaMoeWeights(planes(), planes())
    # Only tensor enumeration is under test; CUDA storage construction has its
    # separate hardware tests. Retain the real container type for classification.
    tier = object.__new__(MegaMoeHostTier)
    tier.weights = cold
    return MegaMoeLayerWeights(hot, None, None, tier,
                              torch.tensor([2, 3, 1], dtype=torch.int32), (1, 2), (0, 1))


def test_export_declares_canonical_rows_and_original_mapping_buffer():
    value = weights()
    catalog, buffers = megamoe_row_storage([(7, value, "weights.layers[7].experts.experts")])
    message = ParseDict(catalog, proto.ExpertRowCatalog())
    restored = proto.ExpertRowCatalog.FromString(message.SerializeToString())
    layer, = restored.layers
    assert layer.layer == 7 and layer.experts == 3
    assert list(layer.host_experts) == [1, 2] and list(layer.hbm_experts) == [0, 1]
    assert buffers[layer.mapping_buffer] is value.expert_map
    assert [p.host_weight_view for p in layer.planes] == [
        f"weights.layers[7].experts.experts.host_tier.weights.{part}[{i}]"
        for part in ("gate_up", "down") for i in (0, 1)]
    assert torch.equal(value.expert_map, torch.tensor([2, 3, 1], dtype=torch.int32))
    assert megamoe_row_storage([]) == (None, {})


def test_declared_plane_names_resolve_through_weight_classification():
    from dataclasses import dataclass
    from ignition.components import weights as _classification
    from ignition.preparation.rebuild_manifest import declare_weights, declared_views

    @dataclass
    class Root:
        experts: MegaMoeLayerWeights

    value = weights()
    catalog, _ = megamoe_row_storage([(7, value, "weights.experts")])
    declaration = declare_weights(Root(value), components=("experts",))
    views = {name: tensor for name, _, tensor, _, _ in declared_views(declaration)}
    cold = (*value.host_tier.weights.gate_up, *value.host_tier.weights.down)
    hot = (*value.routed.gate_up, *value.routed.down)
    for plane, c, h in zip(catalog["layers"][0]["planes"], cold, hot, strict=True):
        assert views[plane["host_weight_view"]] is c
        assert views[plane["hbm_weight_view"]] is h
    assert all(tensor is not value.expert_map for tensor in views.values())


@pytest.mark.parametrize("fault", ["duplicate", "missing", "shape", "holes", "count", "map"])
def test_reject_invalid_prepared_rows(fault):
    from dataclasses import replace

    value = weights()
    if fault == "duplicate":
        value = replace(value, host_experts=(1, 1))
    elif fault == "missing":
        value = replace(value, hbm_experts=(1, 2))
    elif fault == "shape":
        value.host_tier.weights = replace(value.host_tier.weights, down=(torch.empty((2, 8)), torch.empty((2, 8))))
    elif fault == "holes":
        hole = torch.empty_strided((2, 4, 6), (24, 1, 3), dtype=torch.uint8)
        value = replace(value, routed=replace(value.routed, gate_up=(hole, value.routed.gate_up[1])))
        value.host_tier.weights = replace(value.host_tier.weights, gate_up=(hole, value.routed.gate_up[1]))
    elif fault == "count":
        value = replace(value, hbm_experts=(0,))
    elif fault == "map":
        value = replace(value, expert_map=value.expert_map.to(torch.int64))
    with pytest.raises(ValueError):
        megamoe_row_storage([(7, value, "weights")])
