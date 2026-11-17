"""Describe row-mapped expert storage without changing weights or graph operands."""

import torch


def megamoe_row_storage(layers):
    """Return export metadata and mapping buffers for target layers.

    Entries are (canonical layer, MegaMoeLayerWeights, declared weight path).
    Callers exclude draft layers explicitly. Plane names resolve through the
    existing prepared-weight catalog; physical pointers are never configuration.
    """
    descriptions, buffers, seen = [], {}, set()
    for layer, weights, path in layers:
        host, hbm = weights.host_experts, weights.hbm_experts
        count = weights.expert_map.numel()
        if (type(layer) is not int or not 0 <= layer < 2**32 or layer in seen
                or not path or not hbm or not host or weights.host_tier is None
                or count >= 2**31 or count == 0
                or any(type(i) is not int or not 0 <= i < count for i in (*host, *hbm))
                or len(set(host)) != len(host) or len(set(hbm)) != len(hbm)
                or set(host) | set(hbm) != set(range(count))
                or weights.expert_map.shape != (count,)
                or weights.expert_map.dtype != torch.int32 or not weights.expert_map.is_contiguous()):
            raise ValueError("expert row storage requires complete, unique canonical tier layouts")
        seen.add(layer)
        planes = []
        for field in ("gate_up", "down"):
            for part, (cold, hot) in enumerate(zip(getattr(weights.host_tier.weights, field),
                                                  getattr(weights.routed, field), strict=True)):
                if (cold.shape[0] != len(host) or hot.shape[0] != len(hbm)
                        or cold.shape[1:] != hot.shape[1:] or cold.stride() != hot.stride()
                        or cold.dtype != hot.dtype or cold.device != hot.device
                        or cold.device != weights.expert_map.device):
                    raise ValueError("expert row planes disagree with the declared tier layout")
                # Inner axes may be transposed, but each row must occupy one dense span.
                extent = 1
                for stride, size in sorted(zip(hot.stride()[1:], hot.shape[1:])):
                    if size <= 0 or (size > 1 and stride != extent):
                        raise ValueError("expert row planes require dense, nonoverlapping rows")
                    extent *= size
                if hot.stride(0) != extent:
                    raise ValueError("expert row planes require dense, nonoverlapping rows")
                planes.append(dict(host_weight_view=f"{path}.host_tier.weights.{field}[{part}]",
                                   hbm_weight_view=f"{path}.routed.{field}[{part}]"))
        name = f"expert_row_map_{layer}"
        buffers[name] = weights.expert_map
        descriptions.append(dict(layer=layer, experts=count, host_experts=list(host), hbm_experts=list(hbm),
                                 mapping_buffer=name, planes=planes))
    return (dict(version=1, layers=descriptions) if descriptions else None), buffers
