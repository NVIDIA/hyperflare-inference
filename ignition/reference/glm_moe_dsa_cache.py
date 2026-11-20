# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent PyTorch storage arithmetic for GLM 5.3 cache comparisons.

The latent record contains four groups of 128 E4M3 values, four FP32
scales, and 64 BF16 rotary values. Index keys use one FP32 scale per
128 values, and index queries use a power-of-two scale per head. Independent
PyTorch operations reproduce each of these three rounding boundaries.

The latent scale uses FP32 multiplication by the rounded constant 1/448.
The encoder uses a correctly rounded reciprocal. The GPU writer's approximate
reciprocal can change the stored byte near an E4M3 rounding midpoint.
"""

import torch


PRECISIONS = ("bf16", "fp8-latent", "fp8-latent-index")


def _check_values(value, width):
    if value.dtype != torch.bfloat16 or value.ndim < 2 or value.shape[-1] != width:
        raise ValueError(f"GLM reference storage requires BF16 values with trailing width {width}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("GLM reference storage requires finite values")


def encode_latent(latent, rotary):
    """Encode BF16 latent and rotary rows as 656-byte records, quantizing new rows immediately."""
    _check_values(latent, 512)
    _check_values(rotary, 64)
    if latent.shape[:-1] != rotary.shape[:-1]:
        raise ValueError("latent and rotary cache rows must have matching leading dimensions")
    groups = latent.float().reshape(*latent.shape[:-1], 4, 128)
    maximum = groups.abs().amax(-1)
    scales = maximum * torch.tensor(1 / 448, dtype=torch.float32, device=latent.device)
    scales = torch.where(maximum == 0, 1., scales)
    rounded = (groups * scales.reciprocal().unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return torch.cat((rounded.view(torch.uint8).flatten(-2), scales.contiguous().view(torch.uint8),
                      rotary.contiguous().view(torch.uint8)), dim=-1)


def decode_latent(records):
    """Decode stored records into FP32 latent values and BF16 rotary values without requantizing."""
    if records.dtype != torch.uint8 or records.shape[-1] != 656:
        raise ValueError("GLM reference latent records must contain 656 bytes")
    values = records[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = records[..., 512:528].contiguous().view(torch.float32)
    latent = (values.reshape(*values.shape[:-1], 4, 128) * scales.unsqueeze(-1)).flatten(-2)
    rotary = records[..., 528:].contiguous().view(torch.bfloat16)
    return latent, rotary


def quantize_index(values, *, query):
    """Return E4M3 bytes and FP32 scales for BF16 index keys or per-head queries.

    Both paths clamp the absolute maximum to 1e-4. Keys use a correctly
    rounded FP32 division by 448 and reciprocal. Query scales round up to
    powers of two. GLM 5.3 uses no Hadamard transform.
    """
    _check_values(values, 128)
    values = values.float()
    # CUDA tensor/scalar division can multiply by a rounded FP32 reciprocal.
    # The key writer rounds each division itself, including at E4M3 midpoints.
    maximum = values.abs().amax(-1).clamp_min(1e-4)
    scales = (maximum.double() / 448).float()
    if query:
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    inverse = scales.double().reciprocal().float()
    rounded = (values * inverse.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return rounded.view(torch.uint8), scales


def dequantize_index(values, scales):
    """Decode one independently quantized key or query head per scale."""
    return values.view(torch.float8_e4m3fn).float() * scales.unsqueeze(-1)


class ReferenceCache:
    """Append-only single-layer cache for canonical Transformers forwards.

    Each write stores its own quantized bytes and scales. Subsequent writes
    cannot change preceding records. Reads include just-written rows, with
    decoded latents rounded to BF16 before the canonical expansion projection.
    Transformers groups even and odd rotary coordinates into separate halves.
    Importing a native prefix requires de-interleaving the rotary key pairs
    and moving the index key's rotary coordinates ahead of its unrotated ones.
    This diagnostic cache does not implement native paging or serving state.
    """

    def __init__(self, precision):
        if precision not in PRECISIONS:
            raise ValueError(f"GLM reference cache precision must be one of {PRECISIONS}")
        self.precision = precision
        self.layer = None
        self.latent = self.rotary = self.records = self.index = self.index_scales = None

    def _layer(self, layer):
        if self.layer is not None and self.layer != layer:
            raise ValueError("a GLM reference cache belongs to one decoder layer")
        self.layer = layer

    @property
    def length(self):
        stored = self.latent if self.precision == "bf16" else self.records
        return 0 if stored is None else stored.shape[-2]

    def fork(self):
        """Share immutable prefix tensors until either branch appends to its own storage."""
        result = ReferenceCache(self.precision)
        result.layer = self.layer
        result.latent, result.rotary, result.records = self.latent, self.rotary, self.records
        result.index, result.index_scales = self.index, self.index_scales
        return result

    def update(self, latent, rotary, layer):
        self._layer(layer)
        _check_values(latent, 512)
        _check_values(rotary, 64)
        if self.precision == "bf16":
            self.latent = latent.clone() if self.latent is None else torch.cat((self.latent, latent), dim=-2)
            self.rotary = rotary.clone() if self.rotary is None else torch.cat((self.rotary, rotary), dim=-2)
            return self.latent, self.rotary
        records = encode_latent(latent, rotary)
        self.records = records if self.records is None else torch.cat((self.records, records), dim=-2)
        latent, rotary = decode_latent(self.records)
        return latent.to(torch.bfloat16), rotary

    def update_indexer(self, key, layer):
        self._layer(layer)
        if self.precision != "fp8-latent-index":
            self.index = key.clone() if self.index is None else torch.cat((self.index, key), dim=-2)
            return self.index
        values, scales = quantize_index(key, query=False)
        self.index = values if self.index is None else torch.cat((self.index, values), dim=-2)
        self.index_scales = scales if self.index_scales is None else torch.cat((self.index_scales, scales), dim=-1)
        return dequantize_index(self.index, self.index_scales)


class QuantizedIndexer(torch.nn.Module):
    """Canonical GLM indexer equations with explicit FP8 query rounding after rotary embedding.

    The wrapped Transformers module supplies checkpoint projections and
    normalization. Head weights retain its FP32 arithmetic. The cache owns
    key rounding independently. ``scores`` exposes the last call's masked
    scores for the independent cutoff check.
    """

    def __init__(self, canonical):
        super().__init__()
        self.canonical = canonical
        self.scores = None

    @torch.no_grad()
    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask, position_ids,
                past_key_values=None):
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import apply_rotary_pos_emb_interleave

        module = self.canonical
        batch, rows, _ = hidden_states.shape
        query = module.wq_b(q_resid).view(batch, rows, module.n_heads, module.head_dim)
        key = module.k_norm(module.wk(hidden_states)).unsqueeze(2)
        split = (module.qk_rope_head_dim, module.head_dim - module.qk_rope_head_dim)
        q_rot, q_pass = query.split(split, dim=-1)
        k_rot, k_pass = key.split(split, dim=-1)
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, *position_embeddings, unsqueeze_dim=2)
        query = torch.cat((q_rot, q_pass), dim=-1)
        key = torch.cat((k_rot, k_pass), dim=-1).squeeze(2)
        query = dequantize_index(*quantize_index(query, query=True))
        if past_key_values is None or past_key_values.precision != "fp8-latent-index":
            raise ValueError("quantized GLM indexing requires a cache with FP8 index keys")
        key = past_key_values.update_indexer(key, module.layer_idx)
        scores = torch.matmul(query, key.transpose(-1, -2).float().unsqueeze(1)) * module.softmax_scale
        scores = torch.relu(scores)
        weights = module.weights_proj(hidden_states.to(module.weights_proj.weight.dtype)).float() * (module.n_heads**-0.5)
        scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        self.scores = (scores.masked_fill(~attention_mask, float("-inf")) if attention_mask.dtype == torch.bool
                       else scores + attention_mask)
        return self.scores.topk(min(module.index_topk, self.scores.shape[-1]), dim=-1).indices.to(torch.int32)
