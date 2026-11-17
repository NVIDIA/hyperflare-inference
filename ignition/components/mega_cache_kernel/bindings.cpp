// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <optional>

void fused_deepseek_v4_kv_rope_insert(
    torch::stable::Tensor const& kv, torch::stable::Tensor& k_cache,
    torch::stable::Tensor const& slot_mapping,
    torch::stable::Tensor const& position_ids,
    torch::stable::Tensor const& cos_sin_cache, int64_t cache_block_size,
    std::optional<torch::stable::Tensor> fp8_scale, bool kv_mxfp8);

void cp_gather_indexer_k_quant_cache(
    torch::stable::Tensor const& cache, torch::stable::Tensor& keys,
    torch::stable::Tensor& scales, torch::stable::Tensor const& pages,
    torch::stable::Tensor const& offsets);

STABLE_TORCH_LIBRARY_FRAGMENT(flare_reference_cache, ops) {
  ops.def("indexer_gather(Tensor cache, Tensor! keys, Tensor! scales, "
          "Tensor pages, Tensor offsets) -> ()");
  ops.def("rope_insert(Tensor kv, Tensor! k_cache, Tensor slot_mapping, "
          "Tensor position_ids, Tensor cos_sin_cache, int cache_block_size, "
          "Tensor? fp8_scale=None, bool kv_mxfp8=True) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(flare_reference_cache, CUDA, ops) {
  ops.impl("indexer_gather", TORCH_BOX(&cp_gather_indexer_k_quant_cache));
  ops.impl("rope_insert", TORCH_BOX(&fused_deepseek_v4_kv_rope_insert));
}
