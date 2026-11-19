// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#define deep_gemm flare_mqa
#include "csrc/utils/exception.hpp"
#include <deep_gemm/scheduler/mega_moe.cuh>
#include "csrc/jit_kernels/impls/sm100_mqa_logits.hpp"
#include "csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp"
#undef deep_gemm

void initialize(const std::string& root) { flare_mqa::init_jit(root); }

void contiguous(torch::Tensor q, torch::Tensor qs, torch::Tensor keys,
                torch::Tensor key_scales, torch::Tensor weights,
                torch::Tensor starts, torch::Tensor ends, torch::Tensor output) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kInt8 && q.is_contiguous());
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(1) == 32 && q.size(2) == 64);
    const int rows = q.size(0);
    for (const auto& t : {qs, keys, key_scales, weights, starts, ends, output}) {
        TORCH_CHECK(t.device() == q.device() && t.is_contiguous());
    }
    TORCH_CHECK(qs.scalar_type() == torch::kInt && qs.sizes() == torch::IntArrayRef({rows, 32}));
    TORCH_CHECK(keys.dim() == 2 && keys.size(0) > 0 && keys.size(1) == 64);
    TORCH_CHECK(keys.scalar_type() == torch::kInt8);
    const int columns = keys.size(0);
    TORCH_CHECK(key_scales.scalar_type() == torch::kInt);
    TORCH_CHECK(key_scales.sizes() == torch::IntArrayRef({columns}));
    TORCH_CHECK(weights.scalar_type() == torch::kFloat);
    TORCH_CHECK(weights.sizes() == torch::IntArrayRef({rows, 32}));
    for (const auto& t : {starts, ends}) {
        TORCH_CHECK(t.scalar_type() == torch::kInt && t.sizes() == torch::IntArrayRef({rows}));
    }
    // Preserve the reference API's four-query and extra 256-key tile padding.
    const int stride = (columns + 511) / 256 * 256;
    TORCH_CHECK(output.scalar_type() == torch::kFloat);
    TORCH_CHECK(output.sizes() == torch::IntArrayRef({(rows + 3) / 4 * 4, stride}));
    flare_mqa::sm100_mqa_logits(q, qs, keys, key_scales, weights, starts, ends,
        output, torch::kFloat, rows, columns, 0, stride, 32, 128, 4, 256,
        false, true, torch::kInt8, std::nullopt);
}

std::vector<int64_t> sparse_sizes(int rows, int blocks, int block_tokens) {
    TORCH_CHECK(rows > 0 && rows <= (1 << 20));
    const auto metadata = flare_mqa::get_num_metadata_bytes(
        rows, blocks, block_tokens, 640, true, flare_mqa::runtime->get_num_sms());
    using namespace flare_mqa::layout::sparse_mqa_logits;
    return {metadata, sizeof(WorkspaceState) + rows * static_cast<int64_t>(sizeof(QBlockInfo))};
}

void sparse(torch::Tensor q, torch::Tensor qs, torch::Tensor cache,
            torch::Tensor weights, torch::Tensor lengths, torch::Tensor pages,
            torch::Tensor requests, torch::Tensor blocks, torch::Tensor metadata,
            torch::Tensor workspace, torch::Tensor output, int block_tokens) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kInt8 && q.is_contiguous());
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(1) == 32 && q.size(2) == 64);
    const int rows = q.size(0);
    for (const auto& t : {qs, cache, weights, lengths, pages, requests, blocks, metadata, workspace, output})
        TORCH_CHECK(t.device() == q.device());
    TORCH_CHECK(qs.scalar_type() == torch::kInt && qs.is_contiguous());
    TORCH_CHECK(qs.sizes() == torch::IntArrayRef({rows, 32}));
    TORCH_CHECK(cache.dim() == 3 && cache.scalar_type() == torch::kByte && cache.size(2) == 68);
    const int page_size = cache.size(1);
    TORCH_CHECK(page_size == 64 || page_size == 128);
    TORCH_CHECK(cache.size(0) > 0 && cache.stride(2) == 1 && cache.stride(1) == 68);
    TORCH_CHECK(cache.stride(0) >= page_size * 68 && cache.stride(0) % 512 == 0);
    TORCH_CHECK(weights.scalar_type() == torch::kBFloat16 && weights.is_contiguous());
    TORCH_CHECK(weights.sizes() == torch::IntArrayRef({rows, 32}));
    for (const auto& t : {lengths, requests}) {
        TORCH_CHECK(t.scalar_type() == torch::kInt && t.is_contiguous());
        TORCH_CHECK(t.sizes() == torch::IntArrayRef({rows}));
    }
    TORCH_CHECK(pages.scalar_type() == torch::kInt && pages.is_contiguous());
    TORCH_CHECK(pages.dim() == 2 && pages.size(0) == rows && pages.size(1) > 0);
    TORCH_CHECK(pages.stride(0) >= pages.size(1));
    TORCH_CHECK(blocks.scalar_type() == torch::kInt && blocks.is_contiguous());
    TORCH_CHECK(blocks.dim() == 2 && blocks.size(0) == rows);
    const auto sizes = sparse_sizes(rows, blocks.size(1), block_tokens);
    for (const auto& t : {metadata, workspace})
        TORCH_CHECK(t.scalar_type() == torch::kByte && t.is_contiguous() && t.dim() == 1);
    TORCH_CHECK(metadata.numel() >= sizes[0] && workspace.numel() >= sizes[1]);
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16 && output.is_contiguous());
    TORCH_CHECK(output.dim() == 2 && output.size(0) == (rows + 1) / 2 * 2);
    TORCH_CHECK(output.size(1) == (blocks.size(1) * block_tokens + 511) / 512 * 512);
    flare_mqa::launch_sm100_sparse_mqa_logits_metadata(true, false, page_size, 0,
        pages.stride(0), blocks, metadata, workspace, 640, block_tokens, nullptr, nullptr,
        lengths.data_ptr<int>(), pages.data_ptr<int>(), requests.data_ptr<int>());
    flare_mqa::launch_sm100_sparse_mqa_logits(true, false, block_tokens, q.unsqueeze(1),
        qs.unsqueeze(1), cache.unsqueeze(2), torch::Tensor(), weights, metadata,
        output.slice(0, 0, rows));
}

void paged(torch::Tensor q, torch::Tensor qs, torch::Tensor cache,
           torch::Tensor weights, torch::Tensor lengths, torch::Tensor pages,
           torch::Tensor schedule, torch::Tensor output) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kInt8 && q.is_contiguous());
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(1) == 32 && q.size(2) == 64);
    const int rows = q.size(0), heads = q.size(1);
    for (const auto& t : {qs, cache, weights, lengths, pages, schedule, output})
        TORCH_CHECK(t.device() == q.device());
    TORCH_CHECK(qs.scalar_type() == torch::kInt && qs.is_contiguous());
    TORCH_CHECK(qs.sizes() == torch::IntArrayRef({rows, heads}));
    TORCH_CHECK(cache.dim() == 3 && cache.scalar_type() == torch::kByte && cache.size(2) == 68);
    const int page_size = cache.size(1);
    TORCH_CHECK(page_size == 64 || page_size == 128);
    TORCH_CHECK(cache.size(0) > 0 && cache.stride(2) == 1 && cache.stride(1) == 68);
    TORCH_CHECK(cache.stride(0) >= page_size * 68 && cache.stride(0) % 512 == 0);
    TORCH_CHECK(weights.scalar_type() == torch::kFloat && weights.is_contiguous());
    TORCH_CHECK(weights.sizes() == torch::IntArrayRef({rows, heads}));
    TORCH_CHECK(lengths.scalar_type() == torch::kInt && lengths.is_contiguous());
    TORCH_CHECK(lengths.sizes() == torch::IntArrayRef({rows}));
    TORCH_CHECK(pages.scalar_type() == torch::kInt && pages.is_contiguous());
    TORCH_CHECK(pages.dim() == 2 && pages.size(0) == rows && pages.size(1) > 0);
    TORCH_CHECK(pages.stride(0) >= pages.size(1));
    const int sms = flare_mqa::runtime->get_num_sms();
    TORCH_CHECK(schedule.scalar_type() == torch::kInt && schedule.is_contiguous());
    TORCH_CHECK(schedule.sizes() == torch::IntArrayRef({sms + 1, 2}));
    TORCH_CHECK(output.scalar_type() == torch::kFloat && output.dim() == 2);
    TORCH_CHECK(output.size(0) == rows && output.size(1) > 0);
    TORCH_CHECK(output.size(1) <= (pages.size(1) * page_size + 255) / 256 * 256);
    TORCH_CHECK(output.is_contiguous());
    TORCH_CHECK(output.stride(0) % 256 == 0);
    const auto kv = cache.as_strided({cache.size(0), page_size, 64}, {cache.stride(0), 64, 1}).view(torch::kInt8);
    const auto sf = cache.view(torch::kInt).as_strided(
        {cache.size(0), page_size}, {cache.stride(0) / 4, 1}, cache.storage_offset() / 4 + page_size * 16);
    flare_mqa::sm100_paged_mqa_logits_metadata(lengths, schedule, rows, rows, 1, sms, true, false, nullptr);
    flare_mqa::sm100_paged_mqa_logits(q.unsqueeze(1), qs.unsqueeze(1), kv, sf, weights,
        lengths, output, pages, torch::Tensor(), schedule, torch::kFloat,
        rows, rows, 1, heads, 128, cache.size(0), page_size, true, false,
        output.stride(0), pages.stride(0), sms, 256, 16, true, torch::kInt8);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("initialize", &initialize);
    m.def("contiguous", &contiguous);
    m.def("paged", &paged);
    m.def("sparse_sizes", &sparse_sizes);
    m.def("sparse", &sparse);
}
