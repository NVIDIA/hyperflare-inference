// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0

// Host binding for unchanged FlashMLA instantiations; all scratch is caller-owned.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.h"
namespace fm = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;
using Tensor = torch::Tensor;
using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;

void prefill(const Tensor& q, const Tensor& kv, const Tensor& indices,
             const Tensor& lengths, const Tensor& sink, const Tensor& positions,
             const Tensor& cs, Tensor& out, Tensor& sf, Tensor& max_logits,
             Tensor& lse, double scale) {
    c10::cuda::CUDAGuard guard(q.device());
    auto props = at::cuda::getCurrentDeviceProperties();
    fm::ParamT<SparseAttnFwdMode::Prefill> p{};
    p.s_q=q.size(0); p.s_kv=kv.size(0); p.h_q=64; p.h_kv=1; p.d_qk=512; p.d_v=512; p.topk=indices.size(2);
    p.sm_scale=scale; p.sm_scale_div_log2=scale * 1.44269504f;
    p.q=(bf16*)q.data_ptr(); p.kv=(bf16*)kv.data_ptr(); p.indices=(int*)indices.data_ptr();
    p.attn_sink=(float*)sink.data_ptr(); p.topk_length=(int*)lengths.data_ptr();
    p.stride_q_s_q=q.stride(0); p.stride_q_h_q=q.stride(1);
    p.stride_kv_s_kv=kv.stride(0); p.stride_kv_h_kv=kv.stride(1);
    p.stride_indices_s_q=indices.stride(0); p.stride_indices_h_kv=indices.stride(1);
    p.max_logits=(float*)max_logits.data_ptr(); p.lse=(float*)lse.data_ptr();
    p.num_sm=props->multiProcessorCount; p.stream=at::cuda::getCurrentCUDAStream();
    p.enable_q_norm=false; p.rms_norm_eps=0; p.token_positions=(uint32_t*)positions.data_ptr();
    p.is_rope_neox_style=false; p.rope_dim=64; p.cos_sin_cache=(float*)cs.data_ptr();
    p.n_wv_group=8; p.wv_group_size=8; p.num_per_channels=32;
    p.use_tma_aligned_col_major_sf=true; p.round_sf=true; p.use_packed_ue8m0=true;
    p.out_fp8=(fp8*)out.data_ptr(); p.out_sf=(uint32_t*)sf.data_ptr();
    p.stride_out_sf_wv_group=sf.stride(1); p.stride_out_sf_head_dim=sf.stride(2);
    fm::run_fused_norm_rope_attn_rope_cast_fwd_kernel<fm::Config{SparseAttnFwdMode::Prefill, ModelType::V4, ModelType::V4, 64, false}>(p);
}
void decode(const Tensor& q, const Tensor& kv, const Tensor& indices,
            const Tensor& lengths, const Tensor& sink, const Tensor& positions,
            const Tensor& cs, const std::optional<Tensor>& extra,
            const std::optional<Tensor>& extra_indices, const std::optional<Tensor>& extra_lengths,
            Tensor& out, Tensor& sf, Tensor& lse, double scale) {
    c10::cuda::CUDAGuard guard(q.device());
    fm::ParamT<SparseAttnFwdMode::Decode> p{};
    p.b=1; p.s_q=q.size(0); p.h_q=64; p.h_kv=1; p.d_qk=512; p.d_v=512;
    p.sm_scale=scale; p.sm_scale_div_log2=scale * 1.44269504f;
    p.num_blocks=kv.size(0); p.page_block_size=kv.size(1); p.topk=indices.size(1);
    p.model_type=ModelType::V41; p.extra_model_type=extra ? ModelType::V41_FP4 : ModelType::V41;
    p.q=(bf16*)q.data_ptr(); p.kv=(bf16*)kv.data_ptr(); p.indices=(int*)indices.data_ptr();
    p.attn_sink=(float*)sink.data_ptr(); p.topk_length=(int*)lengths.data_ptr(); p.lse=(float*)lse.data_ptr();
    p.stride_q_s_q=q.stride(0); p.stride_q_h_q=q.stride(1);
    p.stride_kv_block=kv.stride(0); p.stride_kv_row=kv.stride(1);
    p.stride_indices_s_q=indices.stride(0); p.stride_lse_s_q=lse.stride(0);
    if (extra) {
        p.extra_num_blocks=extra->size(0); p.extra_page_block_size=extra->size(1); p.extra_topk=extra_indices->size(1);
        p.extra_kv=(bf16*)extra->data_ptr(); p.extra_indices=(int*)extra_indices->data_ptr();
        p.extra_topk_length=(int*)extra_lengths->data_ptr();
        p.stride_extra_kv_block=extra->stride(0); p.stride_extra_kv_row=extra->stride(1);
        p.stride_extra_indices_s_q=extra_indices->stride(0);
    }
    p.stream=at::cuda::getCurrentCUDAStream();
    p.enable_q_norm=false; p.rms_norm_eps=0; p.token_positions=(uint32_t*)positions.data_ptr();
    p.is_rope_neox_style=false; p.rope_dim=64; p.cos_sin_cache=(float*)cs.data_ptr();
    p.n_wv_group=8; p.wv_group_size=8; p.num_per_channels=32;
    p.use_tma_aligned_col_major_sf=true; p.round_sf=true; p.use_packed_ue8m0=true;
    p.out_fp8=(fp8*)out.data_ptr(); p.out_sf=(uint32_t*)sf.data_ptr();
    p.stride_out_sf_wv_group=sf.stride(1); p.stride_out_sf_head_dim=sf.stride(2);
    if (extra) {
        fm::run_fused_norm_rope_attn_rope_cast_fwd_kernel<fm::Config{SparseAttnFwdMode::Decode, ModelType::V41, ModelType::V41_FP4, 64, false}>(p);
    } else {
        fm::run_fused_norm_rope_attn_rope_cast_fwd_kernel<fm::Config{SparseAttnFwdMode::Decode, ModelType::V41, ModelType::V41, 64, false}>(p);
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefill", &prefill);
    m.def("decode", &decode);
}
