// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "torch_utils.h"

// The gather kernel and launcher below are unchanged reference definitions.
template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) {
  return (a + b - 1) / b;
}

namespace vllm {
template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const char* __restrict__ kv_cache,  // [num_blocks, block_size,
                                        // cache_stride]
    char* __restrict__ dst_k,           // [num_tokens, head_dim]
    char* __restrict__ dst_scale,  // [num_tokens, head_dim / quant_block_size *
                                   // 4]
    const int* __restrict__ block_table,  // [batch_size, num_blocks]
    const int* __restrict__ cu_seq_lens,  // [batch_size + 1]
    const int batch_size,                 // batch size
    const int64_t token_stride,           // stride for each token in dst_k
    const int64_t head_dim,               // dimension of each head
    const int64_t block_stride,           // stride for each block in kv_cache
    const int64_t cache_token_stride,     // stride for each token in kv_cache
    const int64_t cache_block_size,  // num_tokens for each block in kv_cache
    const int num_blocks,            // number of blocks
    const int num_tokens,            // number of tokens
    const int quant_block_size       // quantization block size
) {
  constexpr int VEC_SIZE = sizeof(float4) / sizeof(char);
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int head_idx = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
  // Find batch index within a block
  __shared__ int batch_idx[BLOCK_Y_SIZE];
  if (threadIdx.x == 0) {
    batch_idx[threadIdx.y] = -1;
  }
  __syncthreads();

  for (int iter = 0; iter < div_ceil(batch_size, int(blockDim.x)); iter++) {
    int tid = iter * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
      const int seq_start = cu_seq_lens[tid];
      const int seq_end = cu_seq_lens[tid + 1];
      if (token_idx >= seq_start && token_idx < seq_end) {
        batch_idx[threadIdx.y] = tid;
      }
    }
  }

  __syncthreads();

  // num_tokens may be an allocation upper bound when Python avoids a D2H sync.
  // Only tokens covered by the exact device-side cu_seq_lens are valid to
  // gather.
  const int batch = batch_idx[threadIdx.y];
  if (head_idx >= head_dim || token_idx >= num_tokens || batch < 0) {
    return;
  }
  const int inbatch_seq_idx = token_idx - cu_seq_lens[batch];
  const int block_idx =
      block_table[batch * num_blocks + inbatch_seq_idx / cache_block_size];
  const int64_t src_block_offset = block_idx * block_stride;
  const int64_t cache_inblock_offset =
      (inbatch_seq_idx % cache_block_size) * head_dim + head_idx;
  const int64_t src_inblock_offset = src_block_offset + cache_inblock_offset;
  const int64_t dst_inblock_offset = token_idx * token_stride + head_idx;

  reinterpret_cast<float4*>(dst_k)[dst_inblock_offset / VEC_SIZE] =
      reinterpret_cast<const float4*>(kv_cache)[src_inblock_offset / VEC_SIZE];
  ;
  if (threadIdx.x == 0) {
    const int64_t src_scale_offset =
        src_block_offset + cache_block_size * head_dim +
        cache_inblock_offset * 4 / quant_block_size;
    reinterpret_cast<float*>(dst_scale)[dst_inblock_offset / quant_block_size] =
        reinterpret_cast<const float*>(kv_cache)[src_scale_offset / 4];
  }
}

}  // namespace vllm

// Macro to dispatch the kernel based on the data amount.
#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)                    \
  vllm::cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>                  \
      <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,                 \
              (head_dim + 8 * vec_size - 1) / (8 * vec_size)),                \
         dim3(8, BLOCK_Y_SIZE), 0, stream>>>(                                 \
          reinterpret_cast<char*>(kv_cache.data_ptr()),                       \
          reinterpret_cast<char*>(dst_k.data_ptr()),                          \
          reinterpret_cast<char*>(dst_scale.data_ptr()),                      \
          block_table.const_data_ptr<int32_t>(),                              \
          cu_seq_lens.const_data_ptr<int32_t>(), batch_size, dst_k.stride(0), \
          dst_k.size(1), kv_cache.stride(0), kv_cache.stride(1),              \
          kv_cache.size(1), block_table.size(1), num_tokens,                  \
          quant_block_size);

void cp_gather_indexer_k_quant_cache(
    const torch::stable::Tensor&
        kv_cache,                  // [num_blocks, block_size, cache_stride]
    torch::stable::Tensor& dst_k,  // [num_tokens, head_dim]
    torch::stable::Tensor&
        dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::stable::Tensor& block_table,  // [batch_size, num_blocks]
    const torch::stable::Tensor& cu_seq_lens   // [batch_size + 1]
) {
  int batch_size = block_table.size(0);
  int num_tokens = dst_k.size(0);
  int head_dim = dst_k.size(1);
  int quant_block_size = head_dim * 4 / dst_scale.size(1);

  STD_TORCH_CHECK(kv_cache.device() == dst_k.device(),
                  "kv_cache and dst_k must be on the same device");
  STD_TORCH_CHECK(kv_cache.device() == dst_scale.device(),
                  "kv_cache and dst_scale must be on the same device");
  STD_TORCH_CHECK(kv_cache.device() == block_table.device(),
                  "kv_cache and block_table must be on the same device");
  STD_TORCH_CHECK(kv_cache.device() == cu_seq_lens.device(),
                  "kv_cache and cu_seq_lens must be on the same device");
  STD_TORCH_CHECK(head_dim % quant_block_size == 0,
                  "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 16;
  const torch::stable::accelerator::DeviceGuard device_guard(
      kv_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  if (num_tokens < 32) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
  } else if (num_tokens < 64) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
  } else if (num_tokens < 128) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
  } else if (num_tokens < 256) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
  } else if (num_tokens < 512) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
  } else {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(32);
  }
}
