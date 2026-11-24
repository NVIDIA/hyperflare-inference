// SPDX-License-Identifier: Apache-2.0
//
// GPU kernels for lossless GKV1 KV-cache checkpoint packets (VRAM <-> DRAM):
// - Bit-exact 64-pivot low-rank float32 predictor for 128x128 recurrent state matrices
//   (256-thread swizzled 64 KiB shared memory, 1 short CTA per matrix, IEEE-754 parity with cpu.cpp)
// - 4-lane aligned-u32 branchless residual low-bit pack/unpack with bounds & zero-pad validation
// - Byte-lane shuffle (width 2, 4, 20, 32) and compound page value/scale split-shuffle
// - Multi-plane cache gather/scatter kernels
//
// Compiled by Ignition at model core export via `nvcc --cubin` for the target SM architecture,
// embedded as a standalone cubin module in the exported core, and loaded lazily via `cuModuleLoadData`
// only when `--lossless-kv-compression gpu` is enabled (honors `CUDA_DISABLE_JIT=1`).

#include <stdint.h>
#include <stddef.h>

using U = uint32_t;
using B = uint8_t;

static constexpr int MAT_ELEMS = 16384;
static constexpr int PIVOTS = 64;
static constexpr int MAX_LANE_U32S = 3968;
static constexpr int LANE_STRIDE = MAX_LANE_U32S + 2;

__device__ __forceinline__ int swiz(int i, int j) {
    return i * 128 + (j ^ (i & 31));
}
__device__ __forceinline__ U bounded_u(U w) {
    U e = (w >> 23) & 255u;
    return (e >= 77u && e <= 177u) ? w : 0u;
}
__device__ __forceinline__ U order_u(U w) {
    return w ^ ((0u - (w >> 31)) | 0x80000000u);
}
__device__ __forceinline__ U residual_u(U w, U p) {
    U d = order_u(w) - order_u(p);
    return (d << 1) ^ (0u - (d >> 31));
}
__device__ __forceinline__ U restore_u(U z, U p) {
    U b = order_u(p) + ((z >> 1) ^ (0u - (z & 1u)));
    return b ^ (((b >> 31) - 1u) | 0x80000000u);
}

template <bool Decode>
__device__ __forceinline__ void predict_f32_128_impl(const uint64_t *raw_mat_ptrs, U *z_mats, int num_mats) {
    int m = blockIdx.x;
    if (m >= num_mats) return;
    int tid = threadIdx.x; // 0..255
    int j = tid & 127;
    int half = tid >> 7;

    extern __shared__ float p[]; // 128 * 128 floats = 65536 B
    __shared__ float v_sh[128];
    __shared__ float u_sh[128];
    __shared__ uint8_t done_sh[128];
    __shared__ int32_t warp_mag[4];
    __shared__ int32_t warp_idx[4];
    __shared__ int piv_col;
    __shared__ float piv_val;
    __shared__ U piv_bits;

    U *raw = reinterpret_cast<U *>(raw_mat_ptrs[m]);
    U *z = z_mats + (size_t)m * MAT_ELEMS;

    for (int i = half; i < 128; i += 2) p[swiz(i, j)] = 0.0f;
    if (half == 0) done_sh[j] = 0;
    __syncthreads();

    for (int k = 0; k < PIVOTS; ++k) {
        if (half == 0) {
            int32_t my_mag = -1;
            if (!done_sh[j]) {
                U pred = bounded_u(__float_as_uint(p[swiz(k, j)]));
                U w;
                if constexpr (Decode) {
                    w = restore_u(z[k * 128 + j], pred);
                    raw[j * 128 + k] = w;
                } else {
                    w = raw[j * 128 + k];
                    z[k * 128 + j] = residual_u(w, pred);
                }
                U obs = bounded_u(w);
                float diff = __fsub_rn(__uint_as_float(obs), __uint_as_float(pred));
                U dv = bounded_u(__float_as_uint(diff));
                v_sh[j] = __uint_as_float(dv);
                my_mag = (int32_t)(dv & 0x7fffffffu);
            } else {
                v_sh[j] = 0.0f;
            }
            int32_t best_m = my_mag, best_i = j;
            for (int off = 16; off > 0; off >>= 1) {
                int32_t om = __shfl_down_sync(0xffffffffu, best_m, off);
                int32_t oi = __shfl_down_sync(0xffffffffu, best_i, off);
                if (om > best_m || (om == best_m && oi < best_i)) {
                    best_m = om;
                    best_i = oi;
                }
            }
            if ((j & 31) == 0) {
                warp_mag[j >> 5] = best_m;
                warp_idx[j >> 5] = best_i;
            }
        }
        __syncthreads();
        if (tid == 0) {
            int32_t bm = warp_mag[0], bi = warp_idx[0];
            for (int w = 1; w < 4; ++w) {
                if (warp_mag[w] > bm || (warp_mag[w] == bm && warp_idx[w] < bi)) {
                    bm = warp_mag[w];
                    bi = warp_idx[w];
                }
            }
            piv_col = bi;
            piv_val = v_sh[bi];
            piv_bits = __float_as_uint(piv_val);
            done_sh[bi] = 1;
        }
        __syncthreads();

        int c = piv_col;
        if (half == 0 && j > k) {
            U pred = bounded_u(__float_as_uint(p[swiz(j, c)]));
            U w;
            if constexpr (Decode) {
                w = restore_u(z[j * 128 + c], pred);
                raw[c * 128 + j] = w;
            } else {
                w = raw[c * 128 + j];
                z[j * 128 + c] = residual_u(w, pred);
            }
            U obs = bounded_u(w);
            float diff = __fsub_rn(__uint_as_float(obs), __uint_as_float(pred));
            U delta = bounded_u(__float_as_uint(diff));
            u_sh[j] = piv_bits
                ? __uint_as_float(bounded_u(__float_as_uint(__fdiv_rn(__uint_as_float(delta), piv_val))))
                : 0.0f;
        }
        __syncthreads();

        if (!done_sh[j]) {
            float vj = v_sh[j];
            #pragma unroll 4
            for (int i = k + 1 + half; i < 128; i += 2) {
                int idx = swiz(i, j);
                p[idx] = __fmaf_rn(u_sh[i], vj, p[idx]);
            }
        }
        __syncthreads();
    }

    if (!done_sh[j]) {
        for (int i = PIVOTS + half; i < 128; i += 2) {
            U pred = bounded_u(__float_as_uint(p[swiz(i, j)]));
            if constexpr (Decode) {
                raw[j * 128 + i] = restore_u(z[i * 128 + j], pred);
            } else {
                z[i * 128 + j] = residual_u(raw[j * 128 + i], pred);
            }
        }
    }
}

extern "C" {

__global__ void kv_predict_f32_128_enc(const uint64_t *raw_mat_ptrs, U *z_mats, int num_mats) {
    predict_f32_128_impl<false>(raw_mat_ptrs, z_mats, num_mats);
}

__global__ void kv_predict_f32_128_dec(
    const uint64_t *raw_mat_ptrs,
    const U *z_mats,
    int num_mats,
    const uint32_t *err_flag
) {
    if (*err_flag != 0u) return;
    predict_f32_128_impl<true>(raw_mat_ptrs, const_cast<U *>(z_mats), num_mats);
}

__global__ void kv_pack_4lane_gpu(
    const U *z_mats,
    B *widths_out,
    U *low_padded_out,
    uint16_t *lane_u32s_out,
    int num_mats
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int m = gid >> 2;
    int l = gid & 3;
    if (m >= num_mats) return;

    const U *z = z_mats + (size_t)m * MAT_ELEMS;
    B *w_out = widths_out + (size_t)m * MAT_ELEMS;
    U *dst = low_padded_out + ((size_t)m * 4 + l) * LANE_STRIDE;

    uint64_t acc = 0;
    unsigned bits = 0;
    U *p = dst;
    for (int i = l; i < MAT_ELEMS; i += 4) {
        U v = z[i];
        unsigned w = v ? (32u - __clz(v)) : 0u;
        w_out[i] = (B)w;
        unsigned low = w - (w != 0u);
        acc |= (uint64_t)(v & ((1u << low) - 1u)) << bits;
        bits += low;
        if (bits >= 32u) {
            *p++ = (U)acc;
            acc >>= 32;
            bits -= 32u;
        }
    }
    if (bits) *p++ = (U)acc;
    lane_u32s_out[m * 4 + l] = (uint16_t)(p - dst);
}

static __device__ __forceinline__ U load_u32_le(const B *p) {
    return (U)p[0] | ((U)p[1] << 8) | ((U)p[2] << 16) | ((U)p[3] << 24);
}

// Compact padded low-bit lane buffers into one tight contiguous byte buffer:
// lane_byte_offsets[m*4 + l] gives the exact byte offset of lane (m, l) inside low_compact_out.
__global__ void kv_compact_low_lanes(
    const U *low_padded_in,
    const uint16_t *lane_u32s_in,
    const uint32_t *lane_byte_offsets,
    B *low_compact_out,
    int num_mats
) {
    int lane_id = blockIdx.x;
    if (lane_id >= num_mats * 4) return;
    uint32_t nwords = lane_u32s_in[lane_id];
    const U *src = low_padded_in + (size_t)lane_id * LANE_STRIDE;
    B *dst_b = low_compact_out + lane_byte_offsets[lane_id];
    if (((uintptr_t)dst_b & 3u) == 0u) {
        U *dst = reinterpret_cast<U *>(dst_b);
        for (uint32_t i = threadIdx.x; i < nwords; i += blockDim.x) {
            dst[i] = src[i];
        }
    } else {
        const B *src_b = reinterpret_cast<const B *>(src);
        size_t nbytes = (size_t)nwords * 4u;
        for (size_t i = threadIdx.x; i < nbytes; i += blockDim.x) {
            dst_b[i] = src_b[i];
        }
    }
}

// Bounds-checked 4-lane unpack directly from the compacted low-bit byte buffer.
// Sets *err_flag = 1 if any width > 32, bit count disagrees with lane_u32s, or trailing pad bits are non-zero.
__global__ void kv_unpack_4lane_gpu(
    U *z_mats,
    const B *widths_in,
    const B *low_compact_in,
    const uint32_t *lane_byte_offsets,
    const uint16_t *lane_u32s_in,
    int num_mats,
    uint32_t *err_flag
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int m = gid >> 2;
    int l = gid & 3;
    if (m >= num_mats) return;

    int lane_id = m * 4 + l;
    size_t nwords = lane_u32s_in[lane_id];
    if (nwords > (size_t)MAX_LANE_U32S) {
        atomicOr(err_flag, 1u);
        return;
    }
    const B *src_b = low_compact_in + lane_byte_offsets[lane_id];
    bool aligned = (((uintptr_t)src_b & 3u) == 0u);
    const U *src = aligned ? reinterpret_cast<const U *>(src_b) : nullptr;
    auto read_word = [&](size_t wi) -> U {
        return aligned ? src[wi] : load_u32_le(src_b + wi * 4u);
    };
    U *z = z_mats + (size_t)m * MAT_ELEMS;
    const B *w_in = widths_in + (size_t)m * MAT_ELEMS;

    size_t total_bits = nwords * 32u;
    size_t bit = 0;
    for (int i = l; i < MAT_ELEMS; i += 4) {
        unsigned w = w_in[i];
        if (w > 32u) {
            atomicOr(err_flag, 1u);
            return;
        }
        if (w == 0u) {
            z[i] = 0u;
            continue;
        }
        unsigned low = w - 1u;
        if (bit + low > total_bits) {
            atomicOr(err_flag, 1u);
            return;
        }
        size_t wi = bit >> 5;
        uint64_t q = (wi < nwords) ? (uint64_t)read_word(wi) : 0ull;
        if (wi + 1 < nwords) q |= ((uint64_t)read_word(wi + 1)) << 32;
        U hi = 1u << low;
        z[i] = hi | ((U)(q >> (bit & 31u)) & (hi - 1u));
        bit += low;
    }
    if (((bit + 31u) >> 5) != nwords) {
        atomicOr(err_flag, 1u);
        return;
    }
    unsigned rem = (unsigned)(bit & 31u);
    if (rem != 0u && nwords > 0u) {
        if ((read_word(nwords - 1) >> rem) != 0u) {
            atomicOr(err_flag, 1u);
        }
    }
}

__global__ void kv_byte_shuffle_gpu(B *dst, const B *src, size_t n, uint32_t width, int decode) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    size_t elems = n / width;
    if (!decode) {
        size_t i = idx / width, lane = idx % width;
        dst[lane * elems + i] = src[idx];
    } else {
        size_t lane = idx / elems, i = idx % elems;
        dst[i * width + lane] = src[idx];
    }
}

__global__ void kv_page_split_gpu(
    B *dst,
    const B *src,
    size_t n,
    size_t block,
    size_t rows,
    size_t cols,
    size_t stride,
    uint32_t scale_w,
    int planar,
    int decode
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    size_t b = idx / block;
    size_t off = idx % block;
    size_t nval = rows * cols;
    size_t sc_row = (block - nval) / rows;
    size_t ns = block - nval;
    size_t s_elems = scale_w > 0 ? (ns / scale_w) : 0;
    const B *sb = src + b * block;
    B *db = dst + b * block;
    if (!decode) {
        if (planar || sc_row == 0) {
            if (off < nval) {
                db[off] = sb[off];
            } else if (scale_w > 1) {
                size_t si = off - nval;
                size_t i = si / scale_w, lane = si % scale_w;
                db[nval + lane * s_elems + i] = sb[off];
            } else {
                db[off] = sb[off];
            }
        } else {
            size_t r = off / stride, c = off % stride;
            if (c < cols) {
                db[r * cols + c] = sb[off];
            } else {
                size_t si = r * sc_row + (c - cols);
                if (scale_w > 1) {
                    size_t i = si / scale_w, lane = si % scale_w;
                    db[nval + lane * s_elems + i] = sb[off];
                } else {
                    db[nval + si] = sb[off];
                }
            }
        }
    } else {
        if (planar || sc_row == 0) {
            if (off < nval) {
                db[off] = sb[off];
            } else if (scale_w > 1) {
                size_t si = off - nval;
                size_t lane = si / s_elems, i = si % s_elems;
                db[nval + i * scale_w + lane] = sb[off];
            } else {
                db[off] = sb[off];
            }
        } else {
            if (off < nval) {
                size_t r = off / cols, c = off % cols;
                db[r * stride + c] = sb[off];
            } else {
                size_t si = off - nval;
                size_t flat_s = si;
                if (scale_w > 1) {
                    size_t lane = si / s_elems, i = si % s_elems;
                    flat_s = i * scale_w + lane;
                }
                size_t r = flat_s / sc_row, sc = flat_s % sc_row;
                db[r * stride + cols + sc] = sb[off];
            }
        }
    }
}

// Gather/scatter multiple GPU cache plane spans into/from a contiguous staging buffer.
__global__ void kv_copy_span_u32(uint32_t *dst, const uint32_t *src, size_t nwords) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nwords) dst[i] = src[i];
}

// Coalesce/uncoalesce a single homogeneous lane slice (for copying transformed field planes into
// the contiguous ANS batch buffer on GPU without host involvement).
__global__ void kv_copy_bytes_u8(B *dst, const B *src, size_t nbytes) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nbytes) dst[i] = src[i];
}

// Batched device-to-device copy across arbitrary byte spans (1 CTA per span, 128 threads).
// Replaces hundreds of small cuMemcpyDtoDAsync / per-chunk cuMemcpyDtoHAsync calls.
__global__ void kv_copy_batch_gpu(
    const uint64_t *dst_ptrs,
    const uint64_t *src_ptrs,
    const uint64_t *lens,
    int n
) {
    int idx = blockIdx.x;
    if (idx >= n) return;
    B *dst = reinterpret_cast<B *>(dst_ptrs[idx]);
    const B *src = reinterpret_cast<const B *>(src_ptrs[idx]);
    size_t len = (size_t)lens[idx];
    if ((((uintptr_t)dst | (uintptr_t)src) & 3u) == 0u) {
        size_t nw = len >> 2;
        uint32_t *dw = reinterpret_cast<uint32_t *>(dst);
        const uint32_t *sw = reinterpret_cast<const uint32_t *>(src);
        for (size_t i = threadIdx.x; i < nw; i += blockDim.x) {
            dw[i] = sw[i];
        }
        size_t tail = nw << 2;
        for (size_t i = tail + threadIdx.x; i < len; i += blockDim.x) {
            dst[i] = src[i];
        }
    } else {
        for (size_t i = threadIdx.x; i < len; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

} // extern "C"
