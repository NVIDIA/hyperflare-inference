// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KV_AVX2 __attribute__((target("avx2")))
#elif defined(__aarch64__) && !defined(__AARCH64EB__) && (defined(__GNUC__) || defined(__clang__))
#include <arm_neon.h>
#define KV_NEON
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#endif

static bool has_simd() {
#if defined(KV_AVX2)
    return __builtin_cpu_supports("avx2");
#elif defined(KV_NEON) && defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#elif defined(KV_NEON) && defined(__APPLE__)
    return true;
#else
    return false;
#endif
}

#if defined(KV_AVX2)
KV_AVX2 static inline __m256 bounded_vector(__m256 values) {
    __m256i words = _mm256_castps_si256(values);
    __m256i biased = _mm256_add_epi32(_mm256_slli_epi32(words, 1), _mm256_set1_epi32(0x80000000u - (77u << 24)));
    __m256i keep = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x80000000u + (101u << 24)), biased);
    return _mm256_castsi256_ps(_mm256_and_si256(words, keep));
}

KV_AVX2 static inline void divide_scales(U *values, int first, U divisor, int end) {
    __m256 d = _mm256_castsi256_ps(_mm256_set1_epi32(divisor));
    for (int i = first & ~7; i < end; i += 8) {
        __m256 v = _mm256_castsi256_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(values + i)));
        __m256 result = bounded_vector(_mm256_div_ps(v, d));
        _mm256_store_si256(reinterpret_cast<__m256i *>(values + i), _mm256_castps_si256(result));
    }
}

template <bool bound, bool bound_product = bound> KV_AVX2 static void update_simd(U *p, const U *v, int columns, U factor) {
    __m256 scale = _mm256_castsi256_ps(_mm256_set1_epi32(factor));
    for (int j = 0; j < columns; j += 8) {
        __m256 values = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v + j)));
        __m256 prior = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p + j)));
        __m256 product = _mm256_mul_ps(scale, values);
        __m256 result = _mm256_add_ps(prior, bound_product ? bounded_vector(product) : product);
        if (bound)
            result = bounded_vector(result);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(p + j), _mm256_castps_si256(result));
    }
}
#elif defined(KV_NEON)
static inline float32x4_t bounded_vector(float32x4_t values) {
    uint32x4_t words = vreinterpretq_u32_f32(values);
    uint32x4_t biased = vsubq_u32(vshlq_n_u32(words, 1), vdupq_n_u32(77u << 24));
    uint32x4_t keep = vcltq_u32(biased, vdupq_n_u32(101u << 24));
    return vreinterpretq_f32_u32(vandq_u32(words, keep));
}

static inline void divide_scales(U *values, int first, U divisor, int end) {
    float32x4_t d = vreinterpretq_f32_u32(vdupq_n_u32(divisor));
    for (int i = first & ~3; i < end; i += 4) {
        float32x4_t v = vreinterpretq_f32_u32(vld1q_u32(values + i));
        vst1q_u32(values + i, vreinterpretq_u32_f32(bounded_vector(vdivq_f32(v, d))));
    }
}

template <bool bound, bool bound_product = bound> static void update_simd(U *p, const U *v, int columns, U factor) {
    float32x4_t scale = vreinterpretq_f32_u32(vdupq_n_u32(factor));
#pragma GCC unroll 4
    for (int j = 0; j < columns; j += 4) {
        float32x4_t values = vreinterpretq_f32_u32(vld1q_u32(v + j));
        float32x4_t prior = vreinterpretq_f32_u32(vld1q_u32(p + j));
        float32x4_t product = vmulq_f32(scale, values);
        float32x4_t result = vaddq_f32(prior, bound_product ? bounded_vector(product) : product);
        if (bound)
            result = bounded_vector(result);
        vst1q_u32(p + j, vreinterpretq_u32_f32(result));
    }
}
#endif

#if defined(KV_AVX2) || defined(KV_NEON)
// Source and destination cover a complete eight-by-eight tile. Each store writes
// only the eight bytes belonging to this frame, including for interleaved views.
static inline void transpose_tile(const B *src, int stride, B *dst, int pitch) {
#if defined(KV_NEON)
    uint8x8x2_t pairs[4];
    for (int i = 0; i < 4; ++i)
        pairs[i] = vtrn_u8(vld1_u8(src + 2 * i * stride), vld1_u8(src + (2 * i + 1) * stride));
    for (int i = 0; i < 2; ++i) {
        uint16x4x2_t top = vtrn_u16(vreinterpret_u16_u8(pairs[0].val[i]), vreinterpret_u16_u8(pairs[1].val[i]));
        uint16x4x2_t bottom = vtrn_u16(vreinterpret_u16_u8(pairs[2].val[i]), vreinterpret_u16_u8(pairs[3].val[i]));
        for (int j = 0; j < 2; ++j) {
            uint32x2x2_t rows = vtrn_u32(vreinterpret_u32_u16(top.val[j]), vreinterpret_u32_u16(bottom.val[j]));
            vst1_u8(dst + (i + 2 * j) * pitch, vreinterpret_u8_u32(rows.val[0]));
            vst1_u8(dst + (i + 2 * j + 4) * pitch, vreinterpret_u8_u32(rows.val[1]));
        }
    }
#else
    __m128i pairs[4];
    for (int i = 0; i < 4; ++i) {
        __m128i a = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + 2 * i * stride));
        __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + (2 * i + 1) * stride));
        pairs[i] = _mm_unpacklo_epi8(a, b);
    }
    __m128i top[] = {_mm_unpacklo_epi16(pairs[0], pairs[1]), _mm_unpackhi_epi16(pairs[0], pairs[1])};
    __m128i bottom[] = {_mm_unpacklo_epi16(pairs[2], pairs[3]), _mm_unpackhi_epi16(pairs[2], pairs[3])};
    for (int i = 0; i < 2; ++i) {
        __m128i rows[] = {_mm_unpacklo_epi32(top[i], bottom[i]), _mm_unpackhi_epi32(top[i], bottom[i])};
        for (int j = 0; j < 2; ++j) {
            B *out = dst + (4 * i + 2 * j) * pitch;
            _mm_storel_epi64(reinterpret_cast<__m128i *>(out), rows[j]);
            _mm_storel_epi64(reinterpret_cast<__m128i *>(out + pitch), _mm_srli_si128(rows[j], 8));
        }
    }
#endif
}

static void transpose(const B *src, int stride, B *dst, int pitch, int rows, int columns) {
    for (int i = 0; i < rows; i += 8)
        for (int j = 0; j < columns; j += 8)
            transpose_tile(src + i * stride + j, stride, dst + j * pitch + i, pitch);
}
#endif
