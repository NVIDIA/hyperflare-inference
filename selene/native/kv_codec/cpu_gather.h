// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(KV_NEON) && defined(__linux__) && __has_include(<arm_sve.h>)
#include <arm_sve.h>
#if defined(HWCAP2_SVE2)
#define KV_SVE2 __attribute__((target("arch=armv8.5-a+sve2")))
static bool has_sve2() {
    return (getauxval(AT_HWCAP2) & HWCAP2_SVE2) != 0;
}
#endif
#endif

#if defined(KV_AVX2) || defined(KV_SVE2)
// Read the two possible normalization bytes from a four-byte load ending at
// the lane cursor. The first cursor is beyond the checked frame directory, so
// offset zero (a load at the header) is already below every valid lane. Saturate
// a malformed cursor there and reject any lane that didn't finish at its start.
template <int channel_shift, bool matrix>
#if defined(KV_AVX2)
KV_AVX2
#else
KV_SVE2
#endif
static bool gather_decode(B *out, int n, int stride, DecodeLane *lanes,
                          const U *tables, const B *classes, const B *packet) {
    alignas(64) U states[128], offsets[128];
    for (int j = 0; j < stride; ++j) {
        states[j] = lanes[j].state;
        offsets[j] = U(lanes[j].high - packet - 4);
    }
#if defined(KV_AVX2)
    // Keep two independent groups in registers across successive rANS steps.
    const int groups = stride == 8 ? 1 : 2;
    for (int group_start = 0; group_start < stride; group_start += 8 * groups) {
        __m256i current[2], cursor[2];
        for (int j = 0; j < groups; ++j) {
            int first = group_start + j * 8;
            current[j] = _mm256_load_si256(reinterpret_cast<const __m256i *>(states + first));
            cursor[j] = _mm256_load_si256(reinterpret_cast<const __m256i *>(offsets + first));
        }
        for (int row = 0; row < n; row += stride) {
            int group = channel_shift ? classes[(row + group_start) >> channel_shift] : 0;
            const U *model = tables + (group & 3) * 4096;
            for (int j = 0; j < groups; ++j) {
                __m256i state = current[j];
                __m256i offset = cursor[j];
                __m256i mask = _mm256_set1_epi32(4095);
                __m256i entry = _mm256_i32gather_epi32(reinterpret_cast<const int *>(model), _mm256_and_si256(state, mask), 4);
                __m256i frequency = _mm256_add_epi32(_mm256_srli_epi32(entry, 20), _mm256_set1_epi32(1));
                __m256i bias = _mm256_and_si256(_mm256_srli_epi32(entry, 8), mask);
                state = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(state, 12), frequency), bias);
                __m256i need_one = _mm256_cmpgt_epi32(_mm256_set1_epi32(L), state);
                __m256i need_two = _mm256_cmpgt_epi32(_mm256_set1_epi32(L >> 8), state);
                __m256i count = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_setzero_si256(), need_one), need_two);
                __m256i bits = _mm256_slli_epi32(count, 3);
                __m256i bytes = _mm256_srli_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int *>(packet), offset, 1), 16);
                state = _mm256_or_si256(_mm256_sllv_epi32(state, bits),
                                        _mm256_srlv_epi32(bytes, _mm256_sub_epi32(_mm256_set1_epi32(16), bits)));
                offset = _mm256_max_epi32(_mm256_sub_epi32(offset, count), _mm256_setzero_si256());
                current[j] = state;
                cursor[j] = offset;
                __m256i symbols = _mm256_shuffle_epi8(entry, _mm256_set1_epi32(0x0c080400));
                __m128i pair = _mm_unpacklo_epi32(_mm256_castsi256_si128(symbols), _mm256_extracti128_si256(symbols, 1));
                pair = _mm_xor_si128(pair, _mm_set1_epi8((group >> 2) * 128));
                _mm_storel_epi64(reinterpret_cast<__m128i *>(out + row + group_start + j * 8), pair);
            }
        }
        for (int j = 0; j < groups; ++j) {
            int first = group_start + j * 8;
            _mm256_store_si256(reinterpret_cast<__m256i *>(states + first), current[j]);
            _mm256_store_si256(reinterpret_cast<__m256i *>(offsets + first), cursor[j]);
        }
    }
#else
    for (int row = 0; row < n; row += stride) {
        int limit = std::min(stride, n - row);
        for (int col = 0; col < limit;) {
            int group = channel_shift ? classes[(row + col) >> channel_shift] : 0;
            const U *model = tables + (group & 3) * 4096;
            int chunk = channel_shift ? std::min(limit, col + (1 << channel_shift)) : limit;
            int width = int(svcntw());
            for (; col < chunk; col += width) {
                svbool_t pg = svwhilelt_b32(col, chunk);
                svuint32_t state = svld1_u32(pg, states + col);
                svuint32_t offset = svld1_u32(pg, offsets + col);
                svuint32_t entry = svld1_gather_u32index_u32(pg, model, svand_n_u32_x(pg, state, 4095));
                svuint32_t freq = svadd_n_u32_x(pg, svlsr_n_u32_x(pg, entry, 20), 1);
                svuint32_t bias = svand_n_u32_x(pg, svlsr_n_u32_x(pg, entry, 8), 4095);
                state = svmla_u32_x(pg, bias, freq, svlsr_n_u32_x(pg, state, 12));
                svst1b_u32(pg, out + row + col, sveor_n_u32_x(pg, entry, (group >> 2) * 128));
                svuint32_t count = svlsr_n_u32_x(pg, svsub_n_u32_x(pg, svclz_u32_x(pg, state), 1), 3);
                svuint32_t bits = svlsl_n_u32_x(pg, count, 3);
                svuint32_t bytes = svld1_gather_u32offset_u32(pg, reinterpret_cast<const U *>(packet), offset);
                bytes = svlsr_n_u32_x(pg, bytes, 16);
                bytes = svlsr_u32_x(pg, bytes, svsubr_n_u32_x(pg, bits, 16));
                state = svorr_u32_x(pg, svlsl_u32_x(pg, state, bits), bytes);
                svst1_u32(pg, states + col, state);
                svst1_u32(pg, offsets + col, svqsub_u32_x(pg, offset, count));
            }
            col = chunk;
        }
    }
#endif
    for (int j = 0; j < stride; ++j) {
        U begin = U(lanes[j].low - packet - 4);
        if (states[j] != L || (matrix ? offsets[j] < begin : offsets[j] != begin))
            return false;
        lanes[j].high = packet + 4 + offsets[j];
    }
    return true;
}
#endif
