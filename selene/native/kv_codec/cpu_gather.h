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

#ifdef KV_AVX2
template <int scale>
KV_AVX2 static inline __m256i lookup_words(const B *data, __m256i indices) {
    alignas(32) U index[8];
    _mm256_store_si256(reinterpret_cast<__m256i *>(index), indices);
    return _mm256_setr_epi32(load32(data + scale * index[0]), load32(data + scale * index[1]),
                             load32(data + scale * index[2]), load32(data + scale * index[3]),
                             load32(data + scale * index[4]), load32(data + scale * index[5]),
                             load32(data + scale * index[6]), load32(data + scale * index[7]));
}
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
        // One four-byte lookahead covers two symbols even when both need
        // two normalization bytes. The frame directory precedes every lane.
        for (int base = 0; base < n; base += 2 * stride) {
            __m256i lookahead[2];
            for (int j = 0; j < groups; ++j)
                lookahead[j] = lookup_words<1>(packet, cursor[j]);
            for (int step = 0; step < 2 && base + step * stride < n; ++step) {
                int row = base + step * stride;
                int group = channel_shift ? classes[(row + group_start) >> channel_shift] : 0;
                const U *model = tables + (group & 3) * 4096;
                for (int j = 0; j < groups; ++j) {
                    __m256i state = current[j];
                    __m256i offset = cursor[j];
                    __m256i mask = _mm256_set1_epi32(4095);
                    __m256i entry = lookup_words<4>(reinterpret_cast<const B *>(model), _mm256_and_si256(state, mask));
                    __m256i frequency = _mm256_add_epi32(_mm256_srli_epi32(entry, 20), _mm256_set1_epi32(1));
                    __m256i bias = _mm256_and_si256(_mm256_srli_epi32(entry, 8), mask);
                    state = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(state, 12), frequency), bias);
                    __m256i need_one = _mm256_cmpgt_epi32(_mm256_set1_epi32(L), state);
                    __m256i need_two = _mm256_cmpgt_epi32(_mm256_set1_epi32(L >> 8), state);
                    __m256i count = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_setzero_si256(), need_one), need_two);
                    __m256i bits = _mm256_slli_epi32(count, 3);
                    __m256i bytes = _mm256_srlv_epi32(lookahead[j], _mm256_sub_epi32(_mm256_set1_epi32(32), bits));
                    state = _mm256_or_si256(_mm256_sllv_epi32(state, bits), bytes);
                    lookahead[j] = _mm256_sllv_epi32(lookahead[j], bits);
                    offset = _mm256_max_epi32(_mm256_sub_epi32(offset, count), _mm256_setzero_si256());
                    current[j] = state;
                    cursor[j] = offset;
                    __m256i symbols = _mm256_shuffle_epi8(entry, _mm256_set1_epi32(0x0c080400));
                    __m128i pair = _mm_unpacklo_epi32(_mm256_castsi256_si128(symbols), _mm256_extracti128_si256(symbols, 1));
                    pair = _mm_xor_si128(pair, _mm_set1_epi8((group >> 2) * 128));
                    _mm_storel_epi64(reinterpret_cast<__m128i *>(out + row + group_start + j * 8), pair);
                }
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

// The entropy streams have been consumed. Each lane's remaining low bits end
// before its four-byte terminal state, so a word read at that boundary fits.
#ifdef KV_AVX2
KV_AVX2
#else
KV_SVE2
#endif
static bool gather_residuals(U *out, const B *widths, int n, int stride, const DecodeLane *lanes, const B *packet) {
    alignas(32) U starts[128], lengths[128];
    for (int i = 0; i < stride; ++i) {
        starts[i] = U(lanes[i].low - packet);
        lengths[i] = U(lanes[i].high - lanes[i].low);
    }
#ifdef KV_AVX2
    constexpr int width = 8;
#else
    int width = int(svcntw());
#endif
    for (int col = 0; col < stride; col += width) {
#ifdef KV_AVX2
        __m256i begin = _mm256_load_si256(reinterpret_cast<const __m256i *>(starts + col));
        __m256i length = _mm256_load_si256(reinterpret_cast<const __m256i *>(lengths + col));
        __m256i bit = _mm256_setzero_si256();
        const __m256i one = _mm256_set1_epi32(1);
        for (int i = col; i < n; i += stride) {
            __m256i symbol = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(widths + i)));
            __m256i count = _mm256_max_epi32(_mm256_sub_epi32(symbol, one), _mm256_setzero_si256());
            __m256i offset = _mm256_min_epu32(_mm256_srli_epi32(bit, 3), length);
            __m256i shift = _mm256_and_si256(bit, _mm256_set1_epi32(7));
            __m256i index = _mm256_add_epi32(begin, offset);
            __m256i bits = _mm256_srlv_epi32(lookup_words<1>(packet, index), shift);
            __m256i limit = _mm256_set1_epi32(32);
            if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(_mm256_add_epi32(count, shift), limit))) {
                __m256i next = _mm256_add_epi32(begin, _mm256_min_epu32(_mm256_add_epi32(offset, one), length));
                __m256i extra = _mm256_srli_epi32(lookup_words<1>(packet, next), 24);
                bits = _mm256_or_si256(bits, _mm256_sllv_epi32(extra, _mm256_sub_epi32(limit, shift)));
            }
            __m256i leading = _mm256_sllv_epi32(one, count);
            __m256i value = _mm256_or_si256(leading, _mm256_and_si256(bits, _mm256_sub_epi32(leading, one)));
            value = _mm256_andnot_si256(_mm256_cmpeq_epi32(symbol, _mm256_setzero_si256()), value);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), value);
            bit = _mm256_add_epi32(bit, count);
        }
        __m256i bytes = _mm256_srli_epi32(_mm256_add_epi32(bit, _mm256_set1_epi32(7)), 3);
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(bytes, length)) != -1)
            return false;
#else
        svbool_t pg = svwhilelt_b32(col, stride);
        svuint32_t begin = svld1_u32(pg, starts + col);
        svuint32_t length = svld1_u32(pg, lengths + col);
        svuint32_t bit = svdup_u32(0);
        for (int i = col; i < n; i += stride) {
            svuint32_t symbol = svld1ub_u32(pg, widths + i);
            svuint32_t count = svqsub_n_u32_x(pg, symbol, 1);
            svuint32_t offset = svmin_u32_x(pg, svlsr_n_u32_x(pg, bit, 3), length);
            svuint32_t shift = svand_n_u32_x(pg, bit, 7);
            svuint32_t index = svadd_u32_x(pg, begin, offset);
            svuint32_t bits = svld1_gather_u32offset_u32(pg, reinterpret_cast<const U *>(packet), index);
            bits = svlsr_u32_x(pg, bits, shift);
            svbool_t more = svcmpgt_n_u32(pg, svadd_u32_x(pg, count, shift), 32);
            if (svptest_any(pg, more)) {
                svuint32_t next = svadd_u32_x(pg, begin, svmin_u32_x(pg, svadd_n_u32_x(pg, offset, 4), length));
                svuint32_t extra = svld1ub_gather_u32offset_u32(more, packet, next);
                bits = svorr_u32_x(pg, bits, svlsl_u32_x(pg, extra, svsubr_n_u32_x(pg, shift, 32)));
            }
            svuint32_t leading = svlsl_u32_x(pg, svdup_u32(1), count);
            svuint32_t value = svorr_u32_x(pg, leading, svand_u32_x(pg, bits, svsub_n_u32_x(pg, leading, 1)));
            svst1_u32(pg, out + i, svsel_u32(svcmpeq_n_u32(pg, symbol, 0), svdup_u32(0), value));
            bit = svadd_u32_x(pg, bit, count);
        }
        svuint32_t bytes = svlsr_n_u32_x(pg, svadd_n_u32_x(pg, bit, 7), 3);
        if (svptest_any(pg, svcmpne_u32(pg, bytes, length)))
            return false;
#endif
    }
    return true;
}
#endif
