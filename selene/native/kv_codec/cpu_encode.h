// SPDX-License-Identifier: Apache-2.0
#pragma once

// Each lane has up to seven pending bits. The four-byte store can reach its
// terminal state, which is written after all low-bit streams are complete.
#ifdef KV_AVX2
KV_AVX2
#endif
static void put_matrix_bits(B **output, const B *src, const B *widths, int n, int first, int stride, int lane_count) {
    const U *words = reinterpret_cast<const U *>(src);
#ifdef KV_AVX2
    constexpr int width = 8;
#else
    constexpr int width = 4;
#endif
    for (int base = 0; base < lane_count; base += width) {
        U packed[width], advance[width], trailing[width];
#ifdef KV_AVX2
        __m256i shift = _mm256_setzero_si256();
        __m256i carry = shift;
#else
        uint32x4_t shift = vdupq_n_u32(0);
        uint32x4_t carry = shift;
#endif
        for (int i = first + base; i < n; i += stride) {
#ifdef KV_AVX2
            __m256i sizes = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(widths + i)));
            __m256i count = _mm256_max_epi32(_mm256_sub_epi32(sizes, _mm256_set1_epi32(1)), _mm256_setzero_si256());
            __m256i leading = _mm256_sllv_epi32(_mm256_set1_epi32(1), count);
            __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(words + i));
            __m256i bits = _mm256_and_si256(raw, _mm256_sub_epi32(leading, _mm256_set1_epi32(1)));
            __m256i low = _mm256_or_si256(carry, _mm256_sllv_epi32(bits, shift));
            __m256i high = _mm256_srlv_epi32(bits, _mm256_sub_epi32(_mm256_set1_epi32(32), shift));
            __m256i total = _mm256_add_epi32(shift, count);
            __m256i complete = _mm256_and_si256(total, _mm256_set1_epi32(~7));
            carry = _mm256_or_si256(_mm256_srlv_epi32(low, complete), high);
            shift = _mm256_and_si256(total, _mm256_set1_epi32(7));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(packed), low);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(advance), _mm256_srli_epi32(complete, 3));
#else
            U four_widths = load32(widths + i);
            uint8x8_t bytes = vreinterpret_u8_u32(vdup_n_u32(four_widths));
            uint32x4_t sizes = vmovl_u16(vget_low_u16(vmovl_u8(bytes)));
            uint32x4_t count = vqsubq_u32(sizes, vdupq_n_u32(1));
            uint32x4_t leading = vshlq_u32(vdupq_n_u32(1), vreinterpretq_s32_u32(count));
            uint32x4_t bits = vandq_u32(vld1q_u32(words + i), vsubq_u32(leading, vdupq_n_u32(1)));
            uint32x4_t low = vorrq_u32(carry, vshlq_u32(bits, vreinterpretq_s32_u32(shift)));
            uint32x4_t high = vshlq_u32(bits, vreinterpretq_s32_u32(vsubq_u32(shift, vdupq_n_u32(32))));
            uint32x4_t total = vaddq_u32(shift, count);
            uint32x4_t complete = vbicq_u32(total, vdupq_n_u32(7));
            carry = vorrq_u32(vshlq_u32(low, vnegq_s32(vreinterpretq_s32_u32(complete))), high);
            shift = vandq_u32(total, vdupq_n_u32(7));
            vst1q_u32(packed, low);
            vst1q_u32(advance, vshrq_n_u32(complete, 3));
#endif
            for (int lane = 0; lane < width; ++lane) {
                store32(output[base + lane], packed[lane]);
                output[base + lane] += advance[lane];
            }
        }
#ifdef KV_AVX2
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(trailing), carry);
#else
        vst1q_u32(trailing, carry);
#endif
        for (int lane = 0; lane < width; ++lane)
            *output[base + lane] = B(trailing[lane]);
    }
}

// Eight independent rANS states share the symbol loads. Each lane retains its
// own byte stream, in the order consumed by the scalar decoder.
template <int channel_shift, bool matrix>
#ifdef KV_AVX2
KV_AVX2
#endif
static int put_vectors(B *out, int capacity, const B *src, int n, int first, int stride,
                       const uint64_t *entries, const B *classes, B *lengths,
                       const B *widths, const int *low_bits) {
    const int groups = stride == 8 ? 1 : 2;
    // Stagger the lane buffers by a cache line.
    B streams[16][2 * MAX_LANE_SYMBOLS + 64];
    B *next[16];
    for (int lane = 0; lane < groups * 8; ++lane)
        next[lane] = streams[lane];
#ifdef KV_AVX2
    __m256i states[2] = {_mm256_set1_epi32(L), _mm256_set1_epi32(L)};
    const __m256i mask = _mm256_set1_epi32(4095);
    for (int i = n - stride + first; i >= first; i -= stride) {
        const int group = channel_shift ? classes[i >> channel_shift] : 0;
        for (int group_index = 0; group_index < groups; ++group_index) {
            int base = group_index * 8;
            __m256i state = states[group_index];
            const B *symbols = (matrix ? widths : src) + i + base;
            const uint64_t *lookup = entries + (group & 3) * 256;
            int flip = (group >> 2) * 128;
            __m256i lower = _mm256_setr_epi64x(lookup[symbols[0] ^ flip], lookup[symbols[1] ^ flip],
                                               lookup[symbols[2] ^ flip], lookup[symbols[3] ^ flip]);
            __m256i upper = _mm256_setr_epi64x(lookup[symbols[4] ^ flip], lookup[symbols[5] ^ flip],
                                               lookup[symbols[6] ^ flip], lookup[symbols[7] ^ flip]);
            __m256i low_pairs = _mm256_permute2x128_si256(lower, upper, 0x20);
            __m256i high_pairs = _mm256_permute2x128_si256(lower, upper, 0x31);
            __m256i entry = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(low_pairs), _mm256_castsi256_ps(high_pairs), 0x88));
            __m256i reciprocal = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(low_pairs), _mm256_castsi256_ps(high_pairs), 0xdd));
            __m256i limit = _mm256_and_si256(entry, mask);
            __m256i first_byte = _mm256_cmpgt_epi32(_mm256_srli_epi32(state, 19), limit);
            __m256i second_byte = _mm256_cmpgt_epi32(_mm256_srli_epi32(state, 27), limit);
            __m256i count = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_setzero_si256(), first_byte), second_byte);
            alignas(32) U counts[8], emitted[8];
            _mm256_store_si256(reinterpret_cast<__m256i *>(counts), count);
            _mm256_store_si256(reinterpret_cast<__m256i *>(emitted), state);
            for (int lane = 0; lane < 8; ++lane) {
                // A symbol emits at most two bytes. The four-byte terminal slot
                // covers the speculative store even when this symbol emits none.
                uint16_t word = uint16_t(emitted[lane]);
                memcpy(next[base + lane], &word, sizeof(word));
                next[base + lane] += counts[lane];
            }
            state = _mm256_srlv_epi32(state, _mm256_slli_epi32(count, 3));
            __m256i even = _mm256_srli_epi64(_mm256_mul_epu32(state, reciprocal), 32);
            __m256i odd = _mm256_mul_epu32(_mm256_srli_epi64(state, 32), _mm256_srli_epi64(reciprocal, 32));
            __m256i high = _mm256_blend_epi32(even, odd, 0xaa);
            __m256i shift = _mm256_and_si256(_mm256_srli_epi32(entry, 12), _mm256_set1_epi32(15));
            __m256i quotient = _mm256_srlv_epi32(high, shift);
            // With frequency one the all-ones reciprocal gives state - 1.
            quotient = _mm256_sub_epi32(quotient, _mm256_cmpeq_epi32(limit, _mm256_setzero_si256()));
            __m256i complement = _mm256_sub_epi32(mask, limit);
            state = _mm256_add_epi32(state, _mm256_mullo_epi32(quotient, complement));
            state = _mm256_add_epi32(state, _mm256_srli_epi32(entry, 16));
            states[group_index] = state;
        }
    }
    alignas(32) U terminal[16];
    for (int j = 0; j < groups; ++j)
        _mm256_store_si256(reinterpret_cast<__m256i *>(terminal + j * 8), states[j]);
#else
    alignas(16) U terminal[16];
    uint32x4_t states[4];
    for (int j = 0; j < groups * 2; ++j)
        states[j] = vdupq_n_u32(L);
    for (int i = n - stride + first; i >= first; i -= stride) {
        const int group = channel_shift ? classes[i >> channel_shift] : 0;
        const uint64_t *lookup = entries + (group & 3) * 256;
        int flip = (group >> 2) * 128;
        for (int j = 0; j < groups * 2; ++j) {
            const B *symbols = (matrix ? widths : src) + i + j * 4;
            uint64x2_t a = {lookup[symbols[0] ^ flip], lookup[symbols[1] ^ flip]};
            uint64x2_t b = {lookup[symbols[2] ^ flip], lookup[symbols[3] ^ flip]};
            uint32x4_t entry = vuzp1q_u32(vreinterpretq_u32_u64(a), vreinterpretq_u32_u64(b));
            uint32x4_t reciprocal = vuzp2q_u32(vreinterpretq_u32_u64(a), vreinterpretq_u32_u64(b));
            uint32x4_t limit = vandq_u32(entry, vdupq_n_u32(4095));
            uint32x4_t state = states[j];
            uint32x4_t one = vcgtq_u32(vshrq_n_u32(state, 19), limit);
            uint32x4_t two = vcgtq_u32(vshrq_n_u32(state, 27), limit);
            int32x4_t negative = vreinterpretq_s32_u32(vaddq_u32(one, two));
            int shifts[4];
            U emitted[4];
            vst1q_s32(shifts, negative);
            vst1q_u32(emitted, state);
            for (int lane = 0; lane < 4; ++lane) {
                uint16_t word = uint16_t(emitted[lane]);
                memcpy(next[j * 4 + lane], &word, sizeof(word));
                next[j * 4 + lane] -= shifts[lane];
            }
            state = vshlq_u32(state, vshlq_n_s32(negative, 3));
            uint64x2_t lower = vmull_u32(vget_low_u32(state), vget_low_u32(reciprocal));
            uint64x2_t upper = vmull_high_u32(state, reciprocal);
            uint32x4_t high = vshrn_high_n_u64(vshrn_n_u64(lower, 32), upper, 32);
            uint32x4_t shift = vandq_u32(vshrq_n_u32(entry, 12), vdupq_n_u32(15));
            uint32x4_t quotient = vshlq_u32(high, vnegq_s32(vreinterpretq_s32_u32(shift)));
            quotient = vsubq_u32(quotient, vceqzq_u32(limit));
            uint32x4_t complement = vsubq_u32(vdupq_n_u32(4095), limit);
            states[j] = vaddq_u32(vmlaq_u32(state, quotient, complement), vshrq_n_u32(entry, 16));
        }
    }
    for (int j = 0; j < groups * 2; ++j)
        vst1q_u32(terminal + j * 4, states[j]);
#endif
    B *low[16], *high[16];
    int used = 0;
    for (int lane = 0; lane < 8 * groups; ++lane) {
        int length = int(next[lane] - streams[lane]) + 4;
        int prefix = matrix ? (low_bits[first + lane] + 7) / 8 : 0;
        if (length + prefix > capacity - used)
            return -1;
        low[lane] = out + used;
        used += prefix;
        high[lane] = out + used;
        used += length;
        length += prefix;
        lengths[2 * lane] = B(length);
        lengths[2 * lane + 1] = B(length >> 8);
    }
    if (matrix)
        put_matrix_bits(low, src, widths, n, first, stride, 8 * groups);
    for (int lane = 0; lane < 8 * groups; ++lane) {
        store32(next[lane], terminal[lane]);
        memcpy(high[lane], streams[lane], size_t(next[lane] - streams[lane]) + 4);
    }
    return used;
}
