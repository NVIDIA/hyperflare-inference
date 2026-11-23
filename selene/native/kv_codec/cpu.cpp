// SPDX-License-Identifier: Apache-2.0

#include "core.h"
#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <limits>
#include "cpu_simd.h"
#include "cpu_entropy.h"
static U matrix_item(B *words, int index, U &code, U pred, bool decode) {
    B *addr = words + 4 * index;
    U w = decode ? 0 : load32(addr);
    U observed = item(w, code, pred, decode);
    if (decode)
        store32(addr, w);
    return observed;
}
static float value(U bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
static U word(float f) {
    U bits;
    memcpy(&bits, &f, sizeof(bits));
    return bounded(bits);
}
template <bool native> static void predict_matrix(B *words, U *z, bool dec) {
    U p[128][128] = {}, u[128], v[128];
    bool done[128] = {};
    for (int k = 0; k < 64; k++) {
        int c = -1;
        for (int j = 0; j < 128; j++)
            if (!done[j]) {
                int t = j * 128 + k;
                U w = matrix_item(words, t, z[k * 128 + j], p[k][j], dec);
                v[j] = native ? word(value(w) - value(p[k][j])) : minus_word(w, p[k][j]);
                if (c < 0 || (v[j] & 0x7fffffffu) > (v[c] & 0x7fffffffu))
                    c = j;
            }
        U pivot = v[c];
        done[c] = true;
        for (int i = k + 1; i < 128; i++) {
            int t = c * 128 + i;
            U w = matrix_item(words, t, z[i * 128 + c], p[i][c], dec);
            U delta = native ? word(value(w) - value(p[i][c])) : minus_word(w, p[i][c]);
            u[i] = native ? (pivot ? word(value(delta) / value(pivot)) : 0) : divide(delta, pivot);
            for (int j = 0; j < 128; j++)
                if (!done[j]) {
                    U product = native ? word(value(u[i]) * value(v[j])) : times(u[i], v[j]);
                    p[i][j] = native ? word(value(p[i][j]) + value(product)) : plus(p[i][j], product);
                }
        }
    }
    for (int i = 64; i < 128; i++)
        for (int j = 0; j < 128; j++)
            if (!done[j])
                matrix_item(words, j * 128 + i, z[i * 128 + j], p[i][j], dec);
}
#if defined(KV_AVX2) || defined(KV_NEON)
// A pivot row produces its residuals and the differences used by later rows.
template <bool decode>
#ifdef KV_AVX2
KV_AVX2
#endif
static void pivot_items(U *items, const U *predicted, U *differences, int count) {
    for (int j = 0; j < count; ++j) {
        U prior = predicted[j];
        U input = items[j];
        U observed = decode ? restore(input, prior) : input;
        items[j] = decode ? observed : residual(observed, prior);
        differences[j] = word(value(bounded(observed)) - value(prior));
    }
}

static void transpose_matrix_words(const B *input, B *output, int rows, int columns) {
    for (int row = 0; row < rows; row += 4) {
        for (int col = 0; col < columns; col += 4) {
#ifdef KV_AVX2
            __m128 lanes[4];
            for (int i = 0; i < 4; ++i) {
                const B *src = input + 4 * ((row + i) * 128 + col);
                lanes[i] = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src)));
            }
            _MM_TRANSPOSE4_PS(lanes[0], lanes[1], lanes[2], lanes[3]);
            for (int i = 0; i < 4; ++i) {
                B *dst = output + 4 * ((col + i) * 128 + row);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), _mm_castps_si128(lanes[i]));
            }
#else
            uint32x4_t lanes[4];
            for (int i = 0; i < 4; ++i)
                lanes[i] = vreinterpretq_u32_u8(vld1q_u8(input + 4 * ((row + i) * 128 + col)));
            auto top = vtrnq_u32(lanes[0], lanes[1]);
            auto bottom = vtrnq_u32(lanes[2], lanes[3]);
            for (int i = 0; i < 2; ++i) {
                auto a = vreinterpretq_u64_u32(top.val[i]);
                auto b = vreinterpretq_u64_u32(bottom.val[i]);
                vst1q_u8(output + 4 * ((col + i) * 128 + row), vreinterpretq_u8_u64(vzip1q_u64(a, b)));
                vst1q_u8(output + 4 * ((col + i + 2) * 128 + row), vreinterpretq_u8_u64(vzip2q_u64(a, b)));
            }
#endif
        }
    }
}

template <bool decode>
#ifdef KV_AVX2
KV_AVX2
#endif
static void predict_simd(B *words, U *residuals) {
    constexpr int band = 16;
    alignas(32) U basis[64][128];
    U minima[64], divisors[64];
    int columns[64];
    U active[128];
    std::fill_n(active, 128, ~0u);
    // Only the pivot row chooses a column. Each lower band can replay the
    // same pivots in order with 8 KiB of predictions resident at a time.
    for (int base = 0; base < 128; base += band) {
        B *band_words = words + 4 * base;
        B *band_residuals = reinterpret_cast<B *>(residuals + 128 * base);
        if (!decode)
            transpose_matrix_words(band_words, band_residuals, 128, band);
        alignas(32) U prediction[band][128] = {};
        alignas(32) U scales[band] = {};
        bool large[band] = {};
        for (int k = 0; k < std::min(base + band, 64); ++k) {
            U *deltas = basis[k];
            if (k >= base) {
                U *row = residuals + 128 * k;
                pivot_items<decode>(row, prediction[k - base], deltas, 128);
                U smallest = UINT32_MAX;
                U largest = 0;
                for (int j = 0; j < 128; ++j) {
                    deltas[j] &= active[j];
                    U magnitude = deltas[j] & 0x7fffffffu;
                    smallest = std::min(smallest, magnitude - 1);
                    largest = std::max(largest, magnitude);
                }
                int column = 0;
                while (!active[column] || (deltas[column] & 0x7fffffffu) != largest)
                    ++column;
                // A consumed column keeps its current prediction for every
                // lower row. Zero deltas leave that prediction unchanged.
                active[column] = 0;
                columns[k] = column;
                divisors[k] = deltas[column];
                minima[k] = (smallest + 1) >> 23;
                deltas[column] = 0;
            }
            U divisor = divisors[k];
            if (!divisor)
                continue;
            int column = columns[k];
            int start = std::max(0, k + 1 - base);
            for (int i = start; i < band; ++i) {
                U prior = prediction[i][column];
                U input = residuals[(base + i) * 128 + column];
                U observed = decode ? restore(input, prior) : input;
                scales[i] = word(value(bounded(observed)) - value(prior));
            }
            divide_scales(scales, start, divisor, band);
            for (int i = start; i < band; ++i) {
                U scale = scales[i];
                if (!scale)
                    continue;
                U exponent = (scale >> 23) & 255;
                large[i] |= exponent + ((divisor >> 23) & 255) > 294;
                // Above 2^-26, binary32 cancellation cannot leave a nonzero
                // sum below 2^-50. Above 2^-50 only the product is safe.
                if (!large[i] && exponent + minima[k] >= 228)
                    update_simd<false>(prediction[i], deltas, 128, scale);
                else if (!large[i] && exponent + minima[k] >= 204)
                    update_simd<true, false>(prediction[i], deltas, 128, scale);
                else
                    update_simd<true>(prediction[i], deltas, 128, scale);
            }
        }
        if (base >= 64) {
            for (int i = 0; i < band; ++i)
                for (int j = 0; j < 128; ++j) {
                    U &input = residuals[(base + i) * 128 + j];
                    input = decode ? restore(input, prediction[i][j]) : residual(input, prediction[i][j]);
                }
        }
        if (decode)
            transpose_matrix_words(band_residuals, band_words, band, 128);
    }
}
#endif

static void matrix(B *words, U *z, bool dec, bool simd = has_simd()) {
#if (defined(__aarch64__) || defined(__x86_64__)) && !defined(__FAST_MATH__) && FLT_EVAL_METHOD == 0
    static_assert(sizeof(float) == sizeof(U) && std::numeric_limits<float>::is_iec559);
    // Bounded operands produce only normal binary32 intermediates or zero.
    std::fenv_t saved;
    if (std::feholdexcept(&saved) == 0) {
        if (std::fesetround(FE_TONEAREST) == 0) {
#if defined(KV_AVX2) || defined(KV_NEON)
            if (simd) {
                if (dec)
                    predict_simd<true>(words, z);
                else
                    predict_simd<false>(words, z);
            } else
#endif
                predict_matrix<true>(words, z, dec);
            std::fesetenv(&saved);
            return;
        }
        std::fesetenv(&saved);
    }
#endif
    predict_matrix<false>(words, z, dec);
}


template <bool decode>
static void transfer_frame(const Frame &f, B *raw, B *transformed, bool simd) {
    B *base = raw + f.base;
    auto copy = [](B *original, B *plane, size_t len) {
        if (decode)
            memcpy(original, plane, len);
        else
            memcpy(plane, original, len);
    };
    if (f.kind == 0) {
        copy(base + f.sub, transformed, f.n);
        return;
    }
    if (f.kind == 47 || f.kind == 46 || f.kind == 48) {
        int count = f.kind == 47 ? 256 : f.kind == 46 ? 16 : 32;
        int stride = f.kind == 46 ? 528 : 288;
        int offset = f.kind == 47 ? 0 : stride - count;
        for (int i = 0; i < f.n; i += count)
            copy(base + (i / count) * stride + offset, transformed + i, count);
        return;
    }
#if defined(KV_AVX2) || defined(KV_NEON)
    if (simd && channel_rows(f.kind)) {
        int rows = channel_rows(f.kind);
        int columns = f.n / rows;
        int stride = f.kind == 40 || f.kind == 45 ? 528 : columns;
        B *original = base + f.sub;
        if (decode)
            transpose(transformed, rows, original, stride, columns, rows);
        else
            transpose(original, stride, transformed, rows, rows, columns);
        return;
    }
#endif
    if (f.kind == 2 || f.kind == 20 || f.kind == 32) {
        int rows = f.full / f.kind;
        B *source = base + (f.sub % rows) * f.kind + f.sub / rows;
        for (int i = 0; i < f.n; ++i)
            if (decode)
                source[i * f.kind] = transformed[i];
            else
                transformed[i] = source[i * f.kind];
        return;
    }
    for (int i = 0; i < f.n; ++i)
        if (decode)
            raw[address(f, i)] = transformed[i];
        else
            transformed[i] = raw[address(f, i)];
}

static int cpu_frame(const Frame &f, B *raw, B *packet, int capacity, U *scratch, bool decoding, bool simd) {
    const bool mat = f.kind == 1;
    const int size = f.n * (mat ? 4 : 1);
    B *transformed = reinterpret_cast<B *>(scratch);
    if (decoding) {
        bool ok;
        if (mat)
            ok = decode_frame<true, 0>(transformed, packet, capacity, f.n, simd);
        else if (f.kind == 40)
            ok = decode_frame<false, 8>(transformed, packet, capacity, f.n, simd);
        else if (channel_rows(f.kind))
            ok = decode_frame<false, 6>(transformed, packet, capacity, f.n, simd);
        else
            ok = decode_frame<false, 0>(transformed, packet, capacity, f.n, simd);
        if (!ok)
            return -1;
        if (mat)
            matrix(raw + f.base + f.sub, scratch, true, simd);
        else
            transfer_frame<true>(f, raw, transformed, simd);
        return size;
    }
    if (capacity < size + 4)
        return -1;
    if (mat) {
        matrix(raw + f.base + f.sub, scratch, false, simd);
        return encode_frame<true, 0>(packet, transformed, f.n, simd);
    }
    transfer_frame<false>(f, raw, transformed, simd);
    if (f.kind == 40)
        return encode_frame<false, 8>(packet, transformed, f.n, simd);
    if (channel_rows(f.kind))
        return encode_frame<false, 6>(packet, transformed, f.n, simd);
    return encode_frame<false, 0>(packet, transformed, f.n, simd);
}

extern "C" int kv_cpu_frame(const Frame *frame, B *raw, B *packet, int capacity, U *scratch, bool decoding) {
    static const bool simd = has_simd();
    return cpu_frame(*frame, raw, packet, capacity, scratch, decoding, simd);
}

extern "C" int kv_cpu_frame_scalar(const Frame *frame, B *raw, B *packet, int capacity, U *scratch, bool decoding) {
    return cpu_frame(*frame, raw, packet, capacity, scratch, decoding, false);
}
