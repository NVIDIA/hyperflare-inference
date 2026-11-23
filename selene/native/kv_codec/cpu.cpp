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
template <bool decode>
#ifdef KV_AVX2
KV_AVX2
#endif
static void predict_simd(B *words, U *residuals) {
    alignas(32) U prediction[128][128] = {};
    alignas(32) U deltas[128] = {};
    int index[128], position[128];
    alignas(32) U scales[128] = {};
    bool large[128] = {};
    for (int j = 0; j < 128; ++j)
        index[j] = position[j] = j;
    for (int k = 0; k < 64; ++k) {
        int last = 127 - k;
        uint64_t largest = 0;
        U smallest = UINT32_MAX;
        for (int j = 0; j <= last; ++j) {
            int column = index[j];
            U prior = prediction[k][j];
            U observed = matrix_item(words, column * 128 + k, residuals[k * 128 + column], prior, decode);
            deltas[j] = word(value(observed) - value(prior));
            U magnitude = deltas[j] & 0x7fffffffu;
            if (magnitude)
                smallest = std::min(smallest, magnitude);
            uint64_t candidate = (uint64_t(magnitude) << 7) | (127 - column);
            largest = std::max(largest, candidate);
        }
        int column = 127 - int(largest & 127);
        int pivot = position[column];
        U divisor = deltas[pivot];
        index[pivot] = index[last];
        position[index[pivot]] = pivot;
        deltas[pivot] = deltas[last];
        for (int i = k + 1; i < 128; ++i) {
            U *p = prediction[i];
            U prior = p[pivot];
            U observed = matrix_item(words, column * 128 + i, residuals[i * 128 + column], prior, decode);
            p[pivot] = p[last];
            scales[i] = word(value(observed) - value(prior));
        }
        if (!divisor)
            continue;
        divide_scales(scales, k + 1, divisor);
        for (int i = k + 1; i < 128; ++i) {
            U scale = scales[i];
            if (!scale)
                continue;
            U exponent = (scale >> 23) & 255;
            large[i] |= exponent + ((divisor >> 23) & 255) > 294;
            // A product >= 2^-26 cannot cancel a binary32 prior to a nonzero
            // magnitude below 2^-50. At most 64 products below 2^42 stay bounded.
            if (!large[i] && exponent + (smallest >> 23) >= 228)
                update_simd<false>(prediction[i], deltas, last, scale);
            else
                update_simd<true>(prediction[i], deltas, last, scale);
        }
    }
    for (int i = 64; i < 128; ++i)
        for (int j = 0; j < 64; ++j) {
            int column = index[j];
            matrix_item(words, column * 128 + i, residuals[i * 128 + column], prediction[i][j], decode);
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
        return encode_frame<true, 0>(packet, transformed, f.n);
    }
    transfer_frame<false>(f, raw, transformed, simd);
    if (f.kind == 40)
        return encode_frame<false, 8>(packet, transformed, f.n);
    if (channel_rows(f.kind))
        return encode_frame<false, 6>(packet, transformed, f.n);
    return encode_frame<false, 0>(packet, transformed, f.n);
}

extern "C" int kv_cpu_frame(const Frame *frame, B *raw, B *packet, int capacity, U *scratch, bool decoding) {
    static const bool simd = has_simd();
    return cpu_frame(*frame, raw, packet, capacity, scratch, decoding, simd);
}

extern "C" int kv_cpu_frame_scalar(const Frame *frame, B *raw, B *packet, int capacity, U *scratch, bool decoding) {
    return cpu_frame(*frame, raw, packet, capacity, scratch, decoding, false);
}
