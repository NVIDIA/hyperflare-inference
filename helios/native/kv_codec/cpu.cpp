// SPDX-License-Identifier: Apache-2.0

#include "core.h"
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cfenv>
#include <cfloat>
#include <limits>
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
static void matrix(B *words, U *z, bool dec) {
#if (defined(__aarch64__) || defined(__x86_64__)) && !defined(__FAST_MATH__) && FLT_EVAL_METHOD == 0
    static_assert(sizeof(float) == sizeof(U) && std::numeric_limits<float>::is_iec559);
    // Bounded operands produce only normal binary32 intermediates or zero.
    std::fenv_t saved;
    if (std::feholdexcept(&saved) == 0) {
        if (std::fesetround(FE_TONEAREST) == 0) {
            predict_matrix<true>(words, z, dec);
            std::fesetenv(&saved);
            return;
        }
        std::fesetenv(&saved);
    }
#endif
    predict_matrix<false>(words, z, dec);
}

static int encode(B *out, const B *t, int n, int kind) {
    bool mat = kind == 1;
    int rows = channel_rows(kind), nc = rows ? n / rows : 0, models = rows ? 4 : 1;
    int hist[1024] = {}, freq[1024], cum[1024], m = mat ? 33 : 256, nl = lanes(n);
    int populations[4] = {};
    B cl[512], tables[4096];
    for (int c = 0; c < nc; c++)
        cl[c] = classify(t + c * rows, rows);
    for (int i = 0; i < n; i++) {
        int s = model_symbol(value(t, i, mat), i, mat, rows, cl);
        hist[s]++;
        populations[s / 256]++;
    }
    out[0] = 2;
    out[1] = out[2] = out[3] = 0;
    if (!mat && std::all_of(t + 1, t + n, [&](B x) { return x == t[0]; })) {
        out[0] = 1;
        out[2] = t[0];
        return 4;
    }
    int nt = 0;
    for (int j = 0; j < models; j++) {
        normalize(hist + 256 * j, freq + 256 * j, cum + 256 * j, m, populations[j]);
        nt += write_table(tables + nt, freq + 256 * j, m);
    }
    int head = 4 + nc + nt + 2 * nl, pos = head, lo[128], bytes[128];
    for (int l = 0; l < nl; l++) {
        lane_size(t, n, mat, l, nl, freq, cum, lo[l], bytes[l], rows, cl);
        pos += bytes[l];
    }
    int raw = n * (mat ? 4 : 1);
    if (pos >= raw + 4) {
        out[0] = 0;
        if (mat)
            for (int i = 0; i < n; i++)
                store32(out + 4 + 4 * i, value(t, i, true));
        else
            memcpy(out + 4, t, raw);
        return 4 + raw;
    }
    memcpy(out + 4, cl, nc);
    memcpy(out + 4 + nc, tables, nt);
    pos = head;
    for (int l = 0; l < nl; l++) {
        int j = 4 + nc + nt + 2 * l;
        out[j] = bytes[l];
        out[j + 1] = bytes[l] >> 8;
        put_lane(out + pos, t, n, mat, l, nl, freq, cum, lo[l], rows, cl);
        pos += bytes[l];
    }
    return pos;
}
static bool decode(B *t, const B *in, int bytes, int n, int kind) {
    bool mat = kind == 1;
    int rows = channel_rows(kind), nc = rows ? n / rows : 0, models = rows ? 4 : 1;
    if (bytes < 4)
        return false;
    if (in[0] == 0) {
        int raw = n * (mat ? 4 : 1);
        if (bytes != 4 + raw)
            return false;
        if (mat)
            for (int i = 0; i < n; i++)
                reinterpret_cast<U *>(t)[i] = load32(in + 4 + 4 * i);
        else
            memcpy(t, in + 4, raw);
        return true;
    }
    if (in[0] == 1) {
        if (mat || bytes != 4)
            return false;
        memset(t, in[2], n);
        return true;
    }
    int m = mat ? 33 : 256, nl = lanes(n), pos = 4 + nc, freq[1024], cum[1024];
    B lut[4 * 4096] = {};
    if (in[0] != 2 || pos > bytes)
        return false;
    const B *cl = in + 4;
    for (int c = 0; c < nc; c++)
        if (cl[c] >= 8)
            return false;
    for (int j = 0; j < models; j++) {
        int k = read_table(in + pos, bytes - pos, freq + 256 * j, cum + 256 * j, m);
        if (k < 0)
            return false;
        pos += k;
        for (int i = 0; i < m; i++)
            memset(lut + 4096 * j + cum[j * 256 + i], i, freq[j * 256 + i]);
    }
    int directory = pos;
    pos += 2 * nl;
    if (pos > bytes)
        return false;
    for (int l = 0; l < nl; l++) {
        int j = directory + 2 * l, b = in[j] + 256 * in[j + 1];
        if (b > bytes - pos || !get_lane(t, in + pos, b, n, mat, l, nl, freq, cum, lut, rows, cl))
            return false;
        pos += b;
    }
    return pos == bytes;
}

extern "C" int kv_cpu_frame(const Frame *frame, B *raw, B *packet, int capacity, U *scratch, bool decoding) {
    const Frame f = *frame;
    const bool mat = f.kind == 1;
    const int size = f.n * (mat ? 4 : 1);
    B *transformed = reinterpret_cast<B *>(scratch);
    if (decoding) {
        if (!decode(transformed, packet, capacity, f.n, f.kind))
            return -1;
        if (mat)
            matrix(raw + f.base + f.sub, scratch, true);
        else
            for (int i = 0; i < f.n; ++i)
                raw[address(f, i)] = transformed[i];
        return size;
    }
    if (capacity < size + 4)
        return -1;
    if (mat)
        matrix(raw + f.base + f.sub, scratch, false);
    else
        for (int i = 0; i < f.n; ++i)
            transformed[i] = raw[address(f, i)];
    return encode(packet, transformed, f.n, f.kind);
}
