// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
#include <string.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif
using U = uint32_t;
using B = uint8_t;
struct Frame {
    int base, full, sub, n, kind, rank, layer, page, work, arena, reserved;
};
struct Entry {
    int base, full, sub, n, kind, rank, layer, page;
    U pos, len;
};
static_assert(sizeof(Entry) == 40 && sizeof(Frame) == 44, "wire ABI");
HD int lanes(int n) {
    return n < 256 ? 1 : n < 1024 ? 2 : n < 4096 ? 8 : n < 16384 ? 16 : n < 65536 ? 32 : 128;
}
HD int lg(U a) {
#ifdef __CUDA_ARCH__
    return 32 - __clz(a);
#else
#ifdef _MSC_VER
    unsigned long bit;
    return _BitScanReverse(&bit, a) ? int(bit + 1) : 0;
#else
    return a ? 32 - __builtin_clz(a) : 0;
#endif
#endif
}
HD U load32(const B *p) {
    return U(p[0]) | (U(p[1]) << 8) | (U(p[2]) << 16) | (U(p[3]) << 24);
}
HD void store32(B *p, U x) {
    p[0] = x;
    p[1] = x >> 8;
    p[2] = x >> 16;
    p[3] = x >> 24;
}
HD U order(U w) {
    return w ^ ((0u - (w >> 31)) | 0x80000000u);
}
HD U residual(U w, U p) {
    U d = order(w) - order(p);
    return (d << 1) ^ (0u - (d >> 31));
}
HD U restore(U z, U p) {
    U b = order(p) + ((z >> 1) ^ (0u - (z & 1)));
    return b ^ (((b >> 31) - 1u) | 0x80000000u);
}
#include "arith.h"
HD U item(U &w, U &z, U pred, bool decode) {
    if (decode)
        w = restore(z, pred);
    else
        z = residual(w, pred);
    return bounded(w);
}
// Subranges use the domain of the transposed plane. No frame spans two units.
HD int address(Frame f, int i) {
    int k = f.kind, x = f.sub + i;
    if (k == 0)
        return f.base + x;
    if (k < 40) {
        int rows = f.full / k;
        return f.base + (x % rows) * k + x / rows;
    }
    if (k <= 42) {
        int unit = k == 40 ? 135168 : k == 41 ? 8448 : 32768;
        int rows = k == 40 ? 256 : 64, stride = k == 40 ? 528 : k == 41 ? 128 : 512;
        int off = f.sub / unit * unit;
        int t = x - off;
        return f.base + off + (t % rows) * stride + t / rows;
    }
    if (k >= 45 && k <= 48) {
        bool window = k <= 46, scale = (k % 2) == 0;
        int cols = window ? 512 : 256, sc = window ? 16 : 32, stride = cols + sc;
        if (scale)
            return f.base + (i / sc) * stride + cols + i % sc;
        if (window)
            return f.base + (i % 64) * stride + i / 64;
        return f.base + (i / cols) * stride + i % cols;
    }
    int unit = k == 43 ? 135168 : 8448, ns = k == 43 ? 1024 : 64;
    int off = f.sub / unit * unit, t = x - off, s = t % ns, lane = t / ns;
    return f.base + off + (k == 43 ? s / 4 * 528 + 512 + s % 4 * 4 : 8192 + s * 4) + lane;
}
HD B gather(Frame f, int i, const B *raw) {
    return raw[address(f, i)];
}
HD B unfold(Frame, int i, const B *t) {
    return t[i];
}
HD int channel_rows(int kind) {
    return kind == 40 ? 256 : (kind == 41 || kind == 42 || kind == 45) ? 64 : 0;
}
HD B classify(const B *src, int rows) {
    int neg = 0;
    for (int i = 0; i < rows; i++)
        neg += src[i] >> 7;
    bool flip = neg > rows / 2;
    return (flip ? 4 : 0) + (flip ? rows - neg : neg) * 8 / (rows + 1);
}
HD int model_symbol(U v, int i, bool mat, int rows, const B *cl) {
    if (!rows)
        return mat ? lg(v) : int(v);
    int c = cl[i / rows];
    return (c % 4) * 256 + (int(v) ^ ((c / 4) * 128));
}
HD U value(const B *p, int i, bool mat) {
    return mat ? reinterpret_cast<const U *>(p)[i] : p[i];
}
HD int sym(U v, bool mat) {
    return mat ? lg(v) : int(v);
}
HD int lowbits(int s, bool mat) {
    return mat && s ? s - 1 : 0;
}
constexpr U L = 1 << 23;
HD U ans_put(U state, int f, int cum, B *dst, int &used) {
    U limit = U(f) << 19;
    while (state >= limit) {
        if (dst)
            dst[used] = state;
        used++;
        state >>= 8;
    }
    return (state / U(f)) * 4096 + (state % U(f)) + cum;
}
HD void lane_size(const B *src, int n, bool mat, int l, int nl, const int *freq, const int *cum, int &low,
                  int &bytes, int rows, const B *cl) {
    U st = L;
    low = 0;
    bytes = 0;
    for (int i = l + (n - 1 - l) / nl * nl; i >= l; i -= nl) {
        int s = model_symbol(value(src, i, mat), i, mat, rows, cl);
        low += lowbits(s, mat);
        st = ans_put(st, freq[s], cum[s], nullptr, bytes);
    }
    low = (low + 7) / 8;
    bytes += 4 + low;
}
HD void put_lane(B *out, const B *src, int n, bool mat, int l, int nl, const int *freq, const int *cum,
                 int low, int rows, const B *cl) {
    U st = L;
    int pos = low;
    for (int i = l + (n - 1 - l) / nl * nl; i >= l; i -= nl) {
        int s = model_symbol(value(src, i, mat), i, mat, rows, cl);
        st = ans_put(st, freq[s], cum[s], out, pos);
    }
    store32(out + pos, st);
    if (mat) {
        uint64_t acc = 0;
        int bits = 0, k = 0;
        for (int i = l; i < n; i += nl) {
            U v = value(src, i, true);
            int b = lowbits(lg(v), true);
            acc |= uint64_t(v & ((1u << b) - 1)) << bits;
            bits += b;
            while (bits >= 8) {
                out[k++] = acc;
                acc >>= 8;
                bits -= 8;
            }
        }
        if (bits)
            out[k] = acc;
    }
}
HD bool get_lane(B *out, const B *src, int bytes, int n, bool mat, int l, int nl, const int *freq,
                 const int *cum, const B *lut, int rows, const B *cl) {
    if (bytes < 4)
        return false;
    int end = bytes - 4, k = 0, bits = 0;
    U st = load32(src + end);
    uint64_t acc = 0;
    for (int i = l; i < n; i += nl) {
        int c = rows ? cl[i / rows] : 0, ctx = c % 4, ix = st & 4095, s = lut[ctx * 4096 + ix];
        st = freq[ctx * 256 + s] * (st >> 12) + ix - cum[ctx * 256 + s];
        while (st < L) {
            if (!end)
                return false;
            st = (st << 8) | src[--end];
        }
        if (mat) {
            int b = lowbits(s, true);
            while (bits < b) {
                if (k >= end)
                    return false;
                acc |= uint64_t(src[k++]) << bits;
                bits += 8;
            }
            reinterpret_cast<U *>(out)[i] = s ? (1u << b) | (U(acc) & ((1u << b) - 1)) : 0;
            acc >>= b;
            bits -= b;
        } else
            out[i] = s ^ ((c / 4) * 128);
    }
    return st == L && end == k;
}
HD void normalize(const int *hist, int *freq, int *cum, int m, int n) {
    if (!n) {
        for (int i = 0; i < m; i++)
            freq[i] = cum[i] = 0;
        return;
    }
    int active = 0, best = 0;
    for (int i = 0; i < m; i++) {
        active += hist[i] != 0;
        if (hist[i] > hist[best])
            best = i;
    }
    int total = 0;
    for (int i = 0; i < m; i++) {
        freq[i] = hist[i] ? 1 + hist[i] * (4096 - active) / n : 0;
        total += freq[i];
    }
    freq[best] += 4096 - total;
    total = 0;
    for (int i = 0; i < m; i++) {
        cum[i] = total;
        total += freq[i];
    }
}

// One presence bit per symbol, followed by one or two bytes per nonzero count.
HD int write_table(B *dst, const int *freq, int m) {
    int mask = (m + 7) / 8, pos = mask;
    for (int i = 0; i < mask; i++)
        dst[i] = 0;
    for (int i = 0; i < m; i++)
        if (freq[i]) {
            int f = freq[i];
            dst[i / 8] |= 1 << (i % 8);
            dst[pos++] = (f & 127) | (f >= 128 ? 128 : 0);
            if (f >= 128)
                dst[pos++] = f >> 7;
        }
    return pos;
}
HD int read_table(const B *src, int cap, int *freq, int *cum, int m) {
    int pos = (m + 7) / 8, total = 0;
    if (cap < pos)
        return -1;
    for (int i = 0; i < m; i++) {
        int f = 0;
        if (src[i / 8] & (1 << (i % 8))) {
            if (pos >= cap)
                return -1;
            f = src[pos++];
            if (f & 128) {
                if (pos >= cap)
                    return -1;
                f = (f & 127) + int(src[pos++]) * 128;
            }
            if (!f)
                return -1;
        }
        freq[i] = f;
        cum[i] = total;
        total += f;
    }
    return total == 4096 || total == 0 ? pos : -1;
}
