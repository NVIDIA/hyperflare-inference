// SPDX-License-Identifier: Apache-2.0
#pragma once
template <bool FAST> __device__ U ans_opt(U st, int f, int cum, U inv, B *dst, int &used) {
    if (!FAST)
        return ans_put(st, f, cum, dst, used);
    U limit = U(f) << 19;
    while (st >= limit) {
        if (dst)
            dst[used] = st;
        used++;
        st >>= 8;
    }
    U q = f == 1 ? st : __umulhi(st, inv);
    q -= q * unsigned(f) > st;
    return q * 4096 + (st - q * unsigned(f)) + cum;
}
template <bool FAST>
__device__ void lane_size_opt(const unsigned *inv, const B *src, int n, bool mat, int l, int nl,
                              const int *freq, const int *cum, int &low, int &bytes, int rows, const B *cl) {
    U st = L;
    low = 0;
    bytes = 0;
    for (int i = l + (n - 1 - l) / nl * nl; i >= l; i -= nl) {
        int s = model_symbol(value(src, i, mat), i, mat, rows, cl);
        low += lowbits(s, mat);
        st = ans_opt<FAST>(st, freq[s], cum[s], FAST ? inv[s] : 0, nullptr, bytes);
    }
    low = (low + 7) / 8;
    bytes += 4 + low;
}
template <bool FAST>
__device__ void put_lane_opt(const unsigned *inv, B *out, const B *src, int n, bool mat, int l, int nl,
                             const int *freq, const int *cum, int low, int rows, const B *cl) {
    U st = L;
    int pos = low;
    for (int i = l + (n - 1 - l) / nl * nl; i >= l; i -= nl) {
        int s = model_symbol(value(src, i, mat), i, mat, rows, cl);
        st = ans_opt<FAST>(st, freq[s], cum[s], FAST ? inv[s] : 0, out, pos);
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
