// SPDX-License-Identifier: Apache-2.0

#pragma once
// Predictor operands are canonical binary32 words. The exponent window keeps
// even the unrounded intermediate of every operation normal, or exactly zero.
HD U bounded(U w) {
    U e = (w >> 23) & 255;
    return e >= 77 && e <= 177 ? w : 0;
}
#ifndef __CUDA_ARCH__
HD U rounded(uint64_t m, int power, U sign) {
    if (!m)
        return 0;
    int top = uint32_t(m >> 32) ? 31 + lg(uint32_t(m >> 32)) : lg(uint32_t(m)) - 1;
    int shift = top - 23;
    uint64_t q;
    if (shift > 0) {
        q = m >> shift;
        uint64_t mask = (1ull << shift) - 1, half = 1ull << (shift - 1);
        uint64_t rem = m & mask;
        q += rem > half || (rem == half && (q & 1));
    } else
        q = m << (-shift);
    int exponent = top + power + 127;
    if (q == (1ull << 24)) {
        q >>= 1;
        exponent++;
    }
    return exponent < 77 || exponent > 177 ? 0 : sign | (U(exponent) << 23) | (U(q) & 0x7fffff);
}
HD U mant(U w) {
    return (w & 0x7fffff) | 0x800000;
}
#endif
HD U plus(U a, U b) {
#ifdef __CUDA_ARCH__
    return bounded(__float_as_uint(__fadd_rn(__uint_as_float(a), __uint_as_float(b))));
#else
    if (!(a & 0x7fffffff))
        return bounded(b);
    if (!(b & 0x7fffffff))
        return bounded(a);
    int ea = (a >> 23) & 255, eb = (b >> 23) & 255;
    if (ea < eb) {
        U w = a;
        a = b;
        b = w;
        int ec = ea;
        ea = eb;
        eb = ec;
    }
    int gap = ea - eb;
    if (gap > 25)
        return a;
    int64_t am = int64_t(mant(a)) << gap, bm = mant(b);
    int64_t result = (a >> 31 ? -am : am) + (b >> 31 ? -bm : bm);
    return rounded(result < 0 ? -result : result, eb - 150, result < 0 ? 0x80000000 : 0);
#endif
}
HD U minus_word(U a, U b) {
    return plus(a, b ^ 0x80000000u);
}
HD U times(U a, U b) {
#ifdef __CUDA_ARCH__
    return bounded(__float_as_uint(__fmul_rn(__uint_as_float(a), __uint_as_float(b))));
#else
    if (!a || !b)
        return 0;
    return rounded(uint64_t(mant(a)) * mant(b), int((a >> 23) & 255) + int((b >> 23) & 255) - 300,
                   (a ^ b) & 0x80000000u);
#endif
}
HD U divide(U a, U b) {
    if (!b)
        return 0;
#ifdef __CUDA_ARCH__
    return bounded(__float_as_uint(__fdiv_rn(__uint_as_float(a), __uint_as_float(b))));
#else
    if (!a)
        return 0;
    uint64_t num = uint64_t(mant(a)) << 27;
    uint64_t q = num / mant(b), r = num % mant(b);
    return rounded(q * 2 + (r != 0), int((a >> 23) & 255) - int((b >> 23) & 255) - 28, (a ^ b) & 0x80000000u);
#endif
}
