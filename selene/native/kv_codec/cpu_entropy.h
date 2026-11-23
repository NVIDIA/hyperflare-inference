// SPDX-License-Identifier: Apache-2.0
#pragma once

struct EncodeSymbol {
    U limit;
    U reciprocal;
    U shift;
    U complement;
    U bias;
};

struct Quotient {
    U multiplier;
    U shift;
};

struct DivisionTable {
    Quotient entries[4097] = {};

    constexpr DivisionTable() {
        entries[1] = {UINT32_MAX, 0};
        U shift = 0;
        for (U f = 2; f <= 4096; ++f) {
            if (f > (U(1) << shift))
                ++shift;
            entries[f] = {U(((uint64_t(1) << (31 + shift)) + f - 1) / f), shift - 1};
        }
    }
};
static constexpr DivisionTable division_table;

static EncodeSymbol encoder_symbol(int frequency, int start) {
    U f = frequency;
    const Quotient quotient = division_table.entries[f];
    return {f << 19, quotient.multiplier, quotient.shift, 4096 - f, U(start) + (f == 1 ? 4095u : 0u)};
}

static inline U put_symbol(U state, const EncodeSymbol &s, B *&next) {
    // A normalized state is below 2^31. Multiplication by the rounded-up
    // reciprocal gives its exact quotient by f in this range.
    *next = B(state);
    bool emit = state >= s.limit;
    next += emit;
    state >>= 8 * emit;
    if (state >= s.limit) {
        *next++ = B(state);
        state >>= 8;
    }
    U quotient = U((uint64_t(state) * s.reciprocal) >> 32) >> s.shift;
    return state + quotient * s.complement + s.bias;
}

// Lane counts in lanes(n) give at most 2048 symbols per lane. Each emits
// at most two normalization bytes. Low bits for matrix residuals precede them.
constexpr int MAX_LANE_SYMBOLS = 2048;

static void put_low_bits(B *out, const B *src, const B *widths, int n, int first, int stride) {
    B *next_low = out;
    uint64_t bits = 0;
    int count = 0;
    for (int i = first; i < n; i += stride) {
        U v = value(src, i, true);
        int low = widths[i] ? widths[i] - 1 : 0;
        bits |= uint64_t(v & ((U(1) << low) - 1)) << count;
        count += low;
        // At least four bytes of terminal state follow this bit stream.
        store32(next_low, U(bits));
        next_low += count / 8;
        bits >>= count & ~7;
        count &= 7;
    }
    if (count)
        *next_low = B(bits);
}

#if defined(KV_AVX2) || defined(KV_NEON)
#include "cpu_encode.h"
#endif

template <int width, bool mat, int channel_shift>
static int put_group(B *out, int capacity, const B *src, int n, int first, int stride,
                     const EncodeSymbol *table, const B *classes, B *lengths, const B *widths, const int *low_bits) {
    B streams[width][2 * MAX_LANE_SYMBOLS + 4];
    B *next[width];
    U states[width];
    for (int j = 0; j < width; ++j) {
        states[j] = L;
        next[j] = streams[j];
    }
    int index = first + (n - 1 - first) / stride * stride;
    auto put = [&](int i, int count) {
        const int group = channel_shift ? classes[i >> channel_shift] : 0;
        const EncodeSymbol *model = table + (group & 3) * 256;
        const U flip = (group >> 2) * 128;
        for (int j = 0; j < count; ++j) {
            U symbol_id = mat ? widths[i + j] : src[i + j];
            const auto &symbol = model[symbol_id ^ flip];
            states[j] = put_symbol(states[j], symbol, next[j]);
        }
    };
    if (index + width > n) {
        put(index, n - index);
        index -= stride;
    }
    for (; index >= first; index -= stride)
        put(index, width);
    int used = 0;
    for (int j = 0; j < width; ++j) {
        int high = int(next[j] - streams[j]) + 4;
        if (capacity - used < high)
            return -1;
        int begin = used;
        if (mat) {
            int bytes = (low_bits[first + j] + 7) / 8;
            if (capacity - used - high < bytes)
                return -1;
            put_low_bits(out + used, src, widths, n, first + j, stride);
            used += bytes;
        }
        store32(next[j], states[j]);
        memcpy(out + used, streams[j], high);
        used += high;
        int length = used - begin;
        lengths[2 * j] = B(length);
        lengths[2 * j + 1] = B(length >> 8);
    }
    return used;
}

// Lower bound on log2(4096 / frequency), in 1/256 bits. Truncating each
// fixed-point square rounds the bound down.
struct SymbolCost {
    uint16_t bits[4097] = {};
    constexpr SymbolCost() {
        for (U f = 1; f <= 4096; ++f) {
            uint64_t ratio = (4096u << 16) / f;
            U cost = 0;
            while (ratio >= (2u << 16)) {
                ratio >>= 1;
                cost += 256;
            }
            for (int bit = 128; bit; bit >>= 1) {
                ratio = (ratio * ratio) >> 16;
                if (ratio >= (2u << 16)) {
                    ratio >>= 1;
                    cost += bit;
                }
            }
            bits[f] = uint16_t(cost);
        }
    }
};

static const SymbolCost symbol_cost;

static void normalize_counts(const int *hist, int *freq, int *cum, int symbols, int n) {
    if (!n) {
        std::fill_n(freq, symbols, 0);
        std::fill_n(cum, symbols, 0);
        return;
    }
    int active = 0;
    int best = 0;
    for (int s = 0; s < symbols; ++s) {
        active += hist[s] != 0;
        if (hist[s] > hist[best])
            best = s;
    }
    // A frame contains at most 131072 symbols. The upward reciprocal error
    // in a scaled count is less than n * 4096 / 2^48, below 1/n.
    const uint64_t scale = (((uint64_t(1) << 48) + n - 1) / n) * (4096 - active);
    int total = 0;
    for (int s = 0; s < symbols; ++s) {
        freq[s] = (hist[s] != 0) + int((hist[s] * scale) >> 48);
        total += freq[s];
    }
    freq[best] += 4096 - total;
    total = 0;
    for (int s = 0; s < symbols; ++s) {
        cum[s] = total;
        total += freq[s];
    }
}

#if defined(KV_AVX2) || defined(KV_NEON)
#ifdef KV_AVX2
KV_AVX2
#endif
static int column_class(const B *src, int rows) {
    int negative = 0;
#ifdef KV_AVX2
    for (int i = 0; i < rows; i += 32) {
        __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
        negative += __builtin_popcount(unsigned(_mm256_movemask_epi8(bytes)));
    }
#else
    uint8x16_t count = vdupq_n_u8(0);
    for (int i = 0; i < rows; i += 16)
        count = vsubq_u8(count, vreinterpretq_u8_s8(vshrq_n_s8(vreinterpretq_s8_u8(vld1q_u8(src + i)), 7)));
    negative = vaddlvq_u8(count);
#endif
    bool flip = negative > rows / 2;
    return (flip ? 4 : 0) + (flip ? rows - negative : negative) * 8 / (rows + 1);
}
#endif

#if defined(KV_AVX2) || defined(KV_NEON)
#ifdef KV_AVX2
KV_AVX2
#endif
static void matrix_widths(const B *src, int n, int stride, B *widths, int *low_bits) {
    const U *words = reinterpret_cast<const U *>(src);
#ifdef KV_AVX2
    // A nonzero halfword converts to binary32 exactly in every rounding mode.
    __m256i one = _mm256_set1_epi32(1);
    for (int i = 0; i < n; i += 8) {
        __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(words + i));
        __m256i top = _mm256_srli_epi32(value, 16);
        __m256i lower = _mm256_cmpeq_epi32(top, _mm256_setzero_si256());
        __m256i nonzero_half = _mm256_blendv_epi8(top, value, lower);
        __m256i biased = _mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(nonzero_half)), 23);
        __m256i length = _mm256_max_epi32(_mm256_sub_epi32(biased, _mm256_set1_epi32(126)), _mm256_setzero_si256());
        length = _mm256_add_epi32(length, _mm256_andnot_si256(lower, _mm256_set1_epi32(16)));
        __m256i packed = _mm256_shuffle_epi8(length, _mm256_set1_epi32(0x0c080400));
        __m128i pair = _mm_unpacklo_epi32(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(widths + i), pair);
        int *sum = low_bits + (i & (stride - 1));
        __m256i prior = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sum));
        __m256i count = _mm256_max_epi32(_mm256_sub_epi32(length, one), _mm256_setzero_si256());
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sum), _mm256_add_epi32(prior, count));
    }
#else
    for (int i = 0; i < n; i += 8) {
        uint32x4_t a = vsubq_u32(vdupq_n_u32(32), vclzq_u32(vld1q_u32(words + i)));
        uint32x4_t b = vsubq_u32(vdupq_n_u32(32), vclzq_u32(vld1q_u32(words + i + 4)));
        vst1_u8(widths + i, vmovn_u16(vcombine_u16(vmovn_u32(a), vmovn_u32(b))));
        int *sum = low_bits + (i & (stride - 1));
        vst1q_s32(sum, vaddq_s32(vld1q_s32(sum), vreinterpretq_s32_u32(vqsubq_u32(a, vdupq_n_u32(1)))));
        vst1q_s32(sum + 4, vaddq_s32(vld1q_s32(sum + 4), vreinterpretq_s32_u32(vqsubq_u32(b, vdupq_n_u32(1)))));
    }
#endif
}
#endif

template <bool mat, int channel_shift>
static int encode_frame(B *out, const B *src, int n, bool simd = has_simd()) {
    constexpr int models = channel_shift ? 4 : 1;
    constexpr int symbols = mat ? 33 : 256;
    const int columns = channel_shift ? n >> channel_shift : 0;
    const int raw = n * (mat ? 4 : 1);
    out[0] = 2;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    if (!mat && memcmp(src + 1, src, n - 1) == 0) {
        out[0] = 1;
        out[2] = *src;
        return 4;
    }

    // Independent counters avoid a load dependency between repeated symbols.
    int hist[models][4][symbols] = {};
    int population[models] = {};
    B classes[512];
    B widths[mat ? 16384 : 1];
    int low_bits[128] = {};
    const int stride = lanes(n);
    if (channel_shift) {
        constexpr int rows = 1 << channel_shift;
        for (int col = 0; col < columns; ++col) {
            const B *bytes = src + col * rows;
            int group;
#if defined(KV_AVX2) || defined(KV_NEON)
            if (simd)
                group = column_class(bytes, rows);
            else
#endif
                group = classify(bytes, rows);
            classes[col] = B(group);
            auto &counts = hist[group & 3];
            int flip = (group >> 2) * 128;
            for (int i = 0; i < rows; i += 4)
                for (int lane = 0; lane < 4; ++lane)
                    ++counts[lane][bytes[i + lane] ^ flip];
            population[group & 3] += rows;
        }
    }
#if defined(KV_AVX2) || defined(KV_NEON)
    else if (mat && simd) {
        matrix_widths(src, n, stride, widths, low_bits);
        for (int i = 0; i < n; i += 4)
            for (int lane = 0; lane < 4; ++lane)
                ++hist[0][lane][widths[i + lane]];
        population[0] = n;
    }
#endif
    else {
        for (int i = 0; i < n; ++i) {
            int symbol = sym(value(src, i, mat), mat);
            ++hist[0][i & 3][symbol];
            if (mat) {
                widths[i] = B(symbol);
                low_bits[i & (stride - 1)] += symbol ? symbol - 1 : 0;
            }
        }
        population[0] = n;
    }

    int freq[256], cum[256];
    EncodeSymbol table[models * symbols];
#if defined(KV_AVX2) || defined(KV_NEON)
    uint64_t entries[models * symbols];
    const bool vector = simd && stride >= 8 && n % stride == 0;
#endif
    B tables[models * 576];
    int size = 0;
    int minimum_bits = 0;
    for (int model = 0; model < models; ++model) {
        int *counts = hist[model][0];
        for (int s = 0; s < symbols; ++s)
            counts[s] += hist[model][1][s] + hist[model][2][s] + hist[model][3][s];
        if (simd)
            normalize_counts(counts, freq, cum, symbols, population[model]);
        else
            normalize(counts, freq, cum, symbols, population[model]);
        size += write_table(tables + size, freq, symbols);
        for (int symbol = 0; symbol < symbols; ++symbol) {
            int slot = model * 256 + symbol;
            if (!mat && simd)
                minimum_bits += counts[symbol] * symbol_cost.bits[freq[symbol]];
            const auto entry = encoder_symbol(freq[symbol], cum[symbol]);
#if defined(KV_AVX2) || defined(KV_NEON)
            if (vector) {
                U fields = (U(cum[symbol]) << 16) | (entry.shift << 12) | ((U(freq[symbol]) - 1) & 4095);
                entries[slot] = (uint64_t(entry.reciprocal) << 32) | fields;
            } else
#endif
                table[slot] = entry;
        }
    }
    int directory = 4 + columns + size;
    int pos = directory + 2 * stride;
    // A normalized input contains at least 2048 frequency intervals after
    // renormalization. The two integer truncations lose less than 1/256 bit
    // per symbol. Each terminal state can absorb at most eight extra bits.
    const int payload_limit = raw + 4 - pos - 4 * stride;
    bool can_shrink = mat || !simd || minimum_bits - n - 8 * 256 * stride < int64_t(payload_limit) * 8 * 256;
    if (pos < raw + 4 && can_shrink) {
        memcpy(out + 4, classes, columns);
        memcpy(out + 4 + columns, tables, size);
        for (int first = 0; first < stride;) {
            int width = stride >= 8 ? 8 : 1;
            int cap = raw + 3 - pos;
            int length;
#if defined(KV_AVX2) || defined(KV_NEON)
            if (vector) {
                width = std::min(stride, 16);
                length = put_vectors<channel_shift, mat>(out + pos, cap, src, n, first, stride, entries,
                                                         classes, out + directory + 2 * first, widths, low_bits);
            } else
#endif
            if (width == 8)
                length = put_group<8, mat, channel_shift>(out + pos, cap, src, n, first, stride, table,
                                                          classes, out + directory + 2 * first, widths, low_bits);
            else
                length = put_group<1, mat, channel_shift>(out + pos, cap, src, n, first, stride, table,
                                                          classes, out + directory + 2 * first, widths, low_bits);
            if (length < 0)
                break;
            pos += length;
            first += width;
            if (first == stride)
                return pos;
        }
    }
    out[0] = 0;
    if (mat) {
        for (int i = 0; i < n; ++i)
            store32(out + 4 + 4 * i, value(src, i, true));
    } else {
        memcpy(out + 4, src, n);
    }
    return raw + 4;
}

struct DecodeLane {
    const B *low;
    const B *high;
    U state;
};

#include "cpu_gather.h"

// The table contains all three decode operands in one cache lookup. Decode
// interleaved lanes together so the next lookup can start while another retires.
template <int width, int channel_shift, bool matrix>
static bool get_bytes(B *out, int n, int first, int stride, DecodeLane *lanes,
                      const U *tables, const B *classes) {
    U state[width];
    const B *next[width];
    for (int j = 0; j < width; ++j) {
        state[j] = lanes[j].state;
        next[j] = lanes[j].high;
    }
    auto step = [&](int index, int count, bool check) {
        int group = channel_shift ? classes[index >> channel_shift] : 0;
        const U *model = tables + (group & 3) * 4096;
        int flip = (group >> 2) * 128;
        for (int j = 0; j < count; ++j) {
            U entry = model[state[j] & 4095];
            out[index + j] = B(entry ^ flip);
            U updated = ((entry >> 20) + 1) * (state[j] >> 12) + ((entry >> 8) & 4095);
            const B *cursor = next[j];
            bool take = updated < L;
            if (check && take && cursor == lanes[j].low)
                return false;
            U byte = cursor[-1];
            cursor -= take;
            updated = take ? (updated << 8) | byte : updated;
            if (updated < L) {
                if (check && cursor == lanes[j].low)
                    return false;
                updated = (updated << 8) | *--cursor;
            }
            next[j] = cursor;
            state[j] = updated;
        }
        return true;
    };
    int index = first;
    const int block = 16;
    while (n - index >= (block - 1) * stride + width) {
        bool buffered = true;
        for (int j = 0; j < width; ++j)
            buffered &= next[j] - lanes[j].low >= 2 * block;
        if (!buffered)
            break;
        for (int iter = 0; iter < block; ++iter) {
            step(index, width, false);
            index += stride;
        }
    }
    for (; index <= n - width; index += stride)
        if (!step(index, width, true))
            return false;
    if (index < n && !step(index, n - index, true))
        return false;
    for (int j = 0; j < width; ++j) {
        if (state[j] != L || (!matrix && next[j] != lanes[j].low))
            return false;
        lanes[j].high = next[j];
    }
    return true;
}

static bool unpack_residuals(U *out, const B *widths, int n, int stride, const DecodeLane *lanes) {
    for (int l = 0; l < stride; ++l) {
        const B *src = lanes[l].low;
        const int limit = int(lanes[l].high - src) * 8;
        int bit = 0;
        for (int i = l; i < n; i += stride) {
            int symbol = widths[i];
            int count = symbol ? symbol - 1 : 0;
            if (bit + count > limit)
                return false;
            // Each lane ends in a four-byte state. At least four readable bytes
            // follow every low-bit offset, including the end of the bit stream.
            const B *word = src + bit / 8;
            U bits = load32(word) >> (bit % 8);
            if ((bit % 8) + count > 32)
                bits |= U(word[4]) << (32 - bit % 8);
            U leading = U(1) << count;
            out[i] = symbol ? leading | (bits & (leading - 1)) : 0;
            bit += count;
        }
        if ((bit + 7) / 8 != limit / 8)
            return false;
    }
    return true;
}

// Seven spare entries let the last eight-slot store cross a frequency boundary.
// The next symbol overwrites its slots before the table is used.
static int expand_table(const B *in, int bytes, U *table, int symbols, bool populate) {
    int pos = 0;
    int mask_bytes = (symbols + 7) / 8;
    if (mask_bytes > bytes)
        return -1;
    const B *presence = in;
    pos += mask_bytes;
    int total = 0;
    for (int index = 0; index < mask_bytes; ++index) {
        unsigned mask = presence[index];
        if (index == symbols / 8)
            mask &= (1u << (symbols % 8)) - 1;
        while (mask) {
            int symbol = index * 8 + lg(mask & -mask) - 1;
            mask &= mask - 1;
            if (pos == bytes)
                return -1;
            int count = in[pos++];
            if (count & 128) {
                if (pos == bytes)
                    return -1;
                count = (count & 127) + 128 * in[pos++];
            }
            if (!count || count > 4096 - total)
                return -1;
            if (populate) {
                U entry = ((U(count) - 1) << 20) | U(symbol);
                U *slots = table + total;
                for (int offset = 0; offset < count; offset += 8)
                    for (int lane = 0; lane < 8; ++lane)
                        slots[offset + lane] = entry + (U(offset + lane) << 8);
            }
            total += count;
        }
    }
    if (total != 4096 && (total || populate))
        return -1;
    return pos;
}

template <bool mat, int channel_shift>
static bool decode_frame(B *out, const B *in, int bytes, int n, [[maybe_unused]] bool simd) {
    if (bytes < 4)
        return false;
    if (in[0] == 0) {
        int raw = n * (mat ? 4 : 1);
        if (bytes != raw + 4)
            return false;
        if (mat) {
            for (int i = 0; i < n; ++i)
                reinterpret_cast<U *>(out)[i] = load32(in + 4 + 4 * i);
        } else {
            memcpy(out, in + 4, raw);
        }
        return true;
    }
    if (in[0] == 1) {
        if (mat || bytes != 4)
            return false;
        memset(out, in[2], n);
        return true;
    }
    constexpr int models = channel_shift ? 4 : 1;
    constexpr int symbols = mat ? 33 : 256;
    const int columns = channel_shift ? n >> channel_shift : 0;
    int pos = 4 + columns;
    if (in[0] != 2 || pos > bytes)
        return false;
    const B *classes = in + 4;
    int used = channel_shift ? 0 : 1;
    for (int i = 0; i < columns; ++i) {
        if (classes[i] > 7)
            return false;
        used |= 1 << (classes[i] & 3);
    }
    U tables[models * 4096 + 7];
    for (int model = 0; model < models; ++model) {
        int length = expand_table(in + pos, bytes - pos, tables + model * 4096,
                                  symbols, used & (1 << model));
        if (length < 0)
            return false;
        pos += length;
    }

    int stride = lanes(n);
    int directory = pos;
    pos += 2 * stride;
    if (pos > bytes)
        return false;
    DecodeLane readers[128];
    for (int i = 0; i < stride; ++i) {
        int length = in[directory + 2 * i] + 256 * in[directory + 2 * i + 1];
        if (length < 4 || length > bytes - pos)
            return false;
        const B *start = in + pos;
        pos += length;
        const B *end = in + pos - 4;
        U state = load32(end);
        if (state < L || state >= (1u << 31))
            return false;
        readers[i] = {start, end, state};
    }
    if (pos != bytes)
        return false;
    B widths[mat ? 16384 : 1];
    B *symbols_out = mat ? widths : out;
    bool gathered = false;
#if defined(KV_SVE2)
    static const bool sve2 = has_sve2();
    gathered = simd && sve2 && stride >= 8;
#elif defined(KV_AVX2)
    gathered = simd && stride >= 8 && n % stride == 0;
#endif
#if defined(KV_SVE2) || defined(KV_AVX2)
    if (gathered && !gather_decode<channel_shift, mat>(symbols_out, n, stride, readers, tables, classes, in))
        return false;
#endif
    if (!gathered) {
        for (int first = 0; first < stride;) {
            int width = stride >= 4 ? 4 : 1;
            bool ok = width == 4 ?
                get_bytes<4, channel_shift, mat>(symbols_out, n, first, stride, readers + first, tables, classes) :
                get_bytes<1, channel_shift, mat>(symbols_out, n, first, stride, readers + first, tables, classes);
            if (!ok)
                return false;
            first += width;
        }
    }
#if defined(KV_SVE2) || defined(KV_AVX2)
    if (mat && gathered)
        return gather_residuals(reinterpret_cast<U *>(out), widths, n, stride, readers, in);
#endif
    return !mat || unpack_residuals(reinterpret_cast<U *>(out), widths, n, stride, readers);
}
