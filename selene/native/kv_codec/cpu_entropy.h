// SPDX-License-Identifier: Apache-2.0
#pragma once

struct EncodeSymbol {
    U limit;
    U reciprocal;
    U shift;
    U complement;
    U bias;
};

static EncodeSymbol encoder_symbol(int frequency, int start) {
    if (frequency == 0)
        return {};
    const U f = frequency;
    if (f == 1)
        return {f << 19, UINT32_MAX, 0, 4095, U(start) + 4095};
    const U shift = lg(f - 1);
    const U reciprocal = U(((uint64_t(1) << (31 + shift)) + f - 1) / f);
    return {f << 19, reciprocal, shift - 1, 4096 - f, U(start)};
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
            B *next_low = out + used;
            uint64_t bits = 0;
            int count = 0;
            for (int i = first + j; i < n; i += stride) {
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

template <bool mat, int channel_shift>
static int encode_frame(B *out, const B *src, int n) {
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

    int hist[models][256] = {};
    int population[models] = {};
    B classes[512];
    B widths[mat ? 16384 : 1];
    int low_bits[128] = {};
    const int stride = lanes(n);
    if (channel_shift) {
        constexpr int rows = 1 << channel_shift;
        for (int col = 0; col < columns; ++col) {
            const B *bytes = src + col * rows;
            int group = classify(bytes, rows);
            classes[col] = B(group);
            int *counts = hist[group & 3];
            int flip = (group >> 2) * 128;
            for (int i = 0; i < rows; ++i)
                ++counts[bytes[i] ^ flip];
            population[group & 3] += rows;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            int symbol = sym(value(src, i, mat), mat);
            ++hist[0][symbol];
            if (mat) {
                widths[i] = B(symbol);
                low_bits[i % stride] += symbol ? symbol - 1 : 0;
            }
        }
        population[0] = n;
    }

    int freq[256], cum[256];
    EncodeSymbol table[models * 256];
    B tables[models * 576];
    int size = 0;
    for (int model = 0; model < models; ++model) {
        normalize(hist[model], freq, cum, symbols, population[model]);
        size += write_table(tables + size, freq, symbols);
        for (int symbol = 0; symbol < symbols; ++symbol)
            table[model * 256 + symbol] = encoder_symbol(freq[symbol], cum[symbol]);
    }
    int directory = 4 + columns + size;
    int pos = directory + 2 * stride;
    if (pos < raw + 4) {
        memcpy(out + 4, classes, columns);
        memcpy(out + 4 + columns, tables, size);
        for (int first = 0; first < stride;) {
            int width = stride >= 8 ? 8 : 1;
            int cap = raw + 3 - pos;
            int length = width == 8 ?
                put_group<8, mat, channel_shift>(out + pos, cap, src, n, first, stride, table, classes, out + directory + 2 * first, widths, low_bits) :
                put_group<1, mat, channel_shift>(out + pos, cap, src, n, first, stride, table, classes, out + directory + 2 * first, widths, low_bits);
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
    U tables[models * 4096];
    for (int model = 0; model < models; ++model) {
        int frequency[256], start[256];
        int length = read_table(in + pos, bytes - pos, frequency, start, symbols);
        if (length < 0)
            return false;
        pos += length;
        if (!(used & (1 << model)))
            continue;
        if (start[symbols - 1] + frequency[symbols - 1] != 4096)
            return false;
        for (int symbol = 0; symbol < symbols; ++symbol) {
            int f = frequency[symbol];
            U entry = ((U(f) - 1) << 20) | U(symbol);
            U *slots = tables + model * 4096 + start[symbol];
            for (int offset = 0; offset < f; ++offset)
                slots[offset] = entry + (U(offset) << 8);
        }
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
    return !mat || unpack_residuals(reinterpret_cast<U *>(out), widths, n, stride, readers);
}
