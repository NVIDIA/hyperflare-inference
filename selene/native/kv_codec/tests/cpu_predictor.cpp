// SPDX-License-Identifier: Apache-2.0

#include "../cpu.cpp"
#include <array>
#include <cassert>

static void check_frequency_tables() {
    for (int count : {0, 1, 32, 255, 256, 4095, 4096, 65535, 131071, 131072}) {
        int histogram[256] = {};
        for (int i = 0; i < count; ++i)
            ++histogram[(i % 7) ? i % 33 : i % 256];
        int freq[256], start[256], reference[256], reference_start[256];
        normalize_counts(histogram, freq, start, 256, count);
        normalize(histogram, reference, reference_start, 256, count);
        assert(std::equal(freq, freq + 256, reference));
        assert(std::equal(start, start + 256, reference_start));
        B bytes[576];
        int length = write_table(bytes, freq, 256);
        U table[4096 + 7];
        assert(expand_table(bytes, length, table, 256, count != 0) == length);
        for (int symbol = 0; symbol < 256; ++symbol)
            for (int offset = 0; offset < freq[symbol]; ++offset) {
                U entry = ((U(freq[symbol]) - 1) << 20) | (U(offset) << 8) | U(symbol);
                assert(table[start[symbol] + offset] == entry);
            }
    }
}

int main() {
    check_frequency_tables();
    std::array<U, 128 * 128> source, reference, actual, restored;
    U prng = 20261120;
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        assert(std::fesetround(mode) == 0);
        for (size_t i = 0; i < source.size(); ++i) {
            prng = 1664525u * prng + 1013904223u;
            const U exponent[] = {76, 77, 126, 127, 128, 177, 178, 255};
            source[i] = (prng & 0x807fffffu) | (exponent[i % 8] << 23);
            if (i % 11 == 0)
                source[i] = (i & 1) ? 0x80000000u : 1u;
        }
        predict_matrix<false>(reinterpret_cast<B *>(source.data()), reference.data(), false);
        matrix(reinterpret_cast<B *>(source.data()), actual.data(), false);
        assert(actual == reference && std::fegetround() == mode);
        matrix(reinterpret_cast<B *>(restored.data()), reference.data(), true);
        assert(restored == source && std::fegetround() == mode);
    }
}
