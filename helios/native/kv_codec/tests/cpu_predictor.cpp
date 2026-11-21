// SPDX-License-Identifier: Apache-2.0

#include "../cpu.cpp"
#include <array>
#include <cassert>

int main() {
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
