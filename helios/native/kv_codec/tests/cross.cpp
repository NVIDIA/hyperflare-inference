// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include "core.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>
struct Shape {
    int nf, raw, work, wire;
};
struct DeviceSpan {
    uint64_t device;
    int offset, length;
};
struct HostSpan {
    const B *data;
    size_t length;
};
struct RestoreUnit {
    const HostSpan *chunks;
    int nchunks, length;
    U stored, original;
    const DeviceSpan *spans;
    int count;
};
extern "C" int kv_gpu_decode_spans(void *, Shape, const Frame *, const RestoreUnit *, int, uint64_t * = nullptr);
extern "C" void *kv_gpu_open(int, int, int, int, const char **, int *, size_t *);
extern "C" void kv_gpu_close(void *);
extern "C" int kv_cpu_frame(const Frame *, B *, B *, int, U *, bool);
#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        if (!(x)) {                                                                                          \
            fprintf(stderr, "failed at %d: %s\n", __LINE__, #x);                                             \
            return 1;                                                                                        \
        }                                                                                                    \
    } while (0)
U crc(const std::vector<B> &v) {
    U x = ~0u;
    for (auto b : v) {
        x ^= b;
        for (int j = 0; j < 8; ++j)
            x = (x >> 1) ^ ((0u - (x & 1)) & 0xedb88320u);
    }
    return ~x;
}
int main() {
    CHECK(cudaSetDevice(0) == cudaSuccess && cudaFree(nullptr) == cudaSuccess);
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
    B *dev = nullptr;
    CHECK(cudaMalloc(&dev, 65536) == cudaSuccess);
    const char *operation = nullptr;
    int code = 0;
    size_t bytes = 0;
    auto *ctx = kv_gpu_open(65536, 16, 65536, 65596, &operation, &code, &bytes);
    CHECK(ctx);
    std::vector<B> prior_packet, prior_raw;
    for (int trial = 0; trial < 4; ++trial) {
        int raw = trial == 3 ? 65536 : 4096;
        std::vector<B> original(raw), expected, back(raw, 255);
        if (trial < 3) {
            for (int i = 0; i < raw; ++i)
                original[i] = (i / (8 + 2 * trial) + trial) % (2 + trial);
            const char *file = trial == 0 ? "page0.sgc" : trial == 1 ? "page1.sgc" : "row.sgc";
            std::ifstream in(file, std::ios::binary);
            CHECK(in);
            expected.assign(std::istreambuf_iterator<char>(in), {});
        } else {
            for (int i = 0; i < 16384; ++i) {
                float val = float(i % 128) * 0.125f;
                memcpy(original.data() + 4 * i, &val, 4);
            }
        }
        Frame f{0, raw, 0, trial == 3 ? 16384 : raw, trial == 3 ? 1 : 0, 0, 0, 0, 0, 0, 0};
        Shape s{1, raw, raw, raw + 60};
        DeviceSpan span{uintptr_t(dev), 0, raw};
        std::vector<B> cpu(raw + 60);
        std::vector<U> scratch(raw / 4);
        int n = kv_cpu_frame(&f, original.data(), cpu.data() + 56, raw + 4, scratch.data(), false);
        CHECK(n >= 4);
        store32(cpu.data(), 0x31434753);
        store32(cpu.data() + 4, 1);
        store32(cpu.data() + 8, 1);
        store32(cpu.data() + 12, raw);
        memcpy(cpu.data() + 16, &f, 32);
        store32(cpu.data() + 48, 56);
        store32(cpu.data() + 52, n);
        cpu.resize(n + 56);
        if (trial < 3)
            CHECK(cpu == expected);
        U stored = crc(cpu), orig = crc(original);
        CHECK(kv_cpu_frame(&f, back.data(), cpu.data() + 56, n, scratch.data(), true) >= 0 &&
              back == original);
        CHECK(cudaMemset(dev, 255, raw) == cudaSuccess);
        HostSpan chunk{cpu.data(), cpu.size()};
        RestoreUnit unit{&chunk, 1, int(cpu.size()), stored, orig ^ 1, &span, 1};
        CHECK(kv_gpu_decode_spans(ctx, s, &f, &unit, 1) == -2);
        CHECK(cudaMemcpy(back.data(), dev, raw, cudaMemcpyDeviceToHost) == cudaSuccess);
        for (B b : back)
            CHECK(b == 255);
        unit.original = orig;
        CHECK(!kv_gpu_decode_spans(ctx, s, &f, &unit, 1));
        CHECK(cudaMemcpy(back.data(), dev, raw, cudaMemcpyDeviceToHost) == cudaSuccess && back == original);
        if (trial == 2) {
            DeviceSpan rotated[] = {{uintptr_t(dev + 1), 0, raw - 1}, {uintptr_t(dev), raw - 1, 1}};
            unit.spans = rotated;
            unit.count = 2;
            CHECK(!kv_gpu_decode_spans(ctx, s, &f, &unit, 1));
            CHECK(cudaMemcpy(back.data(), dev, raw, cudaMemcpyDeviceToHost) == cudaSuccess);
            CHECK(back[0] == original.back() && !memcmp(back.data() + 1, original.data(), raw - 1));
        }
        if (trial == 0) {
            const B mode = cpu[56];
            cpu[56] = 255;
            unit.stored = crc(cpu);
            CHECK(cudaMemset(dev, 0xa5, raw) == cudaSuccess);
            CHECK(kv_gpu_decode_spans(ctx, s, &f, &unit, 1) == -2);
            CHECK(cudaMemcpy(back.data(), dev, raw, cudaMemcpyDeviceToHost) == cudaSuccess);
            for (B b : back)
                CHECK(b == 0xa5);
            cpu[56] = mode;
            prior_packet = cpu;
            prior_raw = original;
        } else if (trial == 1) {
            HostSpan chunks[] = {{prior_packet.data(), prior_packet.size()}, {cpu.data(), 7},
                                 {cpu.data() + 7, cpu.size() - 7}};
            DeviceSpan second{uintptr_t(dev + raw), 0, raw};
            RestoreUnit units[] = {{chunks, 1, int(prior_packet.size()), crc(prior_packet), crc(prior_raw), &span, 1},
                                   {chunks + 1, 2, int(cpu.size()), stored, orig, &second, 1}};
            std::vector<B> both(2 * raw, 0xa5);
            CHECK(cudaMemcpy(dev, both.data(), both.size(), cudaMemcpyHostToDevice) == cudaSuccess);
            units[1].stored ^= 1;
            CHECK(kv_gpu_decode_spans(ctx, s, &f, units, 2) == -3);
            units[1].stored = stored;
            // The second packet fails after entropy decoding. Neither page may become visible.
            units[1].original ^= 1;
            CHECK(kv_gpu_decode_spans(ctx, s, &f, units, 2) == -2);
            CHECK(cudaMemcpy(both.data(), dev, both.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
            for (B b : both)
                CHECK(b == 0xa5);
            units[1].original = orig;
            CHECK(kv_gpu_decode_spans(ctx, s, &f, units, 2) == 0);
            CHECK(cudaMemcpy(both.data(), dev, both.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
            CHECK(!memcmp(both.data(), prior_raw.data(), raw) && !memcmp(both.data() + raw, original.data(), raw));
        }
        printf("%s SM%d%d fixture=%d bytes=%d crc=%08x cross-read exact\n", prop.name, prop.major, prop.minor,
               trial, int(cpu.size()), stored);
    }
    kv_gpu_close(ctx);
    CHECK(cudaFree(dev) == cudaSuccess);
    return 0;
}
