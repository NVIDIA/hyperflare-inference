// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include "../core.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>
struct Shape {
    int nf, raw, work, arena, wire;
};
struct DeviceSpan {
    uint64_t device;
    int offset, length;
};
extern "C" void *kv_gpu_open(int, int, int, int, int, const char **, int *, size_t *);
extern "C" void kv_gpu_close(void *);
extern "C" int kv_gpu_encode(void *, Shape, const Frame *, const DeviceSpan *, int, B *, int, U *, U *);
extern "C" int kv_gpu_decode(void *, Shape, const Frame *, const B *, int, U, const DeviceSpan *, int);
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
    auto *ctx = kv_gpu_open(65536, 1, 65536, 65540, 65596, &operation, &code, &bytes);
    CHECK(ctx);
    for (int trial = 0; trial < 4; ++trial) {
        int raw = trial == 3 ? 65536 : 4096;
        std::vector<B> original(raw), expected, packet(raw), back(raw, 255);
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
        Shape s{1, raw, raw, raw + 4, raw + 60};
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
        CHECK(cudaMemcpy(dev, original.data(), raw, cudaMemcpyHostToDevice) == cudaSuccess);
        U stored = 0, orig = 0;
        int length = kv_gpu_encode(ctx, s, &f, &span, 1, packet.data(), raw, &stored, &orig);
        CHECK(length > 0);
        packet.resize(length);
        CHECK(packet == cpu && stored == crc(cpu) && orig == crc(original));
        CHECK(kv_cpu_frame(&f, back.data(), packet.data() + 56, length - 56, scratch.data(), true) >= 0 &&
              back == original);
        CHECK(cudaMemset(dev, 255, raw) == cudaSuccess);
        CHECK(kv_gpu_decode(ctx, s, &f, cpu.data(), cpu.size(), orig ^ 1, &span, 1) != 0);
        CHECK(cudaMemcpy(back.data(), dev, raw, cudaMemcpyDeviceToHost) == cudaSuccess);
        for (B b : back)
            CHECK(b == 255);
        CHECK(!kv_gpu_decode(ctx, s, &f, cpu.data(), cpu.size(), orig, &span, 1));
        CHECK(cudaMemcpy(back.data(), dev, raw, cudaMemcpyDeviceToHost) == cudaSuccess && back == original);
        printf("%s SM%d%d fixture=%d bytes=%d crc=%08x cross-read exact\n", prop.name, prop.major, prop.minor,
               trial, length, stored);
    }
    kv_gpu_close(ctx);
    CHECK(cudaFree(dev) == cudaSuccess);
    return 0;
}
