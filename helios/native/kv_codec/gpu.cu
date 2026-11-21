// SPDX-License-Identifier: Apache-2.0
#include "core.h"
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <memory>
#include <new>
#include "entropy_fast.cuh"
struct Shape {
    int nf, raw, work, arena, wire;
};
// Swizzle the decoder tile so a pivot column spans shared-memory banks.
template <bool Decode> __device__ __forceinline__ int matrix_index(int i, int j) {
    return i * 128 + (Decode ? j ^ (i & 31) : j);
}

template <bool Decode> __device__ __forceinline__ U matrix_word(const U *input, U &cell, int i, int j) {
    U word;
    if constexpr (Decode) {
        word = restore(input[i * 128 + j], cell);
        cell = word;
    } else {
        word = input[j * 128 + i];
        cell = residual(word, cell);
    }
    return bounded(word);
}

template <bool Decode> __device__ __forceinline__ void matrix_local(const U *input, U *output = nullptr) {
    int t = threadIdx.x;
    extern __shared__ U p[];
    __shared__ U v[128], u[128];
    __shared__ int active[128], pivot;

    for (int i = t; i < 16384; i += 256)
        p[i] = 0;
    if (t < 128)
        active[t] = 1;

    __syncthreads();
    for (int k = 0; k < 64; k++) {
        if (t < 128 && active[t]) {
            U &cell = p[matrix_index<Decode>(k, t)];
            U pred = cell, w = matrix_word<Decode>(input, cell, k, t);
            v[t] = minus_word(w, pred);
        }
        __syncthreads();
        if (t < 32) {
            unsigned best = 0;
            int c = 128;
            for (int j = t; j < 128; j += 32) {
                unsigned m = v[j] & 0x7fffffffu;
                if (active[j] && (m > best || (m == best && j < c))) {
                    best = m;
                    c = j;
                }
            }
            for (int sh = 16; sh; sh /= 2) {
                unsigned x = __shfl_down_sync(~0u, best, sh);
                int y = __shfl_down_sync(~0u, c, sh);
                if (x > best || (x == best && y < c)) {
                    best = x;
                    c = y;
                }
            }
            __syncwarp(); // All lanes must finish reading active[] before lane 0 clears the pivot.
            if (!t) {
                pivot = c;
                active[c] = 0;
            }
        }
        __syncthreads();
        int c = pivot;
        if (t > k && t < 128) {
            U &cell = p[matrix_index<Decode>(t, c)];
            U pred = cell, w = matrix_word<Decode>(input, cell, t, c);
            u[t] = divide(minus_word(w, pred), v[c]);
        }
        __syncthreads();
        for (int i = k + 1 + t / 128; i < 128; i += 2) {
            int j = t % 128;
            if (active[j])
                p[matrix_index<Decode>(i, j)] = plus(p[matrix_index<Decode>(i, j)], times(u[i], v[j]));
        }
        __syncthreads();
    }
    for (int i = 64 + t / 128; i < 128; i += 2) {
        int j = t % 128;
        if (active[j])
            matrix_word<Decode>(input, p[matrix_index<Decode>(i, j)], i, j);
    }
    if constexpr (Decode) {
        __syncthreads();
        for (int i = t; i < 16384; i += 256)
            output[i] = p[matrix_index<true>(i % 128, i / 128)];
    }
}

template <bool FAST>
__device__ __forceinline__ void entropy_one(int id, Frame *fs, Shape s, const B *t, B *arena, U *sizes) {
    int b = 0, l = threadIdx.x;
    Frame f = fs[id];

    B *out = arena + size_t(b) * s.arena + f.arena;
    sizes += size_t(b) * s.nf + id;
    bool mat = f.kind == 1;
    int m = mat ? 33 : 256, nl = lanes(f.n), raw = f.n * (mat ? 4 : 1);
    int rows = channel_rows(f.kind), nc = rows ? f.n / rows : 0, models = rows ? 4 : 1;
    __shared__ int hist[1024], freq[1024], cum[1024], len[128], start[128], mode, total, nt[4], at[4];
    __shared__ B cl[512], tables[4][576];
    for (int i = l; i < 256 * models; i += 256)
        hist[i] = 0;
    for (int c = l; c < nc; c += 256)
        cl[c] = classify(t + c * rows, rows);
    __syncthreads();
    bool constant = !mat;
    for (int i = l; i < f.n; i += 256) {
        U val = value(t, i, mat);
        constant = constant && val == t[0];
        atomicAdd(hist + model_symbol(val, i, mat, rows, cl), 1);
    }
    bool uniform = __syncthreads_and(constant);
    if (!l) {
        mode = uniform ? 1 : 2;
        out[1] = out[2] = out[3] = 0;
        if (mode == 1) {
            out[0] = 1;
            out[2] = t[0];
            *sizes = 4;
        }
    }
    __syncthreads();
    if (mode == 1)
        return;
    if (l < models) {
        int pop = 0;
        for (int j = 0; j < m; j++)
            pop += hist[256 * l + j];
        normalize(hist + 256 * l, freq + 256 * l, cum + 256 * l, m, pop);
        nt[l] = write_table(tables[l], freq + 256 * l, m);
    }
    __syncthreads();
    __shared__ unsigned inv[1024];
    if (FAST) {
        for (int j = l; j < 256 * models; j += 256) {
            int symbol = j % 256;
            unsigned f = symbol < m ? freq[j] : 0;
            inv[j] = f > 1 ? unsigned((0xffffffffu / f) + 1) : 0;
        }
        __syncthreads();
    }
    int low = 0;
    if (l < nl)
        lane_size_opt<FAST>(inv, t, f.n, mat, l, nl, freq, cum, low, len[l], rows, cl);
    __syncthreads();
    if (!l) {
        total = 4 + nc;
        for (int j = 0; j < models; j++) {
            at[j] = total;
            total += nt[j];
        }
        total += 2 * nl;
        for (int i = 0; i < nl; i++) {
            start[i] = total;
            total += len[i];
        }
        if (total >= raw + 4) {
            mode = 0;
            total = raw + 4;
        }
        out[0] = mode;
        *sizes = total;
    }
    __syncthreads();
    if (!mode) {
        for (int i = l; i < raw; i += 256)
            out[4 + i] = t[i];
        return;
    }
    for (int c = l; c < nc; c += 256)
        out[4 + c] = cl[c];
    for (int j = 0; j < models; j++)
        for (int i = l; i < nt[j]; i += 256)
            out[at[j] + i] = tables[j][i];
    if (l < nl) {
        int j = start[0] - 2 * nl + 2 * l;
        out[j] = len[l];
        out[j + 1] = len[l] >> 8;
        put_lane_opt<FAST>(inv, out + start[l], t, f.n, mat, l, nl, freq, cum, low, rows, cl);
    }
}

__device__ __forceinline__ void pack_one(int id, Frame *fs, Shape s, B *arena, U *ends, B *wire) {
    int b = 0, l = threadIdx.x;
    Frame f = fs[id];
    wire += size_t(b) * s.wire;
    ends += size_t(b) * s.nf;
    U begin = id ? ends[id - 1] : 0, len = ends[id] - begin, pos = 16 + s.nf * sizeof(Entry) + begin;
    if (!l) {
        if (!id) {
            store32(wire, 0x31434753);
            store32(wire + 4, 1);
            store32(wire + 8, s.nf);
            store32(wire + 12, s.raw);
        }
        Entry dir = {f.base, f.full, f.sub, f.n, f.kind, f.rank, f.layer, f.page, pos, len};
        reinterpret_cast<Entry *>(wire + 16)[id] = dir;
    }
    for (U i = l; i < len; i += 256)
        wire[pos + i] = arena[size_t(b) * s.arena + f.arena + i];
}

// A large channel frame has room in its existing encoder extent for four tables
// and all lane offsets. Packing completes before this space is reused.
struct ChannelTable {
    U transition[4 * 4096];
    int offsets[129];
};

__device__ bool split_lanes(Frame f) {
    return channel_rows(f.kind) && f.n >= int(sizeof(ChannelTable));
}

// The rANS state runs backwards through the lane; matrix low bits run forwards.
template <bool Direct>
__device__ bool decode_lane(B *out, const B *src, int bytes, Frame f, int lane, int nl, const B *symbols,
                            const U *table, const B *classes) {
    if (bytes < 4)
        return false;
    int end = bytes - 4, bit = 0, cached = 0;
    U state = load32(src + end), tail = 0;
    if (state < L)
        return false;
    int rows = channel_rows(f.kind), shift = rows ? lg(rows) - 1 : 0;
    for (int i = lane; i < f.n; i += nl) {
        int c = rows ? classes[i >> shift] : 0;
        int model = c % 4, slot = state & 4095, symbol;
        if constexpr (Direct) {
            U entry = table[model * 4096 + slot];
            symbol = entry & 255;
            state = (1 + ((entry >> 8) & 4095)) * (state >> 12) + (entry >> 20);
        } else {
            symbol = symbols[model * 4096 + slot];
            U entry = table[model * 256 + symbol];
            state = (entry >> 12) * (state >> 12) + slot - (entry & 4095);
        }
        while (state < L) {
            if (!end)
                return false;
            if (!cached) {
                cached = min(end, 4);
                tail = load32(src + end - cached) << (8 * (4 - cached));
            }
            state = (state << 8) | (tail >> 24);
            tail <<= 8;
            --cached;
            --end;
        }
        if (f.kind == 1) {
            int n = lowbits(symbol, true), offset = bit / 8, bit_shift = bit % 8;
            if (bit + n > end * 8)
                return false;
            // The four state bytes at the end also bound the unaligned lookahead.
            U low = load32(src + offset) >> bit_shift;
            if (bit_shift + n > 32)
                low |= U(src[offset + 4]) << (32 - bit_shift);
            reinterpret_cast<U *>(out)[i] = symbol ? (1u << n) | (low & ((1u << n) - 1)) : 0;
            bit += n;
        } else
            out[i] = symbol ^ ((c / 4) * 128);
    }
    return state == L && end == (bit + 7) / 8;
}

template <bool Multi, int Threads>
__device__ __forceinline__ void unentropy_one(Frame f, int id, Shape s, const B *wire, B *mid, B *split,
                                              int *err, ChannelTable *large) {
    int l = threadIdx.x;
    if (!l)
        split[id] = 0;
    Entry h = reinterpret_cast<const Entry *>(wire + 16)[id];
    if (h.len < 4 || h.pos > s.wire || h.len > s.wire - h.pos) {
        if (!l)
            atomicAdd(err, 1);
        return;
    }
    const B *in = wire + h.pos;
    B *t = mid + f.work;
    bool mat = f.kind == 1;
    int m = mat ? 33 : 256, nl = lanes(f.n), raw = f.n * (mat ? 4 : 1);
    if (in[0] == 0) {
        if (h.len != raw + 4) {
            if (!l)
                atomicAdd(err, 1);
            return;
        }
        for (int i = l; i < raw; i += Threads)
            t[i] = in[4 + i];
        return;
    }
    if (in[0] == 1) {
        if (mat || h.len != 4) {
            if (!l)
                atomicAdd(err, 1);
            return;
        }
        for (int i = l; i < f.n; i += Threads)
            t[i] = in[2];
        return;
    }
    int rows = channel_rows(f.kind), nc = rows ? f.n / rows : 0, models = rows ? 4 : 1;
    __shared__ int freq[Multi ? 1024 : 256], cum[Multi ? 1024 : 256], offsets[129];
    __shared__ B cl[512];
    if (in[0] != 2 || h.len < 4 + nc + 2 * nl) {
        if (!l)
            atomicAdd(err, 1);
        return;
    }
    extern __shared__ B symbols[];
    for (int c = l; c < nc; c += Threads)
        cl[c] = in[4 + c];
    __shared__ int directory, nonempty;
    if (!l) {
        int pos = 4 + nc;
        nonempty = 0;
        for (int j = 0; j < models; ++j) {
            int size = read_table(in + pos, h.len - pos, freq + 256 * j, cum + 256 * j, m);
            if (size < 0) {
                pos = -1;
                break;
            }
            if (size > (m + 7) / 8)
                nonempty |= 1 << j;
            pos += size;
        }
        directory = pos;
    }
    __syncthreads();
    int pos = directory, payload = pos + 2 * nl;
    if (pos < 0 || payload > h.len) {
        if (!l)
            atomicAdd(err, 1);
        return;
    }
    using Scan = cub::BlockScan<int, Threads>;
    __shared__ typename Scan::TempStorage scan;
    int length = l < nl ? in[pos + 2 * l] + 256 * in[pos + 2 * l + 1] : 0;
    int prefix, total;
    Scan(scan).ExclusiveSum(length, prefix, total);
    if (l < nl)
        offsets[l] = payload + prefix;
    if (!l)
        offsets[nl] = payload + total;
    bool valid = payload + total == h.len && (models > 1 || nonempty);
    for (int c = l; c < nc; c += Threads)
        valid = valid && cl[c] < 8 && (nonempty & (1 << (cl[c] % 4)));
    if (!__syncthreads_and(valid)) {
        if (!l)
            atomicAdd(err, 1);
        return;
    }
    for (int j = 0; j < models; ++j) {
        for (int sym = l / 32; sym < m; sym += Threads / 32) {
            int base = cum[j * 256 + sym], f = freq[j * 256 + sym];
            for (int i = l % 32; i < f; i += 32)
                symbols[j * 4096 + base + i] = sym;
        }
    }
    __syncthreads();
    if (large) {
        for (int i = l; i < models * 4096; i += Threads) {
            int model = i / 4096;
            if (!(nonempty & (1 << model)))
                continue;
            int symbol = symbols[i], ix = model * 256 + symbol;
            large->transition[i] = (U(i % 4096 - cum[ix]) << 20) | (U(freq[ix] - 1) << 8) | symbol;
        }
        if (l < nl)
            large->offsets[l] = offsets[l];
        if (!l) {
            large->offsets[nl] = offsets[nl];
            split[id] = 1;
        }
        return;
    }
    for (int j = 0; j < models; ++j)
        for (int i = l; i < m; i += Threads) {
            int idx = j * 256 + i;
            cum[idx] = freq[idx] * 4096 + cum[idx];
        }
    __syncthreads();
    if (l < nl && !decode_lane<false>(t, in + offsets[l], length, f, l, nl, symbols,
                                      reinterpret_cast<const U *>(cum), cl))
        atomicAdd(err, 1);
}
__global__ void prefix(U *sz, U *ends, int n) {
    using Scan = cub::BlockScan<U, 256>;
    __shared__ typename Scan::TempStorage temp;
    U carry = 0;
    for (int b = 0; b < n; b += 256) {
        int i = b + threadIdx.x;
        U v = i < n ? sz[i] : 0, sum, total;
        Scan(temp).InclusiveSum(v, sum, total);
        if (i < n)
            ends[i] = carry + sum;
        carry += total;
        __syncthreads();
    }
}
struct Workspace {
    Frame *frames{};
    B *raw{}, *decoded{}, *mid{}, *arena{}, *split{}, *wire{};
    U *sizes{}, *ends{}, *status{};
    void *allocation{};
    cudaStream_t stream{}, restore{};
    cudaEvent_t start{}, end{};
    int max_raw{}, max_frames{}, max_work{}, max_arena{}, max_wire{};
};

struct DeviceSpan {
    uint64_t device;
    int offset, length;
};

__global__ void encode_frames(Workspace w, Shape s) {
    extern __shared__ U tile[];
    for (int id = blockIdx.x; id < s.nf; id += gridDim.x) {
        Frame f = w.frames[id];
        B *stage = f.n <= 65536 ? reinterpret_cast<B *>(tile) : w.mid + f.work;
        if (f.kind == 1)
            matrix_local<false>(reinterpret_cast<const U *>(w.raw + f.base + f.sub));
        else
            for (int i = threadIdx.x; i < f.n; i += blockDim.x)
                stage[i] = gather(f, i, w.raw);
        __syncthreads();
        entropy_one<false>(id, w.frames, s, stage, w.arena, w.sizes);
        __syncthreads();
    }
}

__global__ void pack_frames(Workspace w, Shape s) {
    for (int id = blockIdx.x; id < s.nf; id += gridDim.x)
        pack_one(id, w.frames, s, w.arena, w.ends, w.wire);
}

enum class EntropyKind { Matrix, Bytes, Channels };

template <EntropyKind Kind, int Threads> __global__ void decode_entropy(Workspace w, Shape s) {
    for (int id = blockIdx.x; id < s.nf; id += gridDim.x) {
        Frame f = w.frames[id];
        auto kind = f.kind == 1            ? EntropyKind::Matrix
                    : channel_rows(f.kind) ? EntropyKind::Channels
                                           : EntropyKind::Bytes;
        if (kind != Kind)
            continue;
        constexpr bool multi = Kind == EntropyKind::Channels;
        auto *large = multi && split_lanes(f) ? reinterpret_cast<ChannelTable *>(w.arena + f.arena) : nullptr;
        unentropy_one<multi, Threads>(f, id, s, w.wire, w.mid, w.split, reinterpret_cast<int *>(w.status),
                                      large);
        __syncthreads();
    }
}

__global__ void decode_large_lanes(Workspace w, Shape s) {
    constexpr int Groups = 16;
    for (int task = blockIdx.x; task < s.nf * Groups; task += gridDim.x) {
        int id = task / Groups;
        if (!w.split[id])
            continue;
        Frame f = w.frames[id];
        auto *large = reinterpret_cast<const ChannelTable *>(w.arena + f.arena);
        Entry h = reinterpret_cast<const Entry *>(w.wire + 16)[id];
        const B *in = w.wire + h.pos;
        int nl = lanes(f.n), width = nl / Groups;
        int l = (task % Groups) * width + threadIdx.x;
        if (threadIdx.x < width && !decode_lane<true>(w.mid + f.work, in + large->offsets[l],
                                                      large->offsets[l + 1] - large->offsets[l], f, l, nl,
                                                      nullptr, large->transition, in + 4))
            atomicAdd(w.status, 1);
    }
}

template <bool Matrix> __global__ void invert_frames(Workspace w, Shape s) {
    for (int id = blockIdx.x; id < s.nf; id += gridDim.x) {
        Frame f = w.frames[id];
        if ((f.kind == 1) != Matrix)
            continue;
        const B *mid = w.mid + f.work;
        if constexpr (Matrix)
            matrix_local<true>(reinterpret_cast<const U *>(mid),
                               reinterpret_cast<U *>(w.decoded + f.base + f.sub));
        else
            for (int i = threadIdx.x; i < f.n; i += blockDim.x)
                w.decoded[address(f, i)] = mid[i];
        if constexpr (Matrix)
            __syncthreads();
    }
}

static void decode_frames(Workspace w, Shape s, int blocks, cudaStream_t stream) {
    // Matrix frames have 32 entropy lanes. Only their inverse needs the 64 KiB predictor tile.
    decode_entropy<EntropyKind::Matrix, 64><<<blocks, 64, 4096, stream>>>(w, s);
    decode_entropy<EntropyKind::Bytes, 256><<<blocks, 256, 4096, stream>>>(w, s);
    decode_entropy<EntropyKind::Channels, 256><<<blocks, 256, 16384, stream>>>(w, s);
    decode_large_lanes<<<blocks, 32, 0, stream>>>(w, s);
    invert_frames<true><<<blocks, 256, 65536, stream>>>(w, s);
    invert_frames<false><<<blocks, 256, 0, stream>>>(w, s);
}

// The CRC of two concatenated intervals is shift(crc(left), len(right)) XOR crc(right).
struct CrcTables {
    U byte[256]{}, shift[32][32]{};
    static constexpr U apply(const U *matrix, U value) {
        U out = 0;
        for (int j = 0; value; ++j, value >>= 1)
            if (value & 1)
                out ^= matrix[j];
        return out;
    }
    constexpr CrcTables() {
        for (U i = 0; i < 256; ++i) {
            U v = i;
            for (int j = 0; j < 8; ++j)
                v = (v >> 1) ^ ((v & 1) ? 0xedb88320u : 0u);
            byte[i] = v;
        }
        for (int i = 0; i < 32; ++i) {
            U v = 1u << i;
            shift[0][i] = (v >> 8) ^ byte[v & 255];
        }
        for (int j = 1; j < 32; ++j)
            for (int i = 0; i < 32; ++i)
                shift[j][i] = apply(shift[j - 1], shift[j - 1][i]);
    }
};
__device__ __constant__ CrcTables crc_tables{};

__global__ void check_bytes(const B *source, const B *expected, int length, U *crc, U *error) {
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 256; i < length;
         i += gridDim.x * blockDim.x * 256) {
        int end = min(i + 256, length);
        U sum = ~0u, mismatch = 0;
        for (int j = i; j < end; ++j) {
            B v = source[j];
            sum = (sum >> 8) ^ crc_tables.byte[(sum ^ v) & 255];
            if (expected)
                mismatch |= v ^ expected[j];
        }
        if (mismatch)
            atomicOr(error, 1u);
        sum = ~sum;
        for (int tail = length - end, j = 0; tail; tail >>= 1, ++j) {
            if (!(tail & 1))
                continue;
            U shifted = 0;
            while (sum) {
                int bit = __ffs(sum) - 1;
                shifted ^= crc_tables.shift[j][bit];
                sum &= sum - 1;
            }
            sum = shifted;
        }
        atomicXor(crc, sum);
    }
}

static void close_workspace(Workspace *w) {
    if (!w)
        return;
    if (w->stream)
        cudaStreamSynchronize(w->stream);
    if (w->restore)
        cudaStreamSynchronize(w->restore);
    if (w->start)
        cudaEventDestroy(w->start);
    if (w->end)
        cudaEventDestroy(w->end);
    if (w->restore)
        cudaStreamDestroy(w->restore);
    if (w->allocation)
        cudaFree(w->allocation);
    if (w->stream)
        cudaStreamDestroy(w->stream);
    delete w;
}

extern "C" Workspace *kv_gpu_open(int raw, int nf, int work, int arena, int wire, const char **failure,
                                  int *code, size_t *bytes) {
    *failure = "host workspace descriptor";
    *code = int(cudaErrorMemoryAllocation);
    auto failed = [&](const char *operation, cudaError_t status) {
        if (status == cudaSuccess)
            return false;
        *failure = operation;
        *code = int(status);
        return true;
    };
    std::unique_ptr<Workspace, decltype(&close_workspace)> owned(new (std::nothrow) Workspace,
                                                                 close_workspace);
    Workspace *w = owned.get();
    if (!w)
        return nullptr;
    w->max_raw = raw;
    w->max_frames = nf;
    w->max_work = work;
    w->max_arena = arena;
    w->max_wire = wire;
    int low = 0, high = 0;
    cudaDeviceGetStreamPriorityRange(&low, &high);
    if (failed("create codec stream", cudaStreamCreateWithPriority(&w->stream, cudaStreamNonBlocking, low))) {
        return nullptr;
    }
    if (failed("create restore stream",
               cudaStreamCreateWithPriority(&w->restore, cudaStreamNonBlocking, high)) ||
        failed("create start event", cudaEventCreate(&w->start)) ||
        failed("create end event", cudaEventCreate(&w->end))) {
        return nullptr;
    }
    size_t pos = 0;
    auto reserve = [&pos](size_t bytes) {
        const auto p = pos;
        pos += (bytes + 255) & ~size_t(255);
        return p;
    };
    const size_t f = reserve(sizeof(Frame) * nf), r = reserve(raw), d = reserve(raw), m = reserve(work),
                 a = reserve(arena), e = reserve(nf), p = reserve(wire), z = reserve(4 * nf),
                 n = reserve(4 * nf), v = reserve(12);
    *bytes = pos;
    if (failed("cudaMalloc", cudaMalloc(&w->allocation, pos)) ||
        failed("configure encode kernel",
               cudaFuncSetAttribute(encode_frames, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)) ||
        failed(
            "configure inverse kernel",
            cudaFuncSetAttribute(invert_frames<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536))) {
        return nullptr;
    }
    B *base = static_cast<B *>(w->allocation);
    w->frames = reinterpret_cast<Frame *>(base + f);
    w->raw = base + r;
    w->decoded = base + d;
    w->mid = base + m;
    w->arena = base + a;
    w->split = base + e;
    w->wire = base + p;
    w->sizes = reinterpret_cast<U *>(base + z);
    w->ends = reinterpret_cast<U *>(base + n);
    w->status = reinterpret_cast<U *>(base + v);
    return owned.release();
}
extern "C" void kv_gpu_close(Workspace *w) {
    close_workspace(w);
}
extern "C" uint64_t kv_gpu_transform_ns(Workspace *w) {
    float ms = 0;
    return cudaEventElapsedTime(&ms, w->start, w->end) == cudaSuccess ? uint64_t(ms * 1000000) : 0;
}

static bool prepare(Workspace *w, Shape s, const Frame *frames, cudaStream_t stream) {
    return s.nf > 0 && s.nf <= w->max_frames && s.raw > 0 && s.raw <= w->max_raw && s.work <= w->max_work &&
           s.arena <= w->max_arena && s.wire <= w->max_wire &&
           cudaMemcpyAsync(w->frames, frames, sizeof(Frame) * s.nf, cudaMemcpyHostToDevice, stream) ==
               cudaSuccess &&
           cudaMemsetAsync(w->status, 0, 12, stream) == cudaSuccess;
}

// Every return, including a failed enqueue, fences work that may still touch a caller's spans.
struct Fence {
    cudaStream_t stream;
    ~Fence() {
        cudaStreamSynchronize(stream);
    }
};
extern "C" int kv_gpu_capture(Workspace *w, Shape s, const Frame *frames, const DeviceSpan *spans,
                              int count) {
    Fence fence{w->stream};
    if (!prepare(w, s, frames, w->stream) || cudaMemsetAsync(w->raw, 0, s.raw, w->stream))
        return -1;
    for (int i = 0; i < count; ++i) {
        const auto &p = spans[i];
        if (p.offset < 0 || p.length < 0 || p.offset > s.raw - p.length ||
            cudaMemcpyAsync(w->raw + p.offset, reinterpret_cast<void *>(p.device), p.length,
                            cudaMemcpyDeviceToDevice, w->stream))
            return -1;
    }
    return cudaStreamSynchronize(w->stream) ? -1 : 0;
}
extern "C" int kv_gpu_encode_captured(Workspace *w, Shape s, B *output, int capacity, U *stored_crc,
                                      U *original_crc) {
    Fence fence{w->stream};
    if (cudaEventRecord(w->start, w->stream))
        return -1;
    const int blocks = min(8, s.nf);
    encode_frames<<<blocks, 256, 65536, w->stream>>>(*w, s);
    prefix<<<1, 256, 0, w->stream>>>(w->sizes, w->ends, s.nf);
    pack_frames<<<blocks, 256, 0, w->stream>>>(*w, s);
    decode_frames(*w, s, blocks, w->stream);
    check_bytes<<<8, 256, 0, w->stream>>>(w->decoded, w->raw, s.raw, w->status + 1, w->status);
    if (cudaEventRecord(w->end, w->stream))
        return -1;
    U checks[3], length;
    if (cudaGetLastError() || cudaMemcpyAsync(checks, w->status, 12, cudaMemcpyDeviceToHost, w->stream) ||
        cudaMemcpyAsync(&length, w->ends + s.nf - 1, 4, cudaMemcpyDeviceToHost, w->stream) ||
        cudaStreamSynchronize(w->stream))
        return -1;
    if (checks[0])
        return -2;
    length += 16 + 40 * s.nf;
    *original_crc = checks[1];
    if (length >= U(s.raw))
        return 0;
    if (length > U(capacity))
        return -1;
    check_bytes<<<8, 256, 0, w->stream>>>(w->wire, nullptr, length, w->status + 2, w->status);
    if (cudaGetLastError() ||
        cudaMemcpyAsync(stored_crc, w->status + 2, 4, cudaMemcpyDeviceToHost, w->stream) ||
        cudaMemcpyAsync(output, w->wire, length, cudaMemcpyDeviceToHost, w->stream) ||
        cudaStreamSynchronize(w->stream))
        return -1;
    return int(length);
}
extern "C" int kv_gpu_encode(Workspace *w, Shape s, const Frame *frames, const DeviceSpan *spans, int count,
                             B *output, int capacity, U *stored_crc, U *original_crc) {
    if (kv_gpu_capture(w, s, frames, spans, count))
        return -1;
    return kv_gpu_encode_captured(w, s, output, capacity, stored_crc, original_crc);
}
struct HostSpan {
    const B *data;
    size_t length;
};
extern "C" int kv_gpu_decode_spans(Workspace *w, Shape s, const Frame *frames, const HostSpan *chunks,
                                   int nchunks, int length, U crc, const DeviceSpan *spans, int count) {
    Fence fence{w->restore};
    if (!prepare(w, s, frames, w->restore) || length > s.wire || length < 0)
        return -1;
    size_t pos = 0;
    for (int i = 0; i < nchunks; ++i) {
        if (chunks[i].length > size_t(length) - pos ||
            cudaMemcpyAsync(w->wire + pos, chunks[i].data, chunks[i].length, cudaMemcpyHostToDevice,
                            w->restore))
            return -1;
        pos += chunks[i].length;
    }
    if (pos != size_t(length))
        return -1;
    if (cudaEventRecord(w->start, w->restore))
        return -1;
    s.wire = length;
    decode_frames(*w, s, s.nf, w->restore);
    check_bytes<<<min(256, (s.raw + 4095) / 4096), 256, 0, w->restore>>>(w->decoded, nullptr, s.raw,
                                                                         w->status + 1, w->status);
    if (cudaEventRecord(w->end, w->restore))
        return -1;
    U status[2];
    if (cudaGetLastError() || cudaMemcpyAsync(status, w->status, 8, cudaMemcpyDeviceToHost, w->restore) ||
        cudaStreamSynchronize(w->restore))
        return -1;
    if (status[0] || status[1] != crc)
        return -2;
    for (int i = 0; i < count; ++i) {
        const auto &p = spans[i];
        if (p.offset < 0 || p.length < 0 || p.offset > s.raw - p.length ||
            cudaMemcpyAsync(reinterpret_cast<void *>(p.device), w->decoded + p.offset, p.length,
                            cudaMemcpyDeviceToDevice, w->restore))
            return -1;
    }
    return cudaStreamSynchronize(w->restore) ? -1 : 0;
}

extern "C" int kv_gpu_decode(Workspace *w, Shape s, const Frame *frames, const B *packet, int length, U crc,
                             const DeviceSpan *spans, int count) {
    const HostSpan part{packet, size_t(length)};
    return kv_gpu_decode_spans(w, s, frames, &part, 1, length, crc, spans, count);
}
