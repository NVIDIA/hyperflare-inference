// SPDX-License-Identifier: Apache-2.0
#include "core.h"
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_scan.cuh>
#include <algorithm>
#include <memory>
#include <new>
struct Shape {
    int nf, raw, work, wire;
};
// Swizzle the decoder tile so a pivot column spans shared-memory banks.
__device__ __forceinline__ int matrix_index(int i, int j) {
    return i * 128 + (j ^ (i & 31));
}

__device__ __forceinline__ U matrix_word(const U *input, U &cell, int i, int j) {
    U word = restore(input[i * 128 + j], cell);
    cell = word;
    return bounded(word);
}

__device__ __forceinline__ void matrix_local(const U *input, U *output) {
    int t = threadIdx.x;
    extern __shared__ U p[];
    __shared__ U v[128], u[128];
    __shared__ int active[128], pivot;

    for (int i = t; i < 16384; i += blockDim.x)
        p[i] = 0;
    if (t < 128)
        active[t] = 1;

    __syncthreads();
    for (int k = 0; k < 64; k++) {
        if (t < 128 && active[t]) {
            U &cell = p[matrix_index(k, t)];
            U pred = cell, w = matrix_word(input, cell, k, t);
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
            U &cell = p[matrix_index(t, c)];
            U pred = cell, w = matrix_word(input, cell, t, c);
            u[t] = divide(minus_word(w, pred), v[c]);
        }
        __syncthreads();
        for (int i = k + 1 + t / 128; i < 128; i += blockDim.x / 128) {
            int j = t % 128;
            if (active[j])
                p[matrix_index(i, j)] = plus(p[matrix_index(i, j)], times(u[i], v[j]));
        }
        __syncthreads();
    }
    for (int i = 64 + t / 128; i < 128; i += blockDim.x / 128) {
        int j = t % 128;
        if (active[j])
            matrix_word(input, p[matrix_index(i, j)], i, j);
    }
    __syncthreads();
    for (int i = t; i < 16384; i += blockDim.x)
        output[i] = p[matrix_index(i % 128, i / 128)];
}

// The rANS state runs backwards through the lane; matrix low bits run forwards.
template <bool Matrix, bool Channels>
__device__ __forceinline__ bool decode_lane(B *out, const B *src, int bytes, Frame f, int lane, int nl,
                                             const B *symbols, const int *model, const B *classes) {
    if (bytes < 4)
        return false;
    int end = bytes - 4, bit = 0;
    U state = load32(src + end);
    if (state < L)
        return false;
    int rows = channel_rows(f.kind), shift = rows ? lg(rows) - 1 : 0;
    for (int i = lane; i < f.n;) {
        // Four symbols consume at most eight bytes. Stage their refill bytes
        // before the dependent table lookups.
        uint64_t tail;
        if (end >= 8) {
            uintptr_t address = reinterpret_cast<uintptr_t>(src + end - 8);
            const U *words = reinterpret_cast<const U *>(address & ~uintptr_t(3));
            int shift = 8 * (address % 4);
            U low = __funnelshift_r(words[0], words[1], shift);
            U high = __funnelshift_r(words[1], words[2], shift);
            tail = (uint64_t(high) << 32) | low;
        } else {
            int hi = min(end, 4), lo = end - hi;
            U high = hi ? load32(src + end - hi) << (8 * (4 - hi)) : 0;
            U low = lo ? load32(src) << (8 * (4 - lo)) : 0;
            tail = (uint64_t(high) << 32) | low;
        }
        #pragma unroll
        for (int step = 0; step < 4 && i < f.n; ++step, i += nl) {
            int c = Channels ? classes[i >> shift] : 0;
            int slot = (c % 4) * 4096 + (state & 4095);
            int symbol = symbols[slot];
            U entry = model[(c % 4) * 256 + symbol];
            state = (entry >> 12) * (state >> 12) + (state & 4095) - (entry & 4095);
            // A normalized state is at least L. Even frequency one needs at most
            // two bytes after the 12-bit table lookup.
            int refill = int(state < L) + int(state < (L >> 8));
            if (end < refill)
                return false;
            end -= refill;
            state = (state << (8 * refill)) | (U(tail >> 48) >> (16 - 8 * refill));
            tail <<= 8 * refill;
            if constexpr (Matrix) {
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
    }
    return state == L && end == (bit + 7) / 8;
}

enum class EntropyKind { Matrix, Bytes, Channels };

template <bool Warp> __device__ __forceinline__ void sync_entropy() {
    if constexpr (Warp)
        __syncwarp();
    else
        __syncthreads();
}

template <EntropyKind Kind, int Threads>
__device__ __forceinline__ void unentropy_one(Frame f, int id, Shape s, const B *wire, B *mid, int *err) {
    constexpr bool mat = Kind == EntropyKind::Matrix, multi = Kind == EntropyKind::Channels;
    constexpr int Groups = mat ? 4 : 1, Entries = mat ? 33 : multi ? 1024 : 256;
    int group = threadIdx.x / Threads, l = threadIdx.x % Threads;
    Entry h = reinterpret_cast<const Entry *>(wire + 16)[id];
    if (h.len < 4 || h.pos > s.wire || h.len > s.wire - h.pos) {
        if (!l)
            atomicAdd(err, 1);
        return;
    }
    const B *in = wire + h.pos;
    B *t = mid + f.work;
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
        if constexpr (!mat)
            for (int i = l; i < f.n; i += Threads)
                t[i] = in[2];
        return;
    }
    int nc = multi ? f.n / channel_rows(f.kind) : 0, models = multi ? 4 : 1;
    __shared__ int freqs[Groups][Entries], sums[Groups][Entries], starts[Groups][mat ? 33 : 129];
    int *freq = freqs[group], *cum = sums[group], *offsets = starts[group];
    __shared__ B cl[512];
    if (in[0] != 2 || h.len < 4 + nc + 2 * nl) {
        if (!l)
            atomicAdd(err, 1);
        return;
    }
    extern __shared__ B symbols[];
    B *table = symbols + group * 4096;
    for (int c = l; c < nc; c += Threads)
        cl[c] = in[4 + c];
    __shared__ int directories[Groups], masks[Groups];
    int &directory = directories[group], &nonempty = masks[group];
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
    sync_entropy<mat>();
    int pos = directory, payload = pos + 2 * nl;
    if (pos < 0 || payload > h.len) {
        if (!l)
            atomicAdd(err, 1);
        return;
    }
    using Scan = std::conditional_t<mat, cub::WarpScan<int>, cub::BlockScan<int, Threads>>;
    __shared__ typename Scan::TempStorage scan[Groups];
    int length = l < nl ? in[pos + 2 * l] + 256 * in[pos + 2 * l + 1] : 0;
    int prefix, total;
    Scan(scan[group]).ExclusiveSum(length, prefix, total);
    if (l < nl)
        offsets[l] = payload + prefix;
    if (!l)
        offsets[nl] = payload + total;
    bool valid = payload + total == h.len && (models > 1 || nonempty);
    for (int c = l; c < nc; c += Threads)
        valid = valid && cl[c] < 8 && (nonempty & (1 << (cl[c] % 4)));
    if (!(mat ? __all_sync(~0u, valid) : __syncthreads_and(valid))) {
        if (!l)
            atomicAdd(err, 1);
        return;
    }
    for (int j = 0; j < models; ++j) {
        for (int sym = l / 32; sym < m; sym += Threads / 32) {
            int base = cum[j * 256 + sym], f = freq[j * 256 + sym];
            for (int i = l % 32; i < f; i += 32)
                table[j * 4096 + base + i] = sym;
        }
    }
    sync_entropy<mat>();
    for (int i = l; i < Entries; i += Threads)
        freq[i] = (freq[i] << 12) | cum[i];
    sync_entropy<mat>();
    if (l < nl && !decode_lane<mat, multi>(t, in + offsets[l], length, f, l, nl, table, freq, cl))
        atomicAdd(err, 1);
}
struct Workspace {
    Frame *frames{};
    B *decoded{}, *mid{}, *wire{};
    U *sizes{}, *status{};
    void *allocation{};
    cudaStream_t restore{}, upload{};
    cudaEvent_t start{}, end{}, uploaded{};
    bool has_matrix{};
    int max_raw{}, max_frames{}, max_work{}, max_wire{};
};

struct DeviceSpan {
    uint64_t device;
    int offset, length;
};

HD size_t word_stride(int bytes) {
    return (size_t(bytes) + 3) & ~size_t(3);
}

// Adjacent pages share a frame plan. Each has its own packet length and CRC status.
HD void advance_units(Workspace &w, const Shape &s, int units) {
    w.decoded += units * word_stride(s.raw);
    w.wire += size_t(units) * s.wire;
    w.status += 2 * units;
    w.sizes += units;
}

__device__ __forceinline__ void select_unit(Workspace &w, Shape &s) {
    if (gridDim.y > 1) {
        w.mid += blockIdx.y * word_stride(s.work);
        advance_units(w, s, blockIdx.y);
        s.wire = *w.sizes;
    }
}

template <EntropyKind Kind, int Threads> __global__ void decode_entropy(Workspace w, Shape s) {
    select_unit(w, s);
    constexpr int groups = Kind == EntropyKind::Matrix ? 4 : 1;
    for (int id = blockIdx.x * groups + threadIdx.x / Threads; id < s.nf; id += gridDim.x * groups) {
        Frame f = w.frames[id];
        auto kind = f.kind == 1            ? EntropyKind::Matrix
                    : channel_rows(f.kind) ? EntropyKind::Channels
                                           : EntropyKind::Bytes;
        if (kind != Kind)
            continue;
        unentropy_one<Kind, Threads>(f, id, s, w.wire, w.mid, reinterpret_cast<int *>(w.status));
        sync_entropy<Kind == EntropyKind::Matrix>();
    }
}

// Channel frames store entire columns consecutively. Transpose a tile in
// shared memory so both the packed-column read and the native-row write coalesce.
__device__ void invert_channels(Frame f, const B *mid, B *decoded, int rows) {
    __shared__ B tile[32][33];
    int x = threadIdx.x % 32, y = threadIdx.x / 32;
    int cols = f.n / rows, stride = f.kind == 40 || f.kind == 45 ? 528 : cols;
    B *out = decoded + f.base + f.sub;
    for (int col = 0; col < cols; col += 32) {
        for (int row = 0; row < rows; row += 32) {
            #pragma unroll
            for (int j = 0; j < 32; j += 8)
                tile[y + j][x] = mid[(col + y + j) * rows + row + x];
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < 32; j += 8)
                out[(row + y + j) * stride + col + x] = tile[x][y + j];
            __syncthreads();
        }
    }
}

template <bool Matrix> __global__ void invert_frames(Workspace w, Shape s) {
    select_unit(w, s);
    for (int id = blockIdx.x; id < s.nf; id += gridDim.x) {
        Frame f = w.frames[id];
        if ((f.kind == 1) != Matrix)
            continue;
        const B *mid = w.mid + f.work;
        if constexpr (Matrix)
            matrix_local(reinterpret_cast<const U *>(mid),
                               reinterpret_cast<U *>(w.decoded + f.base + f.sub));
        else if (int rows = channel_rows(f.kind))
            invert_channels(f, mid, w.decoded, rows);
        else
            for (int i = threadIdx.x; i < f.n; i += blockDim.x)
                w.decoded[address(f, i)] = mid[i];
        if constexpr (Matrix)
            __syncthreads();
    }
}

static void decode_frames(Workspace w, Shape s, dim3 blocks, cudaStream_t stream) {
    // A warp owns all 32 entropy lanes of a matrix frame.
    dim3 matrices(min(blocks.x, (s.nf + 3u) / 4), blocks.y);
    if (w.has_matrix)
        decode_entropy<EntropyKind::Matrix, 32><<<matrices, 128, 16384, stream>>>(w, s);
    decode_entropy<EntropyKind::Bytes, 256><<<blocks, 256, 4096, stream>>>(w, s);
    decode_entropy<EntropyKind::Channels, 256><<<blocks, 256, 16384, stream>>>(w, s);
    if (w.has_matrix)
        invert_frames<true><<<blocks, 1024, 65536, stream>>>(w, s);
    invert_frames<false><<<blocks, 256, 0, stream>>>(w, s);
}

// The CRC of two concatenated intervals is shift(crc(left), len(right)) XOR crc(right).
struct CrcTables {
    U byte[4][256]{}, shift[32][32]{};
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
            byte[0][i] = v;
        }
        for (int j = 1; j < 4; ++j)
            for (int i = 0; i < 256; ++i) {
                U v = byte[j - 1][i];
                byte[j][i] = (v >> 8) ^ byte[0][v & 255];
            }
        for (int i = 0; i < 32; ++i) {
            U v = 1u << i;
            shift[0][i] = (v >> 8) ^ byte[0][v & 255];
        }
        for (int j = 1; j < 32; ++j)
            for (int i = 0; i < 32; ++i)
                shift[j][i] = apply(shift[j - 1], shift[j - 1][i]);
    }
};
__device__ __constant__ CrcTables crc_tables{};

__global__ void check_bytes(const B *source, int length, U *crc,
                            const U *lengths = nullptr) {
    source += size_t(blockIdx.y) * word_stride(length);
    if (lengths)
        length = lengths[blockIdx.y];
    if (blockIdx.x * blockDim.x * 256 >= length)
        return;
    crc += 2 * blockIdx.y;
    __shared__ U table[4][256], shift[32][32];
    for (int k = threadIdx.x; k < 1024; k += blockDim.x) {
        table[k / 256][k % 256] = crc_tables.byte[k / 256][k % 256];
        shift[k / 32][k % 32] = crc_tables.shift[k / 32][k % 32];
    }
    __syncthreads();
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 256; i < length;
         i += gridDim.x * blockDim.x * 256) {
        int end = min(i + 256, length);
        U sum = ~0u;
        int j = i;
        for (; j + 4 <= end; j += 4) {
            U word = load32(source + j);
            U x = sum ^ word;
            sum = table[3][x & 255] ^ table[2][(x >> 8) & 255] ^
                  table[1][(x >> 16) & 255] ^ table[0][x >> 24];
        }
        for (; j < end; ++j) {
            B v = source[j];
            sum = (sum >> 8) ^ table[0][(sum ^ v) & 255];
        }
        sum = ~sum;
        for (int tail = length - end, j = 0; tail; tail >>= 1, ++j) {
            if (!(tail & 1))
                continue;
            U shifted = 0;
            while (sum) {
                int bit = __ffs(sum) - 1;
                shifted ^= shift[j][bit];
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
    if (w->restore)
        cudaStreamSynchronize(w->restore);
    if (w->upload)
        cudaStreamSynchronize(w->upload);
    if (w->uploaded)
        cudaEventDestroy(w->uploaded);
    if (w->upload)
        cudaStreamDestroy(w->upload);
    if (w->start)
        cudaEventDestroy(w->start);
    if (w->end)
        cudaEventDestroy(w->end);
    if (w->restore)
        cudaStreamDestroy(w->restore);
    if (w->allocation)
        cudaFree(w->allocation);
    delete w;
}

extern "C" Workspace *kv_gpu_open(int raw, int nf, int work, int wire, const char **failure,
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
    w->max_wire = wire;
    int low = 0, high = 0;
    cudaDeviceGetStreamPriorityRange(&low, &high);
    if (failed("create restore stream",
               cudaStreamCreateWithPriority(&w->restore, cudaStreamNonBlocking, high)) ||
        failed("create restore upload stream",
               cudaStreamCreateWithPriority(&w->upload, cudaStreamNonBlocking, high)) ||
        failed("create upload event", cudaEventCreateWithFlags(&w->uploaded, cudaEventDisableTiming)) ||
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
    const size_t f = reserve(sizeof(Frame) * nf), d = reserve(raw), m = reserve(work),
                 p = reserve(wire), z = reserve(4 * nf), v = reserve(8);
    *bytes = pos;
    if (failed("cudaMalloc", cudaMalloc(&w->allocation, pos)) ||
        failed(
            "configure inverse kernel",
            cudaFuncSetAttribute(invert_frames<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536))) {
        return nullptr;
    }
    B *base = static_cast<B *>(w->allocation);
    w->frames = reinterpret_cast<Frame *>(base + f);
    w->decoded = base + d;
    w->mid = base + m;
    w->wire = base + p;
    w->sizes = reinterpret_cast<U *>(base + z);
    w->status = reinterpret_cast<U *>(base + v);
    return owned.release();
}
extern "C" void kv_gpu_close(Workspace *w) {
    close_workspace(w);
}
static uint64_t transform_time_ns(Workspace *w) {
    float ms = 0;
    return cudaEventElapsedTime(&ms, w->start, w->end) == cudaSuccess ? uint64_t(ms * 1000000) : 0;
}

static bool prepare(Workspace *w, Shape s, const Frame *frames, cudaStream_t stream) {
    if (s.nf <= 0 || s.nf > w->max_frames || s.raw <= 0 || s.raw > w->max_raw ||
        s.work > w->max_work || s.wire > w->max_wire)
        return false;
    w->has_matrix = std::any_of(frames, frames + s.nf, [](Frame f) { return f.kind == 1; });
    return cudaMemcpyAsync(w->frames, frames, sizeof(Frame) * s.nf, cudaMemcpyHostToDevice, stream) ==
               cudaSuccess &&
           cudaMemsetAsync(w->status, 0, 8, stream) == cudaSuccess;
}

// Every return, including a failed enqueue, fences work that may still touch a caller's spans.
struct Fence {
    cudaStream_t stream;
    ~Fence() {
        cudaStreamSynchronize(stream);
    }
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

// Batched restores share the single-unit decode buffers.
extern "C" int kv_gpu_decode_capacity(const Workspace *w, Shape s) {
    if (s.raw <= 0 || s.work <= 0 || s.wire <= 0 || s.wire > INT_MAX - 3)
        return 1;
    size_t available = reinterpret_cast<const B *>(w->sizes) - w->decoded;
    size_t slot = 8 + word_stride(max(s.raw, s.wire)), mid = word_stride(s.work);
    // Both groups retain their output. Only the group decoding now needs intermediate storage.
    size_t pair = 2 * slot + mid;
    size_t count = 2 * (available / pair) + (available % pair >= slot + mid);
    return int(std::max(size_t(1), std::min({size_t(64), size_t(w->max_frames), count})));
}

static Workspace restore_view(const Workspace &w, Shape &s, int count) {
    Workspace view = w;
    if (count > 1) {
        s.wire = int(word_stride(max(s.raw, s.wire)));
        view.status = reinterpret_cast<U *>(w.decoded);
        view.wire = w.decoded + 8 * count;
        view.mid = view.wire + size_t(count) * s.wire;
        // Inversion starts after every entropy kernel in its group. Its output
        // fits in the consumed packet slots, before the next upload.
        view.decoded = view.wire;
    }
    return view;
}

struct PlaneSpans {
    DeviceSpan spans[64];
};

__global__ void scatter_plane(const B *decoded, size_t stride, PlaneSpans plane) {
    DeviceSpan span = plane.spans[blockIdx.y];
    const B *src = decoded + blockIdx.y * stride + span.offset;
    auto *dst = reinterpret_cast<B *>(span.device);
    int first = blockIdx.x * blockDim.x + threadIdx.x, step = gridDim.x * blockDim.x;
    int words = (uintptr_t(src) | uintptr_t(dst)) % 16 ? 0 : span.length / 16;
    for (int i = first; i < words; i += step)
        reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
    for (int i = 16 * words + first; i < span.length; i += step)
        dst[i] = src[i];
}

static int decode_uploaded(Workspace view, Shape s, const RestoreUnit *units, int count) {
    U status[128];
    // Validate the staged packet bytes before launching any entropy work.
    check_bytes<<<dim3(min(256, 1 + (s.wire - 1) / 65536), count), 256, 0, view.restore>>>(
        view.wire, s.wire, view.status + 1, count > 1 ? view.sizes : nullptr);
    if (cudaGetLastError() ||
        cudaMemcpyAsync(status, view.status, 8 * count, cudaMemcpyDeviceToHost, view.restore) ||
        cudaStreamSynchronize(view.restore))
        return -1;
    for (int j = 0; j < count; ++j)
        if (status[2 * j + 1] != units[j].stored)
            return -3;
    if (cudaMemsetAsync(view.status, 0, 8 * count, view.restore))
        return -1;
    if (cudaEventRecord(view.start, view.restore))
        return -1;
    decode_frames(view, s, dim3(s.nf, count), view.restore);
    check_bytes<<<dim3(min(256, 1 + (s.raw - 1) / 65536), count), 256, 0, view.restore>>>(
        view.decoded, s.raw, view.status + 1);
    if (cudaEventRecord(view.end, view.restore) || cudaGetLastError() ||
        cudaMemcpyAsync(status, view.status, 8 * count, cudaMemcpyDeviceToHost, view.restore) ||
        cudaStreamSynchronize(view.restore))
        return -1;
    for (int j = 0; j < count; ++j)
        if (status[2 * j] || status[2 * j + 1] != units[j].original)
            return -2;
    return 0;
}

extern "C" int kv_gpu_decode_spans(Workspace *w, Shape s, const Frame *frames, const RestoreUnit *units,
                                   int count, uint64_t *transform_ns = nullptr) {
    U lengths[64];
    Fence restore{w->restore}, upload{w->upload};
    if (count <= 0 || !prepare(w, s, frames, w->restore) || count > kv_gpu_decode_capacity(w, s))
        return -1;
    for (int j = 0; j < count; ++j) {
        if (units[j].length <= 0 || units[j].length > s.wire)
            return -1;
        lengths[j] = units[j].length;
    }
    Workspace view = restore_view(*w, s, count);
    if (count > 1 && (cudaMemsetAsync(view.status, 0, 8 * count, w->restore) ||
                      cudaMemcpyAsync(view.sizes, lengths, 4 * count, cudaMemcpyHostToDevice, w->restore)))
        return -1;

    // Keep the copy engine one group ahead. Each wait captures the current
    // event before the upload stream records it again for the next group.
    int width = (count + 1) / 2;
    auto stage = [&](int first) {
        for (int j = first; j < min(first + width, count); ++j) {
            const auto &u = units[j];
            size_t pos = 0;
            for (int i = 0; i < u.nchunks; ++i) {
                const auto &part = u.chunks[i];
                if (part.length > size_t(u.length) - pos ||
                    cudaMemcpyAsync(view.wire + size_t(j) * s.wire + pos, part.data, part.length,
                                    cudaMemcpyHostToDevice, w->upload))
                    return false;
                pos += part.length;
            }
            if (pos != size_t(u.length))
                return false;
        }
        return cudaEventRecord(w->uploaded, w->upload) == cudaSuccess;
    };
    if (!stage(0))
        return -1;
    uint64_t ns = 0;
    for (int first = 0; first < count; first += width) {
        int n = min(width, count - first);
        if (cudaStreamWaitEvent(w->restore, w->uploaded) || (first + n < count && !stage(first + n)))
            return -1;
        Workspace group = view;
        advance_units(group, s, first);
        Shape shape = s;
        if (n == 1)
            shape.wire = lengths[first];
        if (int rc = decode_uploaded(group, shape, units + first, n))
            return rc;
        ns += transform_time_ns(w);
    }
    // Page IDs in a restore window are distinct. Copy each plane across
    // those pages together, preserving the order of spans within each unit.
    for (int i = 0;; ++i) {
        PlaneSpans plane{};
        int spans = 0, largest = 0;
        for (int j = 0; j < count; ++j) {
            if (i >= units[j].count)
                continue;
            const auto &p = units[j].spans[i];
            if (p.offset < 0 || p.length < 0 || p.offset > s.raw - p.length)
                return -1;
            plane.spans[j] = p;
            largest = max(largest, p.length);
            ++spans;
        }
        if (!spans)
            break;
        dim3 grid(min(256, max(1, (largest + 2047) / 2048)), count);
        scatter_plane<<<grid, 128, 0, w->restore>>>(view.decoded, word_stride(s.raw), plane);
        if (cudaGetLastError())
            return -1;
    }
    if (transform_ns)
        *transform_ns = ns;
    return cudaStreamSynchronize(w->restore) ? -1 : 0;
}
