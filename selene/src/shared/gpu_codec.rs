// SPDX-License-Identifier: Apache-2.0

//! GPU-edge lossless checkpoint codec (`GKV1/v1/64-pivots-nvcomp-ans`) for VRAM <-> DRAM transfers.
//!
//! Zero compile-time CUDA or `nvcc` dependency: loads `libcuda.so.1`, `libnvcomp.so.6`, and the
//! core-embedded ELF cubin at runtime via `dlopen` and `cuModuleLoadData` (honoring `CUDA_DISABLE_JIT=1`).
//!
//! Runs on the caller's lowest-priority CUDA stream with a fixed pre-allocated per-rank device
//! workspace (zero steady-state GPU allocation, zero static Green Context SM reservation):
//! - Encode (`encode_from_device`): gather GPU cache plane spans -> 256-thread swizzled 64-pivot
//!   float32 matrix predictor + 4-lane pack + coalesced per-lane shuffle/page-split ->
//!   `nvcompBatchedANSCompressAsync` (64 KiB homogeneous chunks) -> GPU `nvcompBatchedCRC32Async` ->
//!   D2H compact `GKV1` packet (or `Literal` raw fallback when incompressible).
//! - Decode (`decode_to_device`): H2D packet -> stored CRC32 check ->
//!   `nvcompBatchedANSDecompressAsync` -> bounds-checked 4-lane unpack + inverse 64-pivot predictor
//!   + inverse shuffle/page-split -> GPU original CRC32 check -> scatter into GPU cache planes.

use super::codec::{Checksum, Encoding, Plan};
use super::packed::crc32_chunks;
use super::wire::PackedUnit;
use super::wire::packed_unit::Representation;
use std::io::{Error, ErrorKind, Result};
#[cfg(target_os = "linux")]
use std::path::PathBuf;

pub const MAGIC: u32 = u32::from_le_bytes(*b"GKV1");
pub const VERSION: u32 = 1;
const HEADER_BYTES: usize = 24;
const RAW_FLAG: u32 = 1 << 31;
const ANS_CHUNK: usize = 65536;
const MATRIX_BYTES: usize = 65536;
const MATRIX_ELEMS: usize = 16384;
const MAX_LANE_U32S: usize = 3968;
const LANE_STRIDE: usize = MAX_LANE_U32S + 2;

/// One contiguous GPU device memory span belonging to a rank's page or row share.
#[derive(Clone, Copy, Debug)]
pub struct DeviceSpan {
    pub device: u64,
    pub offset: usize,
    pub length: usize,
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

#[derive(Clone, Copy, Debug)]
struct CopyOp {
    /// Offset in the transformed `d_trans` staging buffer.
    trans_offset: usize,
    /// Offset in the coalesced `d_ans_in` batch buffer.
    ans_offset: usize,
    length: usize,
}

#[derive(Clone, Copy, Debug)]
struct TransformOp {
    offset: usize,
    length: usize,
    encoding: Encoding,
}

/// Static GPU transform & coalesced ANS chunk plan derived from a validated [`Plan`].
#[derive(Clone, Debug)]
pub struct GpuPlan {
    original_bytes: usize,
    num_mats: usize,
    matrix_fields: Vec<(usize, usize)>, // (offset, num_mats_in_field)
    transforms: Vec<TransformOp>,
    coalesce: Vec<CopyOp>,
    chunk_offsets: Vec<usize>,
    chunk_sizes: Vec<usize>,
    ans_total_bytes: usize,
}

impl GpuPlan {
    pub fn from_plan(plan: &Plan) -> Result<Self> {
        let original_bytes = plan.original_bytes();
        let fields = plan.fields();

        // Build complete non-overlapping interval list (including untyped padding as Bytes).
        let mut intervals = Vec::new();
        let mut end = 0usize;
        for &f in fields {
            if f.offset > end {
                intervals.push(TransformOp {
                    offset: end,
                    length: f.offset - end,
                    encoding: Encoding::Bytes,
                });
            }
            intervals.push(TransformOp {
                offset: f.offset,
                length: f.length,
                encoding: f.encoding,
            });
            end = f.offset + f.length;
        }
        if end < original_bytes {
            intervals.push(TransformOp {
                offset: end,
                length: original_bytes - end,
                encoding: Encoding::Bytes,
            });
        }

        // Partition intervals into matrix fields and homogeneous lane pools so every ANS chunk
        // sees a single byte-lane distribution.
        let mut matrix_fields = Vec::new();
        let mut num_mats = 0usize;
        let mut transforms = Vec::new();
        // Homogeneous pools keyed by (group_tag, lane_index) -> Vec<(trans_offset, len)>
        let mut pools: std::collections::BTreeMap<(u32, u32), Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();

        for op in intervals {
            match op.encoding {
                Encoding::Float32Matrix128 => {
                    let count = op.length / MATRIX_BYTES;
                    matrix_fields.push((op.offset, count));
                    num_mats += count;
                }
                Encoding::Bytes => {
                    transforms.push(op);
                    pools.entry((0, 0)).or_default().push((op.offset, op.length));
                }
                Encoding::Bfloat16 | Encoding::Convolution20 | Encoding::Convolution32 => {
                    // Conv20 (10x bf16) and Conv32 (16x bf16) share BF16 2-byte lane structure.
                    transforms.push(TransformOp {
                        offset: op.offset,
                        length: op.length,
                        encoding: Encoding::Bfloat16,
                    });
                    let half = op.length / 2;
                    let tag = match op.encoding {
                        Encoding::Bfloat16 => 2,
                        Encoding::Convolution20 => 20,
                        _ => 32,
                    };
                    pools.entry((tag, 0)).or_default().push((op.offset, half));
                    pools
                        .entry((tag, 1))
                        .or_default()
                        .push((op.offset + half, half));
                }
                Encoding::Latent528
                | Encoding::Index132
                | Encoding::Fp8Page512
                | Encoding::WindowFp4
                | Encoding::LatentFp4 => {
                    transforms.push(op);
                    let (tag, block, rows, cols, scale_w) = page_params(op.encoding);
                    let nval = rows * cols;
                    let ns = block - nval;
                    let nblocks = op.length / block;
                    for b in 0..nblocks {
                        let base = op.offset + b * block;
                        pools.entry((tag, 0)).or_default().push((base, nval));
                        if ns > 0 {
                            if scale_w > 1 {
                                let lane_len = ns / scale_w as usize;
                                for l in 0..scale_w {
                                    pools.entry((tag, 1 + l)).or_default().push((
                                        base + nval + l as usize * lane_len,
                                        lane_len,
                                    ));
                                }
                            } else {
                                pools.entry((tag, 1)).or_default().push((base + nval, ns));
                            }
                        }
                    }
                }
            }
        }

        // Assign contiguous ranges inside `d_ans_in`:
        // 1. Matrix widths come first (at offset 0..num_mats * MATRIX_ELEMS).
        // 2. Each homogeneous non-matrix lane pool follows contiguously.
        let mut chunk_offsets = Vec::new();
        let mut chunk_sizes = Vec::new();
        let mut ans_pos = 0usize;

        let push_chunked = |start: usize,
                            len: usize,
                            offs: &mut Vec<usize>,
                            szs: &mut Vec<usize>| {
            let mut p = 0;
            while p < len {
                let n = (len - p).min(ANS_CHUNK);
                offs.push(start + p);
                szs.push(n);
                p += n;
            }
        };

        let widths_bytes = num_mats * MATRIX_ELEMS;
        if widths_bytes > 0 {
            push_chunked(0, widths_bytes, &mut chunk_offsets, &mut chunk_sizes);
            ans_pos = widths_bytes;
        }

        let mut coalesce = Vec::new();
        for spans in pools.values() {
            let pool_start = ans_pos;
            for &(trans_offset, length) in spans {
                coalesce.push(CopyOp {
                    trans_offset,
                    ans_offset: ans_pos,
                    length,
                });
                ans_pos += length;
            }
            push_chunked(
                pool_start,
                ans_pos - pool_start,
                &mut chunk_offsets,
                &mut chunk_sizes,
            );
        }

        Ok(Self {
            original_bytes,
            num_mats,
            matrix_fields,
            transforms,
            coalesce,
            chunk_offsets,
            chunk_sizes,
            ans_total_bytes: ans_pos,
        })
    }

    pub fn original_bytes(&self) -> usize {
        self.original_bytes
    }

    pub fn num_chunks(&self) -> usize {
        self.chunk_sizes.len()
    }

    pub fn num_mats(&self) -> usize {
        self.num_mats
    }

    pub fn metadata_bytes(&self) -> usize {
        HEADER_BYTES + self.chunk_sizes.len() * 4 + self.num_mats * 8
    }

    pub fn max_packet(&self) -> usize {
        self.metadata_bytes() + self.original_bytes
    }

    pub fn ans_total_bytes(&self) -> usize {
        self.ans_total_bytes
    }

    /// Estimate fixed device workspace bytes needed for one plan.
    pub fn workspace_bytes_estimate(&self) -> u64 {
        Self::workspace_bytes_estimate_pair(self, self)
    }

    /// Estimate fixed device workspace bytes needed for a `(page, row)` plan pair using
    /// per-buffer maxima matching `GpuRankCodec::new` (strictly >= allocated_device_bytes
    /// plus nvCOMP/cubin runtime module headroom).
    pub fn workspace_bytes_estimate_pair(page: &Self, row: &Self) -> u64 {
        let max_orig = (page.original_bytes.max(row.original_bytes)) as u64;
        let max_pkt = (page.max_packet().max(row.max_packet()).max(max_orig as usize)) as u64;
        let max_mats = (page.num_mats.max(row.num_mats).max(1)) as u64;
        let max_chunks = (page.num_chunks().max(row.num_chunks()).max(1)) as u64;
        let max_ans_in = (page.ans_total_bytes.max(row.ans_total_bytes).max(64)) as u64;
        let mat_z = max_mats * (MATRIX_BYTES as u64);
        let mats_low = max_mats * 4 * (LANE_STRIDE as u64) * 4;
        let ans = max_ans_in * 3 + max_chunks * 8192;
        // d_raw + d_trans + d_pkt + d_mat_z + d_low_pad + d_ans_in/out/temp + nvcomp/cubin headroom
        max_orig * 2 + max_pkt + mat_z + mats_low + ans + (64 << 20)
    }
}

fn page_params(enc: Encoding) -> (u32, usize, usize, usize, u32) {
    match enc {
        Encoding::Latent528 => (40, 135_168, 256, 512, 4),
        Encoding::Index132 => (41, 8_448, 64, 128, 4),
        Encoding::Fp8Page512 => (42, 32_768, 64, 512, 1),
        Encoding::WindowFp4 => (45, 33_792, 64, 512, 1),
        Encoding::LatentFp4 => (47, 18_432, 64, 256, 1),
        _ => unreachable!(),
    }
}

fn page_stride_planar(enc: Encoding) -> (usize, i32) {
    match enc {
        Encoding::Latent528 => (528, 0),
        Encoding::Index132 => (128, 1),
        Encoding::Fp8Page512 => (512, 0),
        Encoding::WindowFp4 => (528, 0),
        Encoding::LatentFp4 => (288, 0),
        _ => unreachable!(),
    }
}

// =========================================================================
// Linux CUDA Driver + nvCOMP 6.0 dynamic runtime loader & GPU engine
// =========================================================================
#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
    use std::sync::OnceLock;

    type CUresult = c_int;
    type CUdeviceptr = u64;
    type CUmodule = *mut c_void;
    type CUfunction = *mut c_void;
    type CUstream = *mut c_void;
    const CUDA_SUCCESS: CUresult = 0;
    const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

    type NvcompStatus = c_int;
    const NVCOMP_SUCCESS: NvcompStatus = 0;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct AnsCompressOpts {
        ans_type: i32,
        data_type: i32,
        max_sub_chunk_count: u8,
        states_per_lane: u8,
        histogram_reduction_log2: u8,
        reserved: [u8; 53],
    }

    impl Default for AnsCompressOpts {
        fn default() -> Self {
            Self {
                ans_type: 0,  // nvcomp_rANS
                data_type: 0, // NVCOMP_TYPE_CHAR
                max_sub_chunk_count: 0,
                states_per_lane: 0,
                histogram_reduction_log2: 0,
                reserved: [0; 53],
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct AnsDecompressOpts {
        backend: i32,
        data_type: i32,
        max_sub_chunk_count: u8,
        states_per_lane: u8,
        skip_validate: u8,
        reserved: [u8; 53],
    }

    impl Default for AnsDecompressOpts {
        fn default() -> Self {
            Self {
                backend: 0,      // NVCOMP_DECOMPRESS_BACKEND_DEFAULT
                data_type: 0xff, // NVCOMP_TYPE_BITS
                max_sub_chunk_count: 0,
                states_per_lane: 0,
                skip_validate: 0,
                reserved: [0; 53],
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Crc32Spec {
        poly: u32,
        init: u32,
        ref_in: u8,
        ref_out: u8,
        pad: [u8; 2],
        xorout: u32,
        reserved: [u8; 16],
    }

    impl Crc32Spec {
        fn ieee() -> Self {
            Self {
                poly: 0x04C1_1DB7,
                init: 0xFFFF_FFFF,
                ref_in: 1,
                ref_out: 1,
                pad: [0; 2],
                xorout: 0xFFFF_FFFF,
                reserved: [0; 16],
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Crc32KernelConf {
        kernel_kind: i32,
        bytes_per_read: i32,
        blocks_per_msg: i32,
        reserved: [u8; 20],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Crc32Opts {
        spec: Crc32Spec,
        kernel_conf: Crc32KernelConf,
        reserved: [u8; 64],
    }

    struct Api {
        _cuda_lib: *mut c_void,
        _nvcomp_lib: *mut c_void,
        module_load_data: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
        module_unload: unsafe extern "C" fn(CUmodule) -> CUresult,
        module_get_function:
            unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
        func_set_attribute: unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult,
        launch_kernel: unsafe extern "C" fn(
            CUfunction,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            CUstream,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> CUresult,
        mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
        mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
        mem_host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
        mem_free_host: unsafe extern "C" fn(*mut c_void) -> CUresult,
        memset_d8_async: unsafe extern "C" fn(CUdeviceptr, u8, usize, CUstream) -> CUresult,
        memcpy_htod_async:
            unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
        memcpy_dtoh_async:
            unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult,
        memcpy_dtod_async:
            unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult,
        stream_synchronize: unsafe extern "C" fn(CUstream) -> CUresult,
        ans_max_out: unsafe extern "C" fn(usize, AnsCompressOpts, *mut usize) -> NvcompStatus,
        ans_comp_temp: unsafe extern "C" fn(
            usize,
            usize,
            AnsCompressOpts,
            *mut usize,
            usize,
            CUstream,
        ) -> NvcompStatus,
        ans_comp_async: unsafe extern "C" fn(
            *const *const c_void,
            *const usize,
            usize,
            usize,
            *mut c_void,
            usize,
            *const *mut c_void,
            *mut usize,
            AnsCompressOpts,
            *mut NvcompStatus,
            CUstream,
        ) -> NvcompStatus,
        ans_decomp_temp: unsafe extern "C" fn(
            usize,
            usize,
            AnsDecompressOpts,
            *mut usize,
            usize,
            CUstream,
        ) -> NvcompStatus,
        ans_decomp_async: unsafe extern "C" fn(
            *const *const c_void,
            *const usize,
            *const usize,
            *mut usize,
            usize,
            *mut c_void,
            usize,
            *const *mut c_void,
            AnsDecompressOpts,
            *mut NvcompStatus,
            CUstream,
        ) -> NvcompStatus,
        crc32_conf: unsafe extern "C" fn(
            *const usize,
            usize,
            *mut Crc32KernelConf,
            usize,
            CUstream,
        ) -> NvcompStatus,
        crc32_async: unsafe extern "C" fn(
            *const *const c_void,
            *const usize,
            usize,
            *mut u32,
            Crc32Opts,
            i32,
            *mut NvcompStatus,
            CUstream,
        ) -> NvcompStatus,
    }

    unsafe impl Send for Api {}
    unsafe impl Sync for Api {}

    unsafe fn sym<T>(lib: *mut c_void, name: &[u8]) -> Result<T> {
        let ptr = unsafe { libc::dlsym(lib, name.as_ptr() as *const c_char) };
        if ptr.is_null() {
            let label = CStr::from_bytes_with_nul(name)
                .map(|s| s.to_string_lossy())
                .unwrap_or_default();
            return Err(Error::other(format!("missing GPU codec symbol {label}")));
        }
        Ok(unsafe { std::mem::transmute_copy(&ptr) })
    }

    fn open_lib(candidates: &[PathBuf]) -> Result<*mut c_void> {
        let mut last_err = String::new();
        for path in candidates {
            let Ok(cpath) = CString::new(path.to_string_lossy().as_bytes()) else {
                continue;
            };
            let handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                return Ok(handle);
            }
            let err_ptr = unsafe { libc::dlerror() };
            if !err_ptr.is_null() {
                last_err = unsafe { std::ffi::CStr::from_ptr(err_ptr) }
                    .to_string_lossy()
                    .into_owned();
            }
        }
        Err(Error::new(
            ErrorKind::NotFound,
            format!(
                "cannot load required shared library (tried {:?}; last dlerror: {})",
                candidates, last_err
            ),
        ))
    }

    fn nvcomp_candidates() -> Vec<PathBuf> {
        let mut out = vec![PathBuf::from("libnvcomp.so.6")];
        if let Ok(exe) = std::env::current_exe()
            && let Some(dir) = exe.parent()
        {
            out.push(dir.join("libnvcomp.so.6"));
            out.push(dir.join("../lib/libnvcomp.so.6"));
            out.push(dir.join("../lib64/libnvcomp.so.6"));
        }
        out.push(PathBuf::from("/usr/lib/x86_64-linux-gnu/libnvcomp.so.6"));
        out.push(PathBuf::from("/usr/lib/aarch64-linux-gnu/libnvcomp.so.6"));
        out.push(PathBuf::from("/usr/local/lib/libnvcomp.so.6"));
        out
    }

    fn api() -> Result<&'static Api> {
        static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
        API.get_or_init(|| unsafe {
            let cuda = open_lib(&[PathBuf::from("libcuda.so.1"), PathBuf::from("libcuda.so")])
                .map_err(|e| e.to_string())?;
            let nvcomp = open_lib(&nvcomp_candidates()).map_err(|e| e.to_string())?;
            Ok(Api {
                _cuda_lib: cuda,
                _nvcomp_lib: nvcomp,
                module_load_data: sym(cuda, b"cuModuleLoadData\0").map_err(|e| e.to_string())?,
                module_unload: sym(cuda, b"cuModuleUnload\0").map_err(|e| e.to_string())?,
                module_get_function: sym(cuda, b"cuModuleGetFunction\0")
                    .map_err(|e| e.to_string())?,
                func_set_attribute: sym(cuda, b"cuFuncSetAttribute\0")
                    .map_err(|e| e.to_string())?,
                launch_kernel: sym(cuda, b"cuLaunchKernel\0").map_err(|e| e.to_string())?,
                mem_alloc: sym(cuda, b"cuMemAlloc_v2\0").map_err(|e| e.to_string())?,
                mem_free: sym(cuda, b"cuMemFree_v2\0").map_err(|e| e.to_string())?,
                mem_host_alloc: sym(cuda, b"cuMemHostAlloc\0").map_err(|e| e.to_string())?,
                mem_free_host: sym(cuda, b"cuMemFreeHost\0").map_err(|e| e.to_string())?,
                memset_d8_async: sym(cuda, b"cuMemsetD8Async\0").map_err(|e| e.to_string())?,
                memcpy_htod_async: sym(cuda, b"cuMemcpyHtoDAsync_v2\0")
                    .map_err(|e| e.to_string())?,
                memcpy_dtoh_async: sym(cuda, b"cuMemcpyDtoHAsync_v2\0")
                    .map_err(|e| e.to_string())?,
                memcpy_dtod_async: sym(cuda, b"cuMemcpyDtoDAsync_v2\0")
                    .map_err(|e| e.to_string())?,
                stream_synchronize: sym(cuda, b"cuStreamSynchronize\0")
                    .map_err(|e| e.to_string())?,
                ans_max_out: sym(nvcomp, b"nvcompBatchedANSCompressGetMaxOutputChunkSize\0")
                    .map_err(|e| e.to_string())?,
                ans_comp_temp: sym(nvcomp, b"nvcompBatchedANSCompressGetTempSize\0")
                    .map_err(|e| e.to_string())?,
                ans_comp_async: sym(nvcomp, b"nvcompBatchedANSCompressAsync\0")
                    .map_err(|e| e.to_string())?,
                ans_decomp_temp: sym(nvcomp, b"nvcompBatchedANSDecompressGetTempSize\0")
                    .map_err(|e| e.to_string())?,
                ans_decomp_async: sym(nvcomp, b"nvcompBatchedANSDecompressAsync\0")
                    .map_err(|e| e.to_string())?,
                crc32_conf: sym(nvcomp, b"nvcompBatchedCRC32GetHeuristicConf\0")
                    .map_err(|e| e.to_string())?,
                crc32_async: sym(nvcomp, b"nvcompBatchedCRC32Async\0")
                    .map_err(|e| e.to_string())?,
            })
        })
        .as_ref()
        .map_err(|e| Error::other(e.clone()))
    }

    fn cu_ok(r: CUresult, what: &str) -> Result<()> {
        if r == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Error::other(format!("{what} failed with CUDA error {r}")))
        }
    }

    fn nv_ok(r: NvcompStatus, what: &str) -> Result<()> {
        if r == NVCOMP_SUCCESS {
            Ok(())
        } else {
            Err(Error::other(format!("{what} failed with nvCOMP error {r}")))
        }
    }

    struct DevBuf {
        ptr: CUdeviceptr,
        bytes: usize,
    }

    impl DevBuf {
        fn alloc(bytes: usize) -> Result<Self> {
            let api = api()?;
            let size = bytes.max(64);
            let mut ptr = 0u64;
            cu_ok(unsafe { (api.mem_alloc)(&mut ptr, size) }, "cuMemAlloc")?;
            Ok(Self { ptr, bytes: size })
        }
    }

    impl Drop for DevBuf {
        fn drop(&mut self) {
            if self.ptr != 0
                && let Ok(api) = api()
            {
                unsafe {
                    let _ = (api.mem_free)(self.ptr);
                }
            }
        }
    }

    /// Pinned host buffer used for GPU-codec device-to-host and host-to-device metadata transfers.
    struct PinnedBuf {
        ptr: *mut u8,
        bytes: usize,
    }

    impl PinnedBuf {
        fn alloc(bytes: usize) -> Result<Self> {
            let api = api()?;
            let size = bytes.max(64);
            let mut ptr: *mut c_void = std::ptr::null_mut();
            cu_ok(
                unsafe { (api.mem_host_alloc)(&mut ptr, size, 0) },
                "cuMemHostAlloc",
            )?;
            unsafe {
                std::ptr::write_bytes(ptr as *mut u8, 0, size);
            }
            Ok(Self {
                ptr: ptr as *mut u8,
                bytes: size,
            })
        }
        fn as_slice(&self, len: usize) -> &[u8] {
            assert!(len <= self.bytes);
            unsafe { std::slice::from_raw_parts(self.ptr, len) }
        }
        fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
            assert!(len <= self.bytes);
            unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
        }
    }

    impl Drop for PinnedBuf {
        fn drop(&mut self) {
            if !self.ptr.is_null()
                && let Ok(api) = api()
            {
                unsafe {
                    let _ = (api.mem_free_host)(self.ptr as *mut c_void);
                }
            }
        }
    }

    pub struct GpuRankCodec {
        module: CUmodule,
        fn_pred_enc: CUfunction,
        fn_pred_dec: CUfunction,
        fn_pack: CUfunction,
        fn_compact: CUfunction,
        fn_unpack: CUfunction,
        fn_shuffle: CUfunction,
        fn_page_split: CUfunction,
        fn_copy_batch: CUfunction,
        // Reusable fixed per-rank device buffers (zero steady-state allocation).
        d_raw: DevBuf,
        d_trans: DevBuf,
        d_pkt: DevBuf,
        d_mat_z: DevBuf,
        d_mat_ptrs: DevBuf,
        d_low_pad: DevBuf,
        d_lane_u32s: DevBuf,
        d_lane_offs: DevBuf,
        d_err_flag: DevBuf,
        d_ans_in: DevBuf,
        d_ans_out: DevBuf,
        d_ans_temp: DevBuf,
        d_in_ptrs: DevBuf,
        d_out_ptrs: DevBuf,
        d_in_sizes: DevBuf,
        d_out_sizes: DevBuf,
        d_rt_sizes: DevBuf,
        d_statuses: DevBuf,
        d_copy_dst: DevBuf,
        d_copy_src: DevBuf,
        d_copy_len: DevBuf,
        d_crc_ptr: DevBuf,
        d_crc_size: DevBuf,
        d_crc_out: DevBuf,
        d_crc_status: DevBuf,
        h_pinned: PinnedBuf,
        max_out_chunk: usize,
        max_copies: usize,
        ans_comp_temp_bytes: usize,
        ans_decomp_temp_bytes: usize,
        // Reusable host staging vectors to avoid steady-state heap churn.
        h_ptr_buf: Vec<u64>,
        h_ptr2_buf: Vec<u64>,
        h_len_buf: Vec<u64>,
        h_size_buf: Vec<usize>,
        h_lanes_buf: Vec<u16>,
        h_lane_offs_buf: Vec<u32>,
        h_status_buf: Vec<NvcompStatus>,
    }

    unsafe impl Send for GpuRankCodec {}

    impl Drop for GpuRankCodec {
        fn drop(&mut self) {
            if !self.module.is_null()
                && let Ok(api) = api()
            {
                unsafe {
                    let _ = (api.module_unload)(self.module);
                }
            }
        }
    }

    impl GpuRankCodec {
        /// Verify that `libcuda.so.1` and `libnvcomp.so.6` with required symbols can be loaded.
        pub fn check_runtime() -> Result<()> {
            api().map(|_| ())
        }

        /// Load an embedded ELF cubin image and allocate fixed per-rank workspace for
        /// `page_plan` and `row_plan` in the current CUDA context.
        pub fn new(
            page_plan: &GpuPlan,
            row_plan: &GpuPlan,
            max_spans: usize,
            cubin: &[u8],
            stream: usize,
        ) -> Result<Self> {
            let api = api()?;
            if !cubin.starts_with(b"\x7fELF") {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "embedded GPU KV codec image is not a compiled ELF cubin",
                ));
            }

            let mut module: CUmodule = std::ptr::null_mut();
            cu_ok(
                unsafe { (api.module_load_data)(&mut module, cubin.as_ptr() as *const _) },
                "cuModuleLoadData(kv_codec_gpu)",
            )?;

            let get_fn = |name: &[u8]| -> Result<CUfunction> {
                let mut f: CUfunction = std::ptr::null_mut();
                cu_ok(
                    unsafe { (api.module_get_function)(&mut f, module, name.as_ptr() as *const _) },
                    "cuModuleGetFunction",
                )?;
                Ok(f)
            };

            let fn_pred_enc = get_fn(b"kv_predict_f32_128_enc\0")?;
            let fn_pred_dec = get_fn(b"kv_predict_f32_128_dec\0")?;
            let fn_pack = get_fn(b"kv_pack_4lane_gpu\0")?;
            let fn_compact = get_fn(b"kv_compact_low_lanes\0")?;
            let fn_unpack = get_fn(b"kv_unpack_4lane_gpu\0")?;
            let fn_shuffle = get_fn(b"kv_byte_shuffle_gpu\0")?;
            let fn_page_split = get_fn(b"kv_page_split_gpu\0")?;
            let fn_copy_batch = get_fn(b"kv_copy_batch_gpu\0")?;

            cu_ok(
                unsafe {
                    (api.func_set_attribute)(
                        fn_pred_enc,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        65536,
                    )
                },
                "cuFuncSetAttribute(pred_enc)",
            )?;
            cu_ok(
                unsafe {
                    (api.func_set_attribute)(
                        fn_pred_dec,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        65536,
                    )
                },
                "cuFuncSetAttribute(pred_dec)",
            )?;

            let max_orig = page_plan.original_bytes.max(row_plan.original_bytes);
            let max_mats = page_plan.num_mats.max(row_plan.num_mats);
            let max_chunks = page_plan.num_chunks().max(row_plan.num_chunks()).max(1);
            let max_ans_in = page_plan.ans_total_bytes.max(row_plan.ans_total_bytes).max(64);
            let st = stream as CUstream;

            let mut max_out_chunk = 0usize;
            let mut ans_comp_temp_bytes = 0usize;
            let mut ans_decomp_temp_bytes = 0usize;
            nv_ok(
                unsafe { (api.ans_max_out)(ANS_CHUNK, AnsCompressOpts::default(), &mut max_out_chunk) },
                "nvcompBatchedANSCompressGetMaxOutputChunkSize",
            )?;
            nv_ok(
                unsafe {
                    (api.ans_comp_temp)(
                        max_chunks,
                        ANS_CHUNK,
                        AnsCompressOpts::default(),
                        &mut ans_comp_temp_bytes,
                        max_ans_in,
                        st,
                    )
                },
                "nvcompBatchedANSCompressGetTempSize",
            )?;
            nv_ok(
                unsafe {
                    (api.ans_decomp_temp)(
                        max_chunks,
                        ANS_CHUNK,
                        AnsDecompressOpts::default(),
                        &mut ans_decomp_temp_bytes,
                        max_ans_in,
                        st,
                    )
                },
                "nvcompBatchedANSDecompressGetTempSize",
            )?;
            max_out_chunk = (max_out_chunk.max(ANS_CHUNK) + 15) & !15usize;

            let low_pad_bytes = max_mats.max(1) * 4 * LANE_STRIDE * 4;
            let mat_z_bytes = max_mats.max(1) * MATRIX_BYTES;
            let max_pkt = page_plan.max_packet().max(row_plan.max_packet()).max(max_orig);
            let max_copies = max_chunks
                .max(max_mats)
                .max(page_plan.coalesce.len())
                .max(row_plan.coalesce.len())
                .max(max_spans)
                .max(64);

            let mut codec = Self {
                module,
                fn_pred_enc,
                fn_pred_dec,
                fn_pack,
                fn_compact,
                fn_unpack,
                fn_shuffle,
                fn_page_split,
                fn_copy_batch,
                d_raw: DevBuf::alloc(max_orig)?,
                d_trans: DevBuf::alloc(max_orig)?,
                d_pkt: DevBuf::alloc(max_pkt)?,
                d_mat_z: DevBuf::alloc(mat_z_bytes)?,
                d_mat_ptrs: DevBuf::alloc(max_mats.max(1) * 8)?,
                d_low_pad: DevBuf::alloc(low_pad_bytes)?,
                d_lane_u32s: DevBuf::alloc(max_mats.max(1) * 4 * 2)?,
                d_lane_offs: DevBuf::alloc(max_mats.max(1) * 4 * 4)?,
                d_err_flag: DevBuf::alloc(4)?,
                d_ans_in: DevBuf::alloc(max_ans_in)?,
                d_ans_out: DevBuf::alloc(max_chunks * max_out_chunk)?,
                d_ans_temp: DevBuf::alloc(ans_comp_temp_bytes.max(ans_decomp_temp_bytes) + 256)?,
                d_in_ptrs: DevBuf::alloc(max_chunks * 8)?,
                d_out_ptrs: DevBuf::alloc(max_chunks * 8)?,
                d_in_sizes: DevBuf::alloc(max_chunks * 8)?,
                d_out_sizes: DevBuf::alloc(max_chunks * 8)?,
                d_rt_sizes: DevBuf::alloc(max_chunks * 8)?,
                d_statuses: DevBuf::alloc(max_chunks * 4)?,
                d_copy_dst: DevBuf::alloc(max_copies * 8)?,
                d_copy_src: DevBuf::alloc(max_copies * 8)?,
                d_copy_len: DevBuf::alloc(max_copies * 8)?,
                d_crc_ptr: DevBuf::alloc(8)?,
                d_crc_size: DevBuf::alloc(8)?,
                d_crc_out: DevBuf::alloc(4)?,
                d_crc_status: DevBuf::alloc(4)?,
                h_pinned: PinnedBuf::alloc(max_pkt)?,
                max_out_chunk,
                max_copies,
                ans_comp_temp_bytes,
                ans_decomp_temp_bytes,
                h_ptr_buf: Vec::with_capacity(max_copies),
                h_ptr2_buf: Vec::with_capacity(max_copies),
                h_len_buf: Vec::with_capacity(max_copies),
                h_size_buf: Vec::with_capacity(max_chunks),
                h_lanes_buf: Vec::with_capacity(max_mats * 4),
                h_lane_offs_buf: Vec::with_capacity(max_mats * 4),
                h_status_buf: vec![0; max_chunks],
            };
            // Warm up nvCOMP CRC32 + ANS kernels during startup so CUDA runtime state and
            // modules are initialized before steady-state serving (zero lazy steady-state alloc).
            codec.warmup(st)?;
            Ok(codec)
        }

        fn warmup(&mut self, st: CUstream) -> Result<()> {
            let api = api()?;
            let len = 64usize;
            cu_ok(
                unsafe { (api.memset_d8_async)(self.d_ans_in.ptr, 0, len, st) },
                "memset(warmup)",
            )?;
            self.launch_gpu_crc32(self.d_ans_in.ptr, len, st)?;
            let in_ptr = self.d_ans_in.ptr;
            let out_ptr = self.d_ans_out.ptr;
            cu_ok(
                unsafe { (api.memcpy_htod_async)(self.d_in_ptrs.ptr, &in_ptr as *const _ as _, 8, st) },
                "htod(warmup_in_ptr)",
            )?;
            cu_ok(
                unsafe { (api.memcpy_htod_async)(self.d_out_ptrs.ptr, &out_ptr as *const _ as _, 8, st) },
                "htod(warmup_out_ptr)",
            )?;
            cu_ok(
                unsafe { (api.memcpy_htod_async)(self.d_in_sizes.ptr, &len as *const _ as _, 8, st) },
                "htod(warmup_in_sz)",
            )?;
            nv_ok(
                unsafe {
                    (api.ans_comp_async)(
                        self.d_in_ptrs.ptr as *const *const c_void,
                        self.d_in_sizes.ptr as *const usize,
                        ANS_CHUNK,
                        1,
                        self.d_ans_temp.ptr as *mut c_void,
                        self.ans_comp_temp_bytes,
                        self.d_out_ptrs.ptr as *const *mut c_void,
                        self.d_out_sizes.ptr as *mut usize,
                        AnsCompressOpts::default(),
                        self.d_statuses.ptr as *mut NvcompStatus,
                        st,
                    )
                },
                "warmup(nvcompBatchedANSCompressAsync)",
            )?;
            nv_ok(
                unsafe {
                    (api.ans_decomp_async)(
                        self.d_out_ptrs.ptr as *const *const c_void,
                        self.d_out_sizes.ptr as *const usize,
                        self.d_in_sizes.ptr as *const usize,
                        self.d_rt_sizes.ptr as *mut usize,
                        1,
                        self.d_ans_temp.ptr as *mut c_void,
                        self.ans_decomp_temp_bytes,
                        self.d_in_ptrs.ptr as *const *mut c_void,
                        AnsDecompressOpts::default(),
                        self.d_statuses.ptr as *mut NvcompStatus,
                        st,
                    )
                },
                "warmup(nvcompBatchedANSDecompressAsync)",
            )?;
            cu_ok(unsafe { (api.stream_synchronize)(st) }, "sync(warmup)")
        }

        pub fn allocated_device_bytes(&self) -> u64 {
            (self.d_raw.bytes
                + self.d_trans.bytes
                + self.d_pkt.bytes
                + self.d_mat_z.bytes
                + self.d_mat_ptrs.bytes
                + self.d_low_pad.bytes
                + self.d_lane_u32s.bytes
                + self.d_lane_offs.bytes
                + self.d_err_flag.bytes
                + self.d_ans_in.bytes
                + self.d_ans_out.bytes
                + self.d_ans_temp.bytes
                + self.d_in_ptrs.bytes
                + self.d_out_ptrs.bytes
                + self.d_in_sizes.bytes
                + self.d_out_sizes.bytes
                + self.d_rt_sizes.bytes
                + self.d_statuses.bytes
                + self.d_copy_dst.bytes
                + self.d_copy_src.bytes
                + self.d_copy_len.bytes) as u64
        }

        fn launch_copy_batch(&mut self, count: usize, st: CUstream) -> Result<()> {
            if count == 0 {
                return Ok(());
            }
            if count > self.max_copies {
                return Err(invalid(format!(
                    "GPU copy batch count {count} exceeds allocated descriptor capacity {}",
                    self.max_copies
                )));
            }
            let api = api()?;
            cu_ok(
                unsafe {
                    (api.memcpy_htod_async)(
                        self.d_copy_dst.ptr,
                        self.h_ptr_buf.as_ptr() as _,
                        count * 8,
                        st,
                    )
                },
                "htod(copy_dst)",
            )?;
            cu_ok(
                unsafe {
                    (api.memcpy_htod_async)(
                        self.d_copy_src.ptr,
                        self.h_ptr2_buf.as_ptr() as _,
                        count * 8,
                        st,
                    )
                },
                "htod(copy_src)",
            )?;
            cu_ok(
                unsafe {
                    (api.memcpy_htod_async)(
                        self.d_copy_len.ptr,
                        self.h_len_buf.as_ptr() as _,
                        count * 8,
                        st,
                    )
                },
                "htod(copy_len)",
            )?;
            let mut dp = self.d_copy_dst.ptr;
            let mut sp = self.d_copy_src.ptr;
            let mut lp = self.d_copy_len.ptr;
            let mut n = count as c_int;
            let mut args: [*mut c_void; 4] = [
                &mut dp as *mut _ as _,
                &mut sp as *mut _ as _,
                &mut lp as *mut _ as _,
                &mut n as *mut _ as _,
            ];
            cu_ok(
                unsafe {
                    (api.launch_kernel)(
                        self.fn_copy_batch,
                        count as c_uint,
                        1,
                        1,
                        128,
                        1,
                        1,
                        0,
                        st,
                        args.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                },
                "launch(kv_copy_batch_gpu)",
            )
        }

        fn launch_gpu_crc32(&mut self, dev_ptr: u64, len: usize, st: CUstream) -> Result<()> {
            let api = api()?;
            let mut conf = Crc32KernelConf::default();
            nv_ok(
                unsafe { (api.crc32_conf)(std::ptr::null(), 1, &mut conf, len, st) },
                "nvcompBatchedCRC32GetHeuristicConf",
            )?;
            let opts = Crc32Opts {
                spec: Crc32Spec::ieee(),
                kernel_conf: conf,
                reserved: [0; 64],
            };
            let ptr_val = dev_ptr;
            let len_val = len;
            cu_ok(
                unsafe {
                    (api.memcpy_htod_async)(self.d_crc_ptr.ptr, &ptr_val as *const _ as _, 8, st)
                },
                "htod(crc_ptr)",
            )?;
            cu_ok(
                unsafe {
                    (api.memcpy_htod_async)(self.d_crc_size.ptr, &len_val as *const _ as _, 8, st)
                },
                "htod(crc_size)",
            )?;
            nv_ok(
                unsafe {
                    (api.crc32_async)(
                        self.d_crc_ptr.ptr as *const *const c_void,
                        self.d_crc_size.ptr as *const usize,
                        1,
                        self.d_crc_out.ptr as *mut u32,
                        opts,
                        0, // nvcompCRC32OnlySegment
                        self.d_crc_status.ptr as *mut NvcompStatus,
                        st,
                    )
                },
                "nvcompBatchedCRC32Async",
            )
        }

        /// Encode one unit directly from GPU cache plane spans into `host_out` (`GKV1` packet or `Literal` raw).
        /// Returns `(PackedUnit, bytes_written_to_host_out)`.
        pub fn encode_from_device(
            &mut self,
            plan: &GpuPlan,
            unit: PackedUnit,
            profile: &[u8; 32],
            spans: &[DeviceSpan],
            host_out: &mut [u8],
            stream: usize,
        ) -> Result<(PackedUnit, usize)> {
            if host_out.len() < plan.original_bytes {
                return Err(invalid("GPU encode host buffer smaller than original unit"));
            }
            let (unit, used) =
                self.encode_into_pinned(plan, unit, profile, spans, stream)?;
            host_out[..used].copy_from_slice(self.h_pinned.as_slice(used));
            Ok((unit, used))
        }

        /// Encode one unit directly from GPU cache plane spans into a freshly sized `Vec<u8>`
        /// with a single copy out of pinned DMA staging.
        pub fn encode_from_device_vec(
            &mut self,
            plan: &GpuPlan,
            unit: PackedUnit,
            profile: &[u8; 32],
            spans: &[DeviceSpan],
            stream: usize,
        ) -> Result<(PackedUnit, Vec<u8>)> {
            let (unit, used) =
                self.encode_into_pinned(plan, unit, profile, spans, stream)?;
            Ok((unit, self.h_pinned.as_slice(used).to_vec()))
        }

        /// Encode one unit directly from GPU cache plane spans and pass the
        /// resulting pinned DMA slice directly to `f` without heap allocation.
        pub fn encode_from_device_with<F, R>(
            &mut self,
            plan: &GpuPlan,
            unit: PackedUnit,
            profile: &[u8; 32],
            spans: &[DeviceSpan],
            stream: usize,
            f: F,
        ) -> Result<R>
        where
            F: FnOnce(PackedUnit, &[u8]) -> R,
        {
            let (unit, used) =
                self.encode_into_pinned(plan, unit, profile, spans, stream)?;
            Ok(f(unit, self.h_pinned.as_slice(used)))
        }

        fn encode_into_pinned(
            &mut self,
            plan: &GpuPlan,
            mut unit: PackedUnit,
            profile: &[u8; 32],
            spans: &[DeviceSpan],
            stream: usize,
        ) -> Result<(PackedUnit, usize)> {
            let api = api()?;
            let st = stream as CUstream;
            let n = plan.original_bytes;

            // 1. When the unit is a single contiguous full-size device span (e.g. STATE_ROWS),
            //    read directly from the immutable GPU checkpoint without copying into `d_raw`.
            let full_contiguous = spans.len() == 1 && spans[0].offset == 0 && spans[0].length == n;
            let raw_dev = if full_contiguous {
                spans[0].device
            } else {
                cu_ok(
                    unsafe { (api.memset_d8_async)(self.d_raw.ptr, 0, n, st) },
                    "memset(d_raw)",
                )?;
                if spans.len() > 1 {
                    self.h_ptr_buf.clear();
                    self.h_ptr2_buf.clear();
                    self.h_len_buf.clear();
                    for s in spans {
                        if s.offset.checked_add(s.length).is_none_or(|e| e > n) {
                            return Err(invalid("GPU encode span exceeds unit size"));
                        }
                        if s.length > 0 {
                            self.h_ptr_buf.push(self.d_raw.ptr + s.offset as u64);
                            self.h_ptr2_buf.push(s.device);
                            self.h_len_buf.push(s.length as u64);
                        }
                    }
                    let cnt = self.h_ptr_buf.len();
                    self.launch_copy_batch(cnt, st)?;
                } else if let Some(s) = spans.first() {
                    if s.offset.checked_add(s.length).is_none_or(|e| e > n) {
                        return Err(invalid("GPU encode span exceeds unit size"));
                    }
                    if s.length > 0 {
                        cu_ok(
                            unsafe {
                                (api.memcpy_dtod_async)(
                                    self.d_raw.ptr + s.offset as u64,
                                    s.device,
                                    s.length,
                                    st,
                                )
                            },
                            "dtod(gather_span)",
                        )?;
                    }
                }
                self.d_raw.ptr
            };

            // 2. Launch original CRC32 on GPU asynchronously (read back at sync(encode_meta)).
            self.launch_gpu_crc32(raw_dev, n, st)?;

            // 3. Matrix predictor (1 batched launch across all layers) + 4-lane pack.
            if plan.num_mats > 0 {
                self.h_ptr_buf.clear();
                for &(off, count) in &plan.matrix_fields {
                    for i in 0..count {
                        self.h_ptr_buf
                            .push(raw_dev + (off + i * MATRIX_BYTES) as u64);
                    }
                }
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_mat_ptrs.ptr,
                            self.h_ptr_buf.as_ptr() as _,
                            plan.num_mats * 8,
                            st,
                        )
                    },
                    "htod(mat_ptrs)",
                )?;
                let mut raw_ptrs = self.d_mat_ptrs.ptr;
                let mut z_ptr = self.d_mat_z.ptr;
                let mut nm = plan.num_mats as c_int;
                let mut args: [*mut c_void; 3] = [
                    &mut raw_ptrs as *mut _ as _,
                    &mut z_ptr as *mut _ as _,
                    &mut nm as *mut _ as _,
                ];
                cu_ok(
                    unsafe {
                        (api.launch_kernel)(
                            self.fn_pred_enc,
                            plan.num_mats as c_uint,
                            1,
                            1,
                            256,
                            1,
                            1,
                            65536,
                            st,
                            args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "launch(kv_predict_f32_128_enc)",
                )?;
                let mut z_ptr = self.d_mat_z.ptr;
                let mut w_ptr = self.d_ans_in.ptr; // widths live at offset 0 of d_ans_in!
                let mut low_ptr = self.d_low_pad.ptr;
                let mut lanes_ptr = self.d_lane_u32s.ptr;
                let mut nm = plan.num_mats as c_int;
                let mut args: [*mut c_void; 5] = [
                    &mut z_ptr as *mut _ as _,
                    &mut w_ptr as *mut _ as _,
                    &mut low_ptr as *mut _ as _,
                    &mut lanes_ptr as *mut _ as _,
                    &mut nm as *mut _ as _,
                ];
                let grid = (plan.num_mats * 4).div_ceil(32) as c_uint;
                cu_ok(
                    unsafe {
                        (api.launch_kernel)(
                            self.fn_pack,
                            grid,
                            1,
                            1,
                            32,
                            1,
                            1,
                            0,
                            st,
                            args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "launch(kv_pack_4lane_gpu)",
                )?;
            }

            // 4. Non-matrix transforms (byte-shuffle / page-split) into `d_trans`.
            for op in &plan.transforms {
                let mut dst = self.d_trans.ptr + op.offset as u64;
                let mut src = raw_dev + op.offset as u64;
                let mut len = op.length;
                match op.encoding {
                    Encoding::Bytes => {
                        cu_ok(
                            unsafe { (api.memcpy_dtod_async)(dst, src, len, st) },
                            "dtod(bytes)",
                        )?;
                    }
                    Encoding::Bfloat16 => {
                        let mut width = 2u32;
                        let mut dec = 0i32;
                        let mut args: [*mut c_void; 5] = [
                            &mut dst as *mut _ as _,
                            &mut src as *mut _ as _,
                            &mut len as *mut _ as _,
                            &mut width as *mut _ as _,
                            &mut dec as *mut _ as _,
                        ];
                        let grid = len.div_ceil(256) as c_uint;
                        cu_ok(
                            unsafe {
                                (api.launch_kernel)(
                                    self.fn_shuffle,
                                    grid,
                                    1,
                                    1,
                                    256,
                                    1,
                                    1,
                                    0,
                                    st,
                                    args.as_mut_ptr(),
                                    std::ptr::null_mut(),
                                )
                            },
                            "launch(kv_byte_shuffle_gpu)",
                        )?;
                    }
                    Encoding::Latent528
                    | Encoding::Index132
                    | Encoding::Fp8Page512
                    | Encoding::WindowFp4
                    | Encoding::LatentFp4 => {
                        let (_, mut block, mut rows, mut cols, mut scale_w) =
                            page_params(op.encoding);
                        let (mut stride, mut planar) = page_stride_planar(op.encoding);
                        let mut dec = 0i32;
                        let mut args: [*mut c_void; 10] = [
                            &mut dst as *mut _ as _,
                            &mut src as *mut _ as _,
                            &mut len as *mut _ as _,
                            &mut block as *mut _ as _,
                            &mut rows as *mut _ as _,
                            &mut cols as *mut _ as _,
                            &mut stride as *mut _ as _,
                            &mut scale_w as *mut _ as _,
                            &mut planar as *mut _ as _,
                            &mut dec as *mut _ as _,
                        ];
                        let grid = len.div_ceil(256) as c_uint;
                        cu_ok(
                            unsafe {
                                (api.launch_kernel)(
                                    self.fn_page_split,
                                    grid,
                                    1,
                                    1,
                                    256,
                                    1,
                                    1,
                                    0,
                                    st,
                                    args.as_mut_ptr(),
                                    std::ptr::null_mut(),
                                )
                            },
                            "launch(kv_page_split_gpu)",
                        )?;
                    }
                    _ => unreachable!(),
                }
            }

            // 5. Coalesce non-matrix homogeneous lanes into `d_ans_in` in ONE batched kernel launch.
            if !plan.coalesce.is_empty() {
                self.h_ptr_buf.clear();
                self.h_ptr2_buf.clear();
                self.h_len_buf.clear();
                for c in &plan.coalesce {
                    self.h_ptr_buf.push(self.d_ans_in.ptr + c.ans_offset as u64);
                    self.h_ptr2_buf
                        .push(self.d_trans.ptr + c.trans_offset as u64);
                    self.h_len_buf.push(c.length as u64);
                }
                let cnt = self.h_ptr_buf.len();
                self.launch_copy_batch(cnt, st)?;
            }

            // 6. Run ONE batched `nvcompBatchedANSCompressAsync` across all coalesced chunks.
            let nc = plan.num_chunks();
            if nc > 0 {
                self.h_ptr_buf.clear();
                for &off in &plan.chunk_offsets {
                    self.h_ptr_buf.push(self.d_ans_in.ptr + off as u64);
                }
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_in_ptrs.ptr,
                            self.h_ptr_buf.as_ptr() as _,
                            nc * 8,
                            st,
                        )
                    },
                    "htod(in_ptrs)",
                )?;
                self.h_ptr_buf.clear();
                for i in 0..nc {
                    self.h_ptr_buf
                        .push(self.d_ans_out.ptr + (i * self.max_out_chunk) as u64);
                }
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_out_ptrs.ptr,
                            self.h_ptr_buf.as_ptr() as _,
                            nc * 8,
                            st,
                        )
                    },
                    "htod(out_ptrs)",
                )?;
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_in_sizes.ptr,
                            plan.chunk_sizes.as_ptr() as _,
                            nc * 8,
                            st,
                        )
                    },
                    "htod(in_sizes)",
                )?;
                nv_ok(
                    unsafe {
                        (api.ans_comp_async)(
                            self.d_in_ptrs.ptr as *const *const c_void,
                            self.d_in_sizes.ptr as *const usize,
                            ANS_CHUNK,
                            nc,
                            self.d_ans_temp.ptr as *mut c_void,
                            self.ans_comp_temp_bytes,
                            self.d_out_ptrs.ptr as *const *mut c_void,
                            self.d_out_sizes.ptr as *mut usize,
                            AnsCompressOpts::default(),
                            self.d_statuses.ptr as *mut NvcompStatus,
                            st,
                        )
                    },
                    "nvcompBatchedANSCompressAsync",
                )?;
            }

            // 7. Read back original_crc, CRC status, chunk compressed sizes, and matrix lane word counts (1 sync).
            let mut original_crc = 0u32;
            let mut orig_crc_status: NvcompStatus = -1;
            cu_ok(
                unsafe {
                    (api.memcpy_dtoh_async)(
                        &mut original_crc as *mut _ as _,
                        self.d_crc_out.ptr,
                        4,
                        st,
                    )
                },
                "dtoh(orig_crc)",
            )?;
            cu_ok(
                unsafe {
                    (api.memcpy_dtoh_async)(
                        &mut orig_crc_status as *mut _ as _,
                        self.d_crc_status.ptr,
                        4,
                        st,
                    )
                },
                "dtoh(orig_crc_status)",
            )?;
            self.h_size_buf.resize(nc, 0);
            self.h_status_buf.resize(nc, 0);
            if nc > 0 {
                cu_ok(
                    unsafe {
                        (api.memcpy_dtoh_async)(
                            self.h_size_buf.as_mut_ptr() as _,
                            self.d_out_sizes.ptr,
                            nc * 8,
                            st,
                        )
                    },
                    "dtoh(out_sizes)",
                )?;
                cu_ok(
                    unsafe {
                        (api.memcpy_dtoh_async)(
                            self.h_status_buf.as_mut_ptr() as _,
                            self.d_statuses.ptr,
                            nc * 4,
                            st,
                        )
                    },
                    "dtoh(statuses)",
                )?;
            }
            let total_lanes = plan.num_mats * 4;
            self.h_lanes_buf.resize(total_lanes, 0);
            if total_lanes > 0 {
                cu_ok(
                    unsafe {
                        (api.memcpy_dtoh_async)(
                            self.h_lanes_buf.as_mut_ptr() as _,
                            self.d_lane_u32s.ptr,
                            total_lanes * 2,
                            st,
                        )
                    },
                    "dtoh(lane_u32s)",
                )?;
            }
            cu_ok(unsafe { (api.stream_synchronize)(st) }, "sync(encode_meta)")?;
            nv_ok(orig_crc_status, "nvcompBatchedCRC32(orig)")?;

            // Compute compacted low-bit byte offsets & check total packet size vs raw.
            let mut low_bytes = 0usize;
            self.h_lane_offs_buf.clear();
            for &nw in &self.h_lanes_buf {
                self.h_lane_offs_buf.push(low_bytes as u32);
                low_bytes += usize::from(nw) * 4;
            }
            let mut ans_payload = 0usize;
            for i in 0..nc {
                let raw_sz = plan.chunk_sizes[i];
                let comp_sz = if self.h_status_buf[i] == NVCOMP_SUCCESS {
                    self.h_size_buf[i]
                } else {
                    raw_sz
                };
                ans_payload += if comp_sz < raw_sz { comp_sz } else { raw_sz };
            }
            let meta = plan.metadata_bytes();
            let total_pkt = meta + ans_payload + low_bytes;

            unit.original_length = n as u64;
            if total_pkt >= n {
                // Incompressible: emit Literal raw copy directly via pinned DMA!
                cu_ok(
                    unsafe {
                        (api.memcpy_dtoh_async)(self.h_pinned.ptr as _, raw_dev, n, st)
                    },
                    "dtoh(literal_fallback)",
                )?;
                cu_ok(unsafe { (api.stream_synchronize)(st) }, "sync(literal)")?;
                unit.representation = Representation::Literal as i32;
                unit.profile.clear();
                unit.stored_length = n as u64;
                unit.stored_crc = 0;
                unit.original_crc = 0;
                unit.raw_xxh3 = crate::record::checksum(self.h_pinned.as_slice(n));
                return Ok((unit, n));
            }

            // 8. Build header+dir+lanes in pinned host buffer, copy tiny metadata to `d_pkt`,
            //    compact all ANS chunks + low-bit lanes directly into contiguous `d_pkt` on GPU,
            //    compute `stored_crc` on GPU, and D2H the whole finished packet in ONE pinned DMA!
            {
                let hdr_dst = self.h_pinned.as_mut_slice(meta);
                let hdr_words = [
                    MAGIC,
                    VERSION,
                    n as u32,
                    nc as u32,
                    plan.num_mats as u32,
                    low_bytes as u32,
                ];
                for (dst, w) in hdr_dst[..HEADER_BYTES].chunks_exact_mut(4).zip(hdr_words) {
                    dst.copy_from_slice(&w.to_le_bytes());
                }
                let dir_end = HEADER_BYTES + nc * 4;
                let mut pos = meta;
                self.h_ptr_buf.clear();
                self.h_ptr2_buf.clear();
                self.h_len_buf.clear();
                for i in 0..nc {
                    let raw_sz = plan.chunk_sizes[i];
                    let use_comp =
                        self.h_status_buf[i] == NVCOMP_SUCCESS && self.h_size_buf[i] < raw_sz;
                    let (word, len, dev_src) = if use_comp {
                        let len = self.h_size_buf[i];
                        (
                            len as u32,
                            len,
                            self.d_ans_out.ptr + (i * self.max_out_chunk) as u64,
                        )
                    } else {
                        (
                            RAW_FLAG | raw_sz as u32,
                            raw_sz,
                            self.d_ans_in.ptr + plan.chunk_offsets[i] as u64,
                        )
                    };
                    hdr_dst[HEADER_BYTES + i * 4..HEADER_BYTES + (i + 1) * 4]
                        .copy_from_slice(&word.to_le_bytes());
                    self.h_ptr_buf.push(self.d_pkt.ptr + pos as u64);
                    self.h_ptr2_buf.push(dev_src);
                    self.h_len_buf.push(len as u64);
                    pos += len;
                }
                for (i, &nw) in self.h_lanes_buf.iter().enumerate() {
                    hdr_dst[dir_end + i * 2..dir_end + (i + 1) * 2]
                        .copy_from_slice(&nw.to_le_bytes());
                }
                debug_assert_eq!(pos, meta + ans_payload);
            }
            cu_ok(
                unsafe {
                    (api.memcpy_htod_async)(self.d_pkt.ptr, self.h_pinned.ptr as _, meta, st)
                },
                "htod(pkt_meta)",
            )?;
            let copy_cnt = self.h_ptr_buf.len();
            self.launch_copy_batch(copy_cnt, st)?;

            if plan.num_mats > 0 && low_bytes > 0 {
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_lane_offs.ptr,
                            self.h_lane_offs_buf.as_ptr() as _,
                            total_lanes * 4,
                            st,
                        )
                    },
                    "htod(lane_offs)",
                )?;
                let mut lp = self.d_low_pad.ptr;
                let mut lu = self.d_lane_u32s.ptr;
                let mut lo = self.d_lane_offs.ptr;
                let mut lc = self.d_pkt.ptr + (meta + ans_payload) as u64;
                let mut nm = plan.num_mats as c_int;
                let mut args: [*mut c_void; 5] = [
                    &mut lp as *mut _ as _,
                    &mut lu as *mut _ as _,
                    &mut lo as *mut _ as _,
                    &mut lc as *mut _ as _,
                    &mut nm as *mut _ as _,
                ];
                cu_ok(
                    unsafe {
                        (api.launch_kernel)(
                            self.fn_compact,
                            total_lanes as c_uint,
                            1,
                            1,
                            128,
                            1,
                            1,
                            0,
                            st,
                            args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "launch(kv_compact_low_lanes)",
                )?;
            }

            // Compute stored_crc on GPU over the complete `d_pkt[0..total_pkt]`, then ONE pinned D2H!
            self.launch_gpu_crc32(self.d_pkt.ptr, total_pkt, st)?;
            let mut stored_crc = 0u32;
            let mut stored_crc_status: NvcompStatus = -1;
            cu_ok(
                unsafe {
                    (api.memcpy_dtoh_async)(
                        &mut stored_crc as *mut _ as _,
                        self.d_crc_out.ptr,
                        4,
                        st,
                    )
                },
                "dtoh(stored_crc)",
            )?;
            cu_ok(
                unsafe {
                    (api.memcpy_dtoh_async)(
                        &mut stored_crc_status as *mut _ as _,
                        self.d_crc_status.ptr,
                        4,
                        st,
                    )
                },
                "dtoh(stored_crc_status)",
            )?;
            cu_ok(
                unsafe {
                    (api.memcpy_dtoh_async)(
                        self.h_pinned.ptr.add(meta) as _,
                        self.d_pkt.ptr + meta as u64,
                        total_pkt - meta,
                        st,
                    )
                },
                "dtoh(pkt_body)",
            )?;
            cu_ok(unsafe { (api.stream_synchronize)(st) }, "sync(encode_d2h)")?;
            nv_ok(stored_crc_status, "nvcompBatchedCRC32(stored)")?;

            unit.representation = Representation::Packet as i32;
            unit.profile = profile.to_vec();
            unit.stored_length = total_pkt as u64;
            unit.stored_crc = stored_crc;
            unit.original_crc = original_crc;
            unit.raw_xxh3 = 0;
            Ok((unit, total_pkt))
        }

        /// Verify stored CRC32 on GPU, decode a `GKV1` packet on GPU, verify original CRC32 on GPU,
        /// and scatter the verified bytes into the destination GPU cache planes.
        pub fn decode_to_device(
            &mut self,
            plan: &GpuPlan,
            chunks: &[&[u8]],
            spans: &[DeviceSpan],
            stored_crc: u32,
            original_crc: u32,
            stream: usize,
        ) -> Result<()> {
            let total: usize = chunks.iter().map(|c| c.len()).sum();
            if total < plan.metadata_bytes() || total > plan.max_packet() {
                if crc32_chunks(chunks) != stored_crc {
                    return Err(Error::new(ErrorKind::InvalidData, Checksum::Stored));
                }
                return Err(invalid("invalid GKV1 packet length"));
            }

            // Copy packet chunks into pinned host staging once, then ONE pinned H2D into `d_pkt`.
            {
                let dst = self.h_pinned.as_mut_slice(total);
                let mut off = 0usize;
                for c in chunks {
                    dst[off..off + c.len()].copy_from_slice(c);
                    off += c.len();
                }
            }

            let api = api()?;
            let st = stream as CUstream;
            cu_ok(
                unsafe {
                    (api.memcpy_htod_async)(self.d_pkt.ptr, self.h_pinned.ptr as _, total, st)
                },
                "htod(pkt)",
            )?;
            // Launch GPU stored_crc immediately on `d_pkt`!
            self.launch_gpu_crc32(self.d_pkt.ptr, total, st)?;
            let mut got_stored_crc = 0u32;
            let mut stored_crc_status: NvcompStatus = -1;
            cu_ok(
                unsafe {
                    (api.memcpy_dtoh_async)(
                        &mut got_stored_crc as *mut _ as _,
                        self.d_crc_out.ptr,
                        4,
                        st,
                    )
                },
                "dtoh(stored_crc)",
            )?;
            cu_ok(
                unsafe {
                    (api.memcpy_dtoh_async)(
                        &mut stored_crc_status as *mut _ as _,
                        self.d_crc_status.ptr,
                        4,
                        st,
                    )
                },
                "dtoh(stored_crc_status)",
            )?;

            // Validate header, chunk directory, and lane table from pinned buffer while GPU runs stored_crc!
            let pkt = self.h_pinned.as_slice(total);
            let word = |p: usize| u32::from_le_bytes(pkt[p..p + 4].try_into().unwrap());
            let n = plan.original_bytes;
            let nc = plan.num_chunks();
            let header_valid = word(0) == MAGIC
                && word(4) == VERSION
                && word(8) as usize == n
                && word(12) as usize == nc
                && word(16) as usize == plan.num_mats;
            if !header_valid {
                cu_ok(unsafe { (api.stream_synchronize)(st) }, "sync(stored_crc_err)")?;
                nv_ok(stored_crc_status, "nvcompBatchedCRC32(stored_decode)")?;
                if got_stored_crc != stored_crc {
                    return Err(Error::new(ErrorKind::InvalidData, Checksum::Stored));
                }
                return Err(invalid("invalid GKV1 packet header"));
            }
            let low_bytes = word(20) as usize;
            let dir_end = HEADER_BYTES + nc * 4;
            let total_lanes = plan.num_mats * 4;
            let meta = plan.metadata_bytes();

            let mut pos = meta;
            let mut comp_indices = Vec::with_capacity(nc);
            let mut comp_ranges = Vec::with_capacity(nc);
            let mut raw_copies = Vec::new();
            let mut dir_err: Option<&'static str> = None;
            for i in 0..nc {
                let entry = word(HEADER_BYTES + i * 4);
                let is_raw = (entry & RAW_FLAG) != 0;
                let len = (entry & !RAW_FLAG) as usize;
                let expected_raw = plan.chunk_sizes[i];
                if len > total - pos {
                    dir_err = Some("truncated GKV1 chunk");
                    break;
                }
                if is_raw {
                    if len != expected_raw {
                        dir_err = Some("invalid GKV1 raw chunk length");
                        break;
                    }
                    raw_copies.push((i, pos, len));
                } else {
                    if len == 0 || len >= expected_raw || len > self.max_out_chunk {
                        dir_err = Some("invalid GKV1 compressed chunk length");
                        break;
                    }
                    comp_indices.push(i);
                    comp_ranges.push((pos, len));
                }
                pos += len;
            }
            if dir_err.is_none() && pos + low_bytes != total {
                dir_err = Some("GKV1 payload length mismatch");
            }
            self.h_lanes_buf.clear();
            self.h_lane_offs_buf.clear();
            let mut expected_low = 0usize;
            if dir_err.is_none() {
                for i in 0..total_lanes {
                    let nw =
                        u16::from_le_bytes([pkt[dir_end + i * 2], pkt[dir_end + i * 2 + 1]]);
                    if usize::from(nw) > MAX_LANE_U32S {
                        dir_err = Some("GKV1 matrix lane exceeds maximum words");
                        break;
                    }
                    self.h_lanes_buf.push(nw);
                    self.h_lane_offs_buf.push(expected_low as u32);
                    expected_low += usize::from(nw) * 4;
                }
                if dir_err.is_none() && expected_low != low_bytes {
                    dir_err = Some("GKV1 matrix low-bit byte count mismatch");
                }
            }
            // Verify stored CRC32 BEFORE handing any chunk payload to nvCOMP ANS decompression.
            cu_ok(unsafe { (api.stream_synchronize)(st) }, "sync(stored_crc)")?;
            nv_ok(stored_crc_status, "nvcompBatchedCRC32(stored_decode)")?;
            if got_stored_crc != stored_crc {
                return Err(Error::new(ErrorKind::InvalidData, Checksum::Stored));
            }
            if let Some(msg) = dir_err {
                return Err(invalid(msg));
            }

            // 1. Copy raw fallback chunks (`d_pkt` -> `d_ans_in`) and align compressed chunks
            //    (`d_pkt` -> 16-byte-aligned `d_ans_out` slots) in ONE batched GPU kernel launch!
            let ncomp = comp_indices.len();
            if !raw_copies.is_empty() || ncomp > 0 {
                self.h_ptr_buf.clear();
                self.h_ptr2_buf.clear();
                self.h_len_buf.clear();
                for &(i, off, len) in &raw_copies {
                    self.h_ptr_buf
                        .push(self.d_ans_in.ptr + plan.chunk_offsets[i] as u64);
                    self.h_ptr2_buf.push(self.d_pkt.ptr + off as u64);
                    self.h_len_buf.push(len as u64);
                }
                for (slot, &(off, len)) in comp_ranges.iter().enumerate() {
                    self.h_ptr_buf
                        .push(self.d_ans_out.ptr + (slot * self.max_out_chunk) as u64);
                    self.h_ptr2_buf.push(self.d_pkt.ptr + off as u64);
                    self.h_len_buf.push(len as u64);
                }
                let cnt = self.h_ptr_buf.len();
                self.launch_copy_batch(cnt, st)?;
            }

            let mut actual_sizes = Vec::new();
            if ncomp > 0 {
                self.h_ptr_buf.clear();
                self.h_ptr2_buf.clear();
                self.h_size_buf.clear();
                let mut out_caps = Vec::with_capacity(ncomp);
                for (slot, (&ci, &(_off, len))) in
                    comp_indices.iter().zip(&comp_ranges).enumerate()
                {
                    self.h_ptr_buf
                        .push(self.d_ans_out.ptr + (slot * self.max_out_chunk) as u64);
                    self.h_size_buf.push(len);
                    self.h_ptr2_buf
                        .push(self.d_ans_in.ptr + plan.chunk_offsets[ci] as u64);
                    out_caps.push(plan.chunk_sizes[ci]);
                }
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_out_ptrs.ptr,
                            self.h_ptr_buf.as_ptr() as _,
                            ncomp * 8,
                            st,
                        )
                    },
                    "htod(decomp_in_ptrs)",
                )?;
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_out_sizes.ptr,
                            self.h_size_buf.as_ptr() as _,
                            ncomp * 8,
                            st,
                        )
                    },
                    "htod(decomp_in_sizes)",
                )?;
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_in_ptrs.ptr,
                            self.h_ptr2_buf.as_ptr() as _,
                            ncomp * 8,
                            st,
                        )
                    },
                    "htod(decomp_out_ptrs)",
                )?;
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_in_sizes.ptr,
                            out_caps.as_ptr() as _,
                            ncomp * 8,
                            st,
                        )
                    },
                    "htod(decomp_out_caps)",
                )?;
                nv_ok(
                    unsafe {
                        (api.ans_decomp_async)(
                            self.d_out_ptrs.ptr as *const *const c_void,
                            self.d_out_sizes.ptr as *const usize,
                            self.d_in_sizes.ptr as *const usize,
                            self.d_rt_sizes.ptr as *mut usize,
                            ncomp,
                            self.d_ans_temp.ptr as *mut c_void,
                            self.ans_decomp_temp_bytes,
                            self.d_in_ptrs.ptr as *const *mut c_void,
                            AnsDecompressOpts::default(),
                            self.d_statuses.ptr as *mut NvcompStatus,
                            st,
                        )
                    },
                    "nvcompBatchedANSDecompressAsync",
                )?;
                self.h_status_buf.resize(ncomp, 0);
                actual_sizes.resize(ncomp, 0);
                cu_ok(
                    unsafe {
                        (api.memcpy_dtoh_async)(
                            self.h_status_buf.as_mut_ptr() as _,
                            self.d_statuses.ptr,
                            ncomp * 4,
                            st,
                        )
                    },
                    "dtoh(decomp_statuses)",
                )?;
                cu_ok(
                    unsafe {
                        (api.memcpy_dtoh_async)(
                            actual_sizes.as_mut_ptr() as _,
                            self.d_rt_sizes.ptr,
                            ncomp * 8,
                            st,
                        )
                    },
                    "dtoh(decomp_sizes)",
                )?;
            }
            if ncomp > 0 {
                cu_ok(unsafe { (api.stream_synchronize)(st) }, "sync(decomp)")?;
            }
            for (slot, &ci) in comp_indices.iter().enumerate() {
                if self.h_status_buf[slot] != NVCOMP_SUCCESS
                    || actual_sizes[slot] != plan.chunk_sizes[ci]
                {
                    return Err(invalid("invalid nvCOMP ANS data in GKV1 packet"));
                }
            }

            // 2. Uncoalesce non-matrix homogeneous lanes back into `d_trans` in ONE launch.
            if !plan.coalesce.is_empty() {
                self.h_ptr_buf.clear();
                self.h_ptr2_buf.clear();
                self.h_len_buf.clear();
                for c in &plan.coalesce {
                    self.h_ptr_buf
                        .push(self.d_trans.ptr + c.trans_offset as u64);
                    self.h_ptr2_buf
                        .push(self.d_ans_in.ptr + c.ans_offset as u64);
                    self.h_len_buf.push(c.length as u64);
                }
                let cnt = self.h_ptr_buf.len();
                self.launch_copy_batch(cnt, st)?;
            }

            for op in &plan.transforms {
                let mut dst = self.d_raw.ptr + op.offset as u64;
                let mut src = self.d_trans.ptr + op.offset as u64;
                let mut len = op.length;
                match op.encoding {
                    Encoding::Bytes => {
                        cu_ok(
                            unsafe { (api.memcpy_dtod_async)(dst, src, len, st) },
                            "dtod(inv_bytes)",
                        )?;
                    }
                    Encoding::Bfloat16 => {
                        let mut width = 2u32;
                        let mut dec = 1i32;
                        let mut args: [*mut c_void; 5] = [
                            &mut dst as *mut _ as _,
                            &mut src as *mut _ as _,
                            &mut len as *mut _ as _,
                            &mut width as *mut _ as _,
                            &mut dec as *mut _ as _,
                        ];
                        let grid = len.div_ceil(256) as c_uint;
                        cu_ok(
                            unsafe {
                                (api.launch_kernel)(
                                    self.fn_shuffle,
                                    grid,
                                    1,
                                    1,
                                    256,
                                    1,
                                    1,
                                    0,
                                    st,
                                    args.as_mut_ptr(),
                                    std::ptr::null_mut(),
                                )
                            },
                            "launch(inv_byte_shuffle)",
                        )?;
                    }
                    Encoding::Latent528
                    | Encoding::Index132
                    | Encoding::Fp8Page512
                    | Encoding::WindowFp4
                    | Encoding::LatentFp4 => {
                        let (_, mut block, mut rows, mut cols, mut scale_w) =
                            page_params(op.encoding);
                        let (mut stride, mut planar) = page_stride_planar(op.encoding);
                        let mut dec = 1i32;
                        let mut args: [*mut c_void; 10] = [
                            &mut dst as *mut _ as _,
                            &mut src as *mut _ as _,
                            &mut len as *mut _ as _,
                            &mut block as *mut _ as _,
                            &mut rows as *mut _ as _,
                            &mut cols as *mut _ as _,
                            &mut stride as *mut _ as _,
                            &mut scale_w as *mut _ as _,
                            &mut planar as *mut _ as _,
                            &mut dec as *mut _ as _,
                        ];
                        let grid = len.div_ceil(256) as c_uint;
                        cu_ok(
                            unsafe {
                                (api.launch_kernel)(
                                    self.fn_page_split,
                                    grid,
                                    1,
                                    1,
                                    256,
                                    1,
                                    1,
                                    0,
                                    st,
                                    args.as_mut_ptr(),
                                    std::ptr::null_mut(),
                                )
                            },
                            "launch(inv_page_split)",
                        )?;
                    }
                    _ => unreachable!(),
                }
            }

            // 3. Matrix 4-lane unpack (directly from `d_pkt + pos`, zero extra H2D) + inverse predictor.
            let mut err_val = 0u32;
            if plan.num_mats > 0 {
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_lane_u32s.ptr,
                            self.h_lanes_buf.as_ptr() as _,
                            total_lanes * 2,
                            st,
                        )
                    },
                    "htod(lane_u32s)",
                )?;
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_lane_offs.ptr,
                            self.h_lane_offs_buf.as_ptr() as _,
                            total_lanes * 4,
                            st,
                        )
                    },
                    "htod(lane_offs)",
                )?;
                cu_ok(
                    unsafe { (api.memset_d8_async)(self.d_err_flag.ptr, 0, 4, st) },
                    "memset(err_flag)",
                )?;

                let mut zp = self.d_mat_z.ptr;
                let mut wp = self.d_ans_in.ptr;
                let mut lc = self.d_pkt.ptr + pos as u64;
                let mut lo = self.d_lane_offs.ptr;
                let mut lu = self.d_lane_u32s.ptr;
                let mut nm = plan.num_mats as c_int;
                let mut ef = self.d_err_flag.ptr;
                let mut args: [*mut c_void; 7] = [
                    &mut zp as *mut _ as _,
                    &mut wp as *mut _ as _,
                    &mut lc as *mut _ as _,
                    &mut lo as *mut _ as _,
                    &mut lu as *mut _ as _,
                    &mut nm as *mut _ as _,
                    &mut ef as *mut _ as _,
                ];
                let grid = total_lanes.div_ceil(32) as c_uint;
                cu_ok(
                    unsafe {
                        (api.launch_kernel)(
                            self.fn_unpack,
                            grid,
                            1,
                            1,
                            32,
                            1,
                            1,
                            0,
                            st,
                            args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "launch(kv_unpack_4lane_gpu)",
                )?;
                cu_ok(
                    unsafe {
                        (api.memcpy_dtoh_async)(
                            &mut err_val as *mut _ as _,
                            self.d_err_flag.ptr,
                            4,
                            st,
                        )
                    },
                    "dtoh(err_flag)",
                )?;

                self.h_ptr_buf.clear();
                for &(off, count) in &plan.matrix_fields {
                    for i in 0..count {
                        self.h_ptr_buf
                            .push(self.d_raw.ptr + (off + i * MATRIX_BYTES) as u64);
                    }
                }
                cu_ok(
                    unsafe {
                        (api.memcpy_htod_async)(
                            self.d_mat_ptrs.ptr,
                            self.h_ptr_buf.as_ptr() as _,
                            plan.num_mats * 8,
                            st,
                        )
                    },
                    "htod(mat_ptrs_dec)",
                )?;
                let mut raw_ptrs = self.d_mat_ptrs.ptr;
                let mut z_ptr = self.d_mat_z.ptr;
                let mut nm = plan.num_mats as c_int;
                let mut err_ptr = self.d_err_flag.ptr;
                let mut args: [*mut c_void; 4] = [
                    &mut raw_ptrs as *mut _ as _,
                    &mut z_ptr as *mut _ as _,
                    &mut nm as *mut _ as _,
                    &mut err_ptr as *mut _ as _,
                ];
                cu_ok(
                    unsafe {
                        (api.launch_kernel)(
                            self.fn_pred_dec,
                            plan.num_mats as c_uint,
                            1,
                            1,
                            256,
                            1,
                            1,
                            65536,
                            st,
                            args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "launch(kv_predict_f32_128_dec)",
                )?;
            }

            // 4. Verify original CRC32 on GPU (and unpack err_flag) BEFORE scattering into live GPU cache planes!
            self.launch_gpu_crc32(self.d_raw.ptr, n, st)?;
            let mut got_orig_crc = 0u32;
            let mut orig_crc_status: NvcompStatus = -1;
            cu_ok(
                unsafe {
                    (api.memcpy_dtoh_async)(
                        &mut got_orig_crc as *mut _ as _,
                        self.d_crc_out.ptr,
                        4,
                        st,
                    )
                },
                "dtoh(orig_crc)",
            )?;
            cu_ok(
                unsafe {
                    (api.memcpy_dtoh_async)(
                        &mut orig_crc_status as *mut _ as _,
                        self.d_crc_status.ptr,
                        4,
                        st,
                    )
                },
                "dtoh(orig_crc_status)",
            )?;
            cu_ok(unsafe { (api.stream_synchronize)(st) }, "sync(orig_crc)")?;
            if err_val != 0 {
                return Err(invalid("invalid matrix residual data in GKV1 packet"));
            }
            nv_ok(orig_crc_status, "nvcompBatchedCRC32(orig_decode)")?;
            if got_orig_crc != original_crc {
                return Err(Error::new(ErrorKind::InvalidData, Checksum::Original));
            }

            // 5. Scatter verified raw bytes into destination GPU cache planes and synchronize.
            if spans.len() > 1 {
                self.h_ptr_buf.clear();
                self.h_ptr2_buf.clear();
                self.h_len_buf.clear();
                for s in spans {
                    if s.offset.checked_add(s.length).is_none_or(|e| e > n) {
                        return Err(invalid("GPU decode span exceeds unit size"));
                    }
                    if s.length > 0 {
                        self.h_ptr_buf.push(s.device);
                        self.h_ptr2_buf.push(self.d_raw.ptr + s.offset as u64);
                        self.h_len_buf.push(s.length as u64);
                    }
                }
                let cnt = self.h_ptr_buf.len();
                self.launch_copy_batch(cnt, st)?;
            } else if let Some(s) = spans.first() {
                if s.offset.checked_add(s.length).is_none_or(|e| e > n) {
                    return Err(invalid("GPU decode span exceeds unit size"));
                }
                if s.length > 0 {
                    cu_ok(
                        unsafe {
                            (api.memcpy_dtod_async)(
                                s.device,
                                self.d_raw.ptr + s.offset as u64,
                                s.length,
                                st,
                            )
                        },
                        "dtod(scatter_span)",
                    )?;
                }
            }
            cu_ok(unsafe { (api.stream_synchronize)(st) }, "sync(scatter)")?;
            Ok(())
        }

        /// Copy verified raw `Literal` chunks into `d_raw`, scatter into destination GPU cache planes,
        /// and synchronize.
        pub fn copy_literal_to_device(
            &mut self,
            plan: &GpuPlan,
            chunks: &[&[u8]],
            spans: &[DeviceSpan],
            stream: usize,
        ) -> Result<()> {
            let api = api()?;
            let st = stream as CUstream;
            let n = plan.original_bytes;
            let mut pos = 0usize;
            for chunk in chunks {
                let end = pos
                    .checked_add(chunk.len())
                    .filter(|&e| e <= n)
                    .ok_or_else(|| invalid("literal length mismatch"))?;
                if !chunk.is_empty() {
                    cu_ok(
                        unsafe {
                            (api.memcpy_htod_async)(
                                self.d_raw.ptr + pos as u64,
                                chunk.as_ptr() as _,
                                chunk.len(),
                                st,
                            )
                        },
                        "htod(literal)",
                    )?;
                }
                pos = end;
            }
            if pos != n {
                return Err(invalid("literal length mismatch"));
            }
            for s in spans {
                if s.offset.checked_add(s.length).is_none_or(|e| e > n) {
                    return Err(invalid("GPU literal span exceeds unit size"));
                }
                if s.length > 0 {
                    cu_ok(
                        unsafe {
                            (api.memcpy_dtod_async)(
                                s.device,
                                self.d_raw.ptr + s.offset as u64,
                                s.length,
                                st,
                            )
                        },
                        "dtod(scatter_literal)",
                    )?;
                }
            }
            cu_ok(unsafe { (api.stream_synchronize)(st) }, "sync(literal_scatter)")?;
            Ok(())
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::*;
    pub struct GpuRankCodec;
    impl GpuRankCodec {
        pub fn check_runtime() -> Result<()> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "GPU KV codec requires Linux with CUDA and libnvcomp",
            ))
        }
        pub fn new(_: &GpuPlan, _: &GpuPlan, _: usize, _: &[u8], _: usize) -> Result<Self> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "GPU KV codec requires Linux with CUDA and libnvcomp",
            ))
        }
        pub fn allocated_device_bytes(&self) -> u64 {
            0
        }
        pub fn encode_from_device(
            &mut self,
            _: &GpuPlan,
            _: PackedUnit,
            _: &[u8; 32],
            _: &[DeviceSpan],
            _: &mut [u8],
            _: usize,
        ) -> Result<(PackedUnit, usize)> {
            Err(Error::new(ErrorKind::Unsupported, "GPU KV codec requires Linux"))
        }
        pub fn encode_from_device_vec(
            &mut self,
            _: &GpuPlan,
            _: PackedUnit,
            _: &[u8; 32],
            _: &[DeviceSpan],
            _: usize,
        ) -> Result<(PackedUnit, Vec<u8>)> {
            Err(Error::new(ErrorKind::Unsupported, "GPU KV codec requires Linux"))
        }
        pub fn decode_to_device(
            &mut self,
            _: &GpuPlan,
            _: &[&[u8]],
            _: &[DeviceSpan],
            _: u32,
            _: u32,
            _: usize,
        ) -> Result<()> {
            Err(Error::new(ErrorKind::Unsupported, "GPU KV codec requires Linux"))
        }
        pub fn copy_literal_to_device(
            &mut self,
            _: &GpuPlan,
            _: &[&[u8]],
            _: &[DeviceSpan],
            _: usize,
        ) -> Result<()> {
            Err(Error::new(ErrorKind::Unsupported, "GPU KV codec requires Linux"))
        }
    }
}

pub use imp::GpuRankCodec;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_plan_builds_coalesced_chunks_for_all_encodings() {
        use super::super::codec::Field;
        let fields = vec![
            Field {
                offset: 0,
                length: 65536 * 2,
                encoding: Encoding::Float32Matrix128,
            },
            Field {
                offset: 65536 * 2 + 4096,
                length: 135_168,
                encoding: Encoding::Latent528,
            },
            Field {
                offset: 65536 * 2 + 4096 + 135_168,
                length: 8_448,
                encoding: Encoding::Index132,
            },
            Field {
                offset: 65536 * 2 + 4096 + 135_168 + 8_448,
                length: 4096,
                encoding: Encoding::Bfloat16,
            },
        ];
        let total = 65536 * 2 + 4096 + 135_168 + 8_448 + 4096 + 512;
        let cpu_plan = Plan::new(total, &fields).unwrap();
        let gpu_plan = GpuPlan::from_plan(&cpu_plan).unwrap();
        assert_eq!(gpu_plan.original_bytes(), total);
        assert_eq!(gpu_plan.num_mats(), 2);
        assert!(gpu_plan.num_chunks() >= 5);
        assert!(gpu_plan.chunk_sizes.iter().all(|&s| s > 0 && s <= ANS_CHUNK));
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "exercises real CUDA + nvCOMP ANS GKV1 encode/decode on a GPU"]
    fn gpu_codec_round_trips_all_encodings_and_rejects_corruption() {
        use super::super::codec::Field;
        use super::super::packed::crc32;
        use std::ffi::{CString, c_int, c_void};
        let lib = unsafe {
            let p = CString::new("libcuda.so.1").unwrap();
            libc::dlopen(p.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL)
        };
        assert!(!lib.is_null(), "libcuda.so.1 required");
        let cu_init: unsafe extern "C" fn(u32) -> c_int = unsafe {
            std::mem::transmute(libc::dlsym(lib, c"cuInit".as_ptr()))
        };
        let cu_ctx_retain: unsafe extern "C" fn(*mut *mut c_void, c_int) -> c_int = unsafe {
            std::mem::transmute(libc::dlsym(lib, c"cuDevicePrimaryCtxRetain".as_ptr()))
        };
        let cu_ctx_set: unsafe extern "C" fn(*mut c_void) -> c_int = unsafe {
            std::mem::transmute(libc::dlsym(lib, c"cuCtxSetCurrent".as_ptr()))
        };
        let cu_ctx_release: unsafe extern "C" fn(c_int) -> c_int = unsafe {
            std::mem::transmute(libc::dlsym(lib, c"cuDevicePrimaryCtxRelease".as_ptr()))
        };
        let cu_stream_create: unsafe extern "C" fn(*mut *mut c_void, u32) -> c_int = unsafe {
            std::mem::transmute(libc::dlsym(lib, c"cuStreamCreate".as_ptr()))
        };
        let cu_stream_destroy: unsafe extern "C" fn(*mut c_void) -> c_int = unsafe {
            std::mem::transmute(libc::dlsym(lib, c"cuStreamDestroy_v2".as_ptr()))
        };
        let cu_mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> c_int = unsafe {
            std::mem::transmute(libc::dlsym(lib, c"cuMemAlloc_v2".as_ptr()))
        };
        let cu_mem_free: unsafe extern "C" fn(u64) -> c_int = unsafe {
            std::mem::transmute(libc::dlsym(lib, c"cuMemFree_v2".as_ptr()))
        };
        let cu_htod: unsafe extern "C" fn(u64, *const c_void, usize) -> c_int = unsafe {
            std::mem::transmute(libc::dlsym(lib, c"cuMemcpyHtoD_v2".as_ptr()))
        };
        let cu_dtoh: unsafe extern "C" fn(*mut c_void, u64, usize) -> c_int = unsafe {
            std::mem::transmute(libc::dlsym(lib, c"cuMemcpyDtoH_v2".as_ptr()))
        };

        let mut ctx: *mut c_void = std::ptr::null_mut();
        let mut stream: *mut c_void = std::ptr::null_mut();
        unsafe {
            assert_eq!(cu_init(0), 0);
            assert_eq!(cu_ctx_retain(&mut ctx, 0), 0);
            assert_eq!(cu_ctx_set(ctx), 0);
            assert_eq!(cu_stream_create(&mut stream, 0), 0);
        }

        // Build a synthetic compound unit exercising EVERY encoding + alignment padding!
        let mut fields = Vec::new();
        let mut pos = 0usize;
        for (enc, len) in [
            (Encoding::Float32Matrix128, MATRIX_BYTES * 2),
            (Encoding::Bfloat16, 8192),
            (Encoding::Convolution20, 20_000),
            (Encoding::Convolution32, 32_768),
            (Encoding::Latent528, 135_168),
            (Encoding::Index132, 8_448),
            (Encoding::Fp8Page512, 32_768),
            (Encoding::WindowFp4, 33_792),
            (Encoding::LatentFp4, 18_432),
            (Encoding::Bytes, 4096),
        ] {
            // Align to 4096 between fields to exercise padding intervals.
            pos = pos.div_ceil(4096) * 4096;
            fields.push(Field {
                offset: pos,
                length: len,
                encoding: enc,
            });
            pos += len;
        }
        let total = pos.div_ceil(4096) * 4096;
        let cpu_plan = Plan::new(total, &fields).unwrap();
        let gpu_plan = GpuPlan::from_plan(&cpu_plan).unwrap();

        // Fill realistic compressible data with special float32 values in matrices (NaN, -0, subnormals, rank-1).
        let mut raw = vec![0u8; total];
        for f in &fields {
            if f.encoding == Encoding::Float32Matrix128 {
                for m in 0..2 {
                    let base = f.offset + m * MATRIX_BYTES;
                    for c in 0..128 {
                        for r in 0..128 {
                            let v = if m == 0 {
                                // Smooth rank-2 matrix that compresses strongly under 64-pivot predictor.
                                let a = (r as f32 * 0.01).sin();
                                let b = (c as f32 * 0.02).cos();
                                (a * b + 0.001 * ((r ^ c) as f32)).to_bits()
                            } else {
                                // Edge cases: -0.0, subnormals, inf, NaN payload, normal floats.
                                match (r * 128 + c) % 7 {
                                    0 => 0x0000_0000,
                                    1 => 0x8000_0000,
                                    2 => 1u32 << (r % 23),
                                    3 => 0x7f80_0000,
                                    4 => 0x7fc0_0001 | ((c as u32) & 0xff),
                                    _ => (((r as i32 - 64) as f32) * 0.125).to_bits(),
                                }
                            };
                            let idx = base + (c * 128 + r) * 4;
                            raw[idx..idx + 4].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                }
            } else {
                for i in 0..f.length {
                    raw[f.offset + i] = ((i / 17 + (i % 4) * 3) % 19) as u8;
                }
            }
        }

        let mut d_src = 0u64;
        let mut d_dst = 0u64;
        unsafe {
            assert_eq!(cu_mem_alloc(&mut d_src, total), 0);
            assert_eq!(cu_mem_alloc(&mut d_dst, total), 0);
            assert_eq!(cu_htod(d_src, raw.as_ptr() as _, total), 0);
        }

        let cubin_path = std::env::var("FLARE_TEST_KV_CODEC_CUBIN")
            .expect("set FLARE_TEST_KV_CODEC_CUBIN to a compiled kv_codec_gpu.cubin for GPU test");
        let cubin = std::fs::read(&cubin_path).unwrap();
        let mut codec = GpuRankCodec::new(&gpu_plan, &gpu_plan, 1, &cubin, stream as usize).unwrap();
        let profile = [0x42u8; 32];
        let src_span = [DeviceSpan {
            device: d_src,
            offset: 0,
            length: total,
        }];
        let dst_span = [DeviceSpan {
            device: d_dst,
            offset: 0,
            length: total,
        }];

        let mut pkt = vec![0u8; gpu_plan.max_packet()];
        let (unit, used) = codec
            .encode_from_device(
                &gpu_plan,
                PackedUnit::default(),
                &profile,
                &src_span,
                &mut pkt,
                stream as usize,
            )
            .unwrap();
        assert_eq!(unit.representation, Representation::Packet as i32);
        assert!(used < total, "expected compression: used={used} total={total}");
        pkt.truncate(used);

        // Decode with fragmented slices and verify exact bit-for-bit equality!
        let frags: Vec<&[u8]> = pkt.chunks(7919).collect();
        codec
            .decode_to_device(
                &gpu_plan,
                &frags,
                &dst_span,
                unit.stored_crc,
                unit.original_crc,
                stream as usize,
            )
            .unwrap();
        let mut back = vec![0xa5u8; total];
        unsafe {
            assert_eq!(cu_dtoh(back.as_mut_ptr() as _, d_dst, total), 0);
        }
        assert_eq!(back, raw, "GPU GKV1 round-trip must be bit-exact across all encodings");

        // 1. Corrupted stored_crc must fail with Checksum::Stored.
        let err = codec
            .decode_to_device(
                &gpu_plan,
                &[&pkt],
                &dst_span,
                unit.stored_crc ^ 1,
                unit.original_crc,
                stream as usize,
            )
            .unwrap_err();
        assert!(matches!(
            err.get_ref().and_then(|c| c.downcast_ref::<Checksum>()),
            Some(Checksum::Stored)
        ));

        // 2. Corrupted original_crc must fail with Checksum::Original.
        let err = codec
            .decode_to_device(
                &gpu_plan,
                &[&pkt],
                &dst_span,
                unit.stored_crc,
                unit.original_crc ^ 1,
                stream as usize,
            )
            .unwrap_err();
        assert!(matches!(
            err.get_ref().and_then(|c| c.downcast_ref::<Checksum>()),
            Some(Checksum::Original)
        ));

        // 3. Truncated packet must fail cleanly.
        assert!(
            codec
                .decode_to_device(
                    &gpu_plan,
                    &[&pkt[..pkt.len() - 1]],
                    &dst_span,
                    crc32(&pkt[..pkt.len() - 1]),
                    unit.original_crc,
                    stream as usize,
                )
                .is_err()
        );

        // 4. Incompressible random bytes must fall back to Literal with raw_xxh3.
        let mut rng = 0x1234_5678_9abc_def0u64;
        let rand_raw: Vec<u8> = (0..total)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng >> 56) as u8
            })
            .collect();
        unsafe {
            assert_eq!(cu_htod(d_src, rand_raw.as_ptr() as _, total), 0);
        }
        let mut lit_buf = vec![0u8; gpu_plan.max_packet()];
        let (lit_unit, lit_used) = codec
            .encode_from_device(
                &gpu_plan,
                PackedUnit::default(),
                &profile,
                &src_span,
                &mut lit_buf,
                stream as usize,
            )
            .unwrap();
        assert_eq!(lit_unit.representation, Representation::Literal as i32);
        assert_eq!(lit_used, total);
        assert_eq!(&lit_buf[..total], &rand_raw[..]);
        assert_eq!(
            lit_unit.raw_xxh3,
            crate::record::checksum(&rand_raw)
        );

        unsafe {
            cu_mem_free(d_src);
            cu_mem_free(d_dst);
            cu_stream_destroy(stream);
            cu_ctx_release(0);
        }
    }
}
