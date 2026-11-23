// SPDX-License-Identifier: Apache-2.0

//! Lossless packets for one rank's share of a cache page or continuation row.

use std::io::{Error, ErrorKind, Result};
use std::sync::OnceLock;

pub use flare_selene::shared::codec::{
    Frame, Plan, ORIGINAL_FAILURES,
};

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

pub(super) fn restore_pool() -> flare_selene::Result<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Result<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .thread_name(|i| format!("kv-restore-{i}"))
            .build()
            .map_err(Error::other)
    })
    .as_ref()
    .map_err(|e| flare_selene::geometry::invalid(e.to_string()))
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DeviceSpan {
    pub device: u64,
    pub offset: i32,
    pub length: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Shape {
    frames: i32,
    raw: i32,
    work: i32,
    wire: i32,
}

impl Shape {
    fn from_plan(plan: &Plan) -> Shape {
        Shape {
            frames: plan.frames().len() as i32,
            raw: plan.original_bytes() as i32,
            work: plan.workspace_bytes() as i32,
            wire: plan.max_packet() as i32,
        }
    }
}

#[cfg(kv_codec_cuda)]
unsafe extern "C" {
    fn cuCtxGetCurrent(context: *mut usize) -> i32;
    fn kv_gpu_open(
        raw: i32,
        nf: i32,
        work: i32,
        wire: i32,
        operation: *mut *const std::ffi::c_char,
        code: *mut i32,
        bytes: *mut usize,
    ) -> *mut std::ffi::c_void;
    fn kv_gpu_close(work: *mut std::ffi::c_void);
    fn kv_gpu_decode_capacity(work: *const std::ffi::c_void, shape: Shape) -> i32;
    fn kv_gpu_decode_spans(
        work: *mut std::ffi::c_void,
        shape: Shape,
        frames: *const Frame,
        units: *const NativeRestoreUnit,
        count: i32,
        transform_ns: *mut u64,
    ) -> i32;
}

#[cfg(kv_codec_cuda)]
#[repr(C)]
struct HostSpan {
    data: *const u8,
    length: usize,
}

#[cfg(kv_codec_cuda)]
#[repr(C)]
struct NativeRestoreUnit {
    chunks: *const HostSpan,
    nchunks: i32,
    length: i32,
    stored: u32,
    original: u32,
    spans: *const DeviceSpan,
    count: i32,
}

pub(super) struct RestoreUnit<'a> {
    pub chunks: Vec<&'a [u8]>,
    pub stored: u32,
    pub original: u32,
    pub spans: Vec<DeviceSpan>,
}

pub struct GpuCodec {
    #[cfg(kv_codec_cuda)]
    workspace: std::ptr::NonNull<std::ffi::c_void>,
    pub reserved: u64,
}

impl GpuCodec {
    fn limits(plans: &[&Plan]) -> [i32; 4] {
        let mut limits = [0; 4];
        for p in plans {
            let s = Shape::from_plan(p);
            for (limit, v) in limits
                .iter_mut()
                .zip([s.raw, s.frames, s.work, s.wire])
            {
                *limit = (*limit).max(v);
            }
        }
        limits
    }
    #[cfg(test)]
    pub(crate) fn workspace_parts(plans: &[&Plan]) -> [u64; 6] {
        let [raw, nf, work, wire] = Self::limits(plans).map(|x| x as u64);
        [44 * nf, raw, work, wire, 4 * nf, 8].map(|n| n.next_multiple_of(256))
    }
    pub fn open(model_context: usize, plans: &[&Plan]) -> Result<Self> {
        #[cfg(kv_codec_cuda)]
        {
            let mut current = 0;
            // The native codec creates its upload and restore streams in the current context.
            if unsafe { cuCtxGetCurrent(&mut current) } != 0
                || current == 0
                || current != model_context
            {
                return Err(invalid("GPU codec requires the current model context"));
            }
            let [raw, nf, work, wire] = Self::limits(plans);
            let mut operation = std::ptr::null();
            let mut code = 0;
            let mut reserved = 0usize;
            let workspace = std::ptr::NonNull::new(unsafe {
                kv_gpu_open(raw, nf, work, wire, &mut operation, &mut code, &mut reserved)
            })
            .ok_or_else(|| {
                // The native opener returns a static operation name on every failure.
                let stage = unsafe { std::ffi::CStr::from_ptr(operation) }.to_string_lossy();
                Error::new(ErrorKind::Other, format!("cannot reserve GPU codec workspace ({reserved} bytes): {stage}, CUDA error {code}"))
            })?;
            Ok(Self {
                reserved: reserved as u64,
                workspace,
            })
        }
        #[cfg(not(kv_codec_cuda))]
        {
            let _ = (model_context, plans);
            Err(invalid("GPU KV codec requires the CUDA build"))
        }
    }

    pub(super) unsafe fn decode_units(
        &mut self,
        plan: &Plan,
        units: &[RestoreUnit<'_>],
    ) -> Result<u64> {
        #[cfg(kv_codec_cuda)]
        {
            let shape = Shape::from_plan(plan);
            let capacity =
                unsafe { kv_gpu_decode_capacity(self.workspace.as_ptr(), shape) } as usize;
            let mut ns = 0;
            for batch in units.chunks(capacity) {
                let mut host_spans = Vec::with_capacity(batch.len());
                let mut native = Vec::with_capacity(batch.len());
                for unit in batch {
                    let length = plan.validate_chunks(&unit.chunks).map_err(|error| {
                        if flare_selene::shared::packed::crc32_chunks(&unit.chunks) == unit.stored {
                            error
                        } else {
                            invalid("stored codec checksum mismatch")
                        }
                    })?;
                    let host: Vec<_> = unit
                        .chunks
                        .iter()
                        .map(|c| HostSpan {
                            data: c.as_ptr(),
                            length: c.len(),
                        })
                        .collect();
                    native.push(NativeRestoreUnit {
                        chunks: host.as_ptr(),
                        nchunks: i32::try_from(host.len())
                            .map_err(|_| invalid("too many host spans"))?,
                        length: length as i32,
                        stored: unit.stored,
                        original: unit.original,
                        spans: unit.spans.as_ptr(),
                        count: i32::try_from(unit.spans.len())
                            .map_err(|_| invalid("too many device spans"))?,
                    });
                    host_spans.push(host);
                }
                // The native fence completes all reads and destination writes before these spans drop.
                let mut elapsed = 0;
                let rc = unsafe {
                    kv_gpu_decode_spans(
                        self.workspace.as_ptr(),
                        shape,
                        plan.frames().as_ptr(),
                        native.as_ptr(),
                        native.len() as i32,
                        &mut elapsed,
                    )
                };
                if rc == -3 {
                    flare_selene::shared::packed::STORED_FAILURES
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(invalid("stored codec checksum mismatch"));
                }
                if rc == -2 {
                    ORIGINAL_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(invalid("GPU original checksum failed"));
                }
                if rc != 0 {
                    return Err(invalid("GPU decode failed"));
                }
                ns += elapsed;
            }
            Ok(ns)
        }
        #[cfg(not(kv_codec_cuda))]
        {
            let _ = (plan, units);
            Err(invalid("GPU KV codec requires the CUDA build"))
        }
    }
}

#[cfg(kv_codec_cuda)]
impl Drop for GpuCodec {
    fn drop(&mut self) {
        unsafe { kv_gpu_close(self.workspace.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flare_selene::shared::packed::crc32;

    #[test]
    fn cpu_restores_a_verified_packet() {
        let plan = Plan::new(1 << 18, &[]).unwrap();
        let source = vec![47; 1 << 18];
        let packet = plan
            .encode_verified(&source, &mut vec![0; source.len()], None)
            .unwrap();
        let mut dst = vec![0; source.len()];
        plan.decode_checked(&[&packet], &mut dst, crc32(&packet), crc32(&source), Some(restore_pool().unwrap())).unwrap();
        assert_eq!(dst, source);
    }
}
