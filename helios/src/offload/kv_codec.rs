// SPDX-License-Identifier: Apache-2.0

//! Lossless packets for one rank's share of a cache page or continuation row.

use rayon::prelude::*;
use std::io::{Error, ErrorKind, Result};
use std::sync::OnceLock;

const MAGIC: u32 = u32::from_le_bytes(*b"SGC1");
const HEADER: usize = 16;
const ENTRY: usize = 40;
const MAX_UNIT: usize = 256 << 20;
const CHUNK: usize = 65536;

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

/// Each field describes a contiguous interval of the padded rank share.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub offset: usize,
    pub length: usize,
    pub encoding: Encoding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Bytes,
    Bfloat16,
    Float32Matrix128,
    Convolution20,
    Convolution32,
    Latent528,
    Index132,
    Fp8Page512,
    WindowFp4,
    LatentFp4,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Frame {
    base: i32,
    full: i32,
    sub: i32,
    n: i32,
    kind: i32,
    rank: i32,
    layer: i32,
    page: i32,
    work: i32,
    arena: i32,
    reserved: i32,
}

impl Frame {
    fn shape(&self) -> [i32; 8] {
        [
            self.base, self.full, self.sub, self.n, self.kind, self.rank, self.layer, self.page,
        ]
    }

    fn raw_bytes(&self) -> usize {
        self.n as usize * if self.kind == 1 { 4 } else { 1 }
    }
}

unsafe extern "C" {
    fn kv_cpu_frame(
        frame: *const Frame,
        raw: *mut u8,
        packet: *mut u8,
        capacity: i32,
        scratch: *mut u32,
        decoding: bool,
    ) -> i32;
}

#[derive(Clone, Debug)]
pub struct Plan {
    frames: Vec<Frame>,
    pub original_bytes: usize,
    work: usize,
    arena: usize,
}

pub(crate) static MIRROR_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static ORIGINAL_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
const ENCODE_WORKERS: usize = 16;
const RESTORE_WORKERS: usize = 8;

fn frame_pool(restore: bool) -> Result<&'static rayon::ThreadPool> {
    static IN: OnceLock<Result<rayon::ThreadPool>> = OnceLock::new();
    static OUT: OnceLock<Result<rayon::ThreadPool>> = OnceLock::new();
    let (pool, limit, name) = if restore {
        (&OUT, RESTORE_WORKERS, "restore")
    } else {
        (&IN, ENCODE_WORKERS, "mirror")
    };
    pool.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(limit)
            .thread_name(move |i| format!("kv-{name}-{i}"))
            // Mirror encodes run at the rank workers' lower priority so they never compete
            // with request handling; restores keep normal priority because a request waits on them.
            .start_handler(move |_| {
                if !restore {
                    crate::os::lower_priority(crate::offload::transfer::NICE);
                }
            })
            .build()
            .map_err(Error::other)
    })
    .as_ref()
    .map_err(Error::other)
}

impl Plan {
    /// Padding and untyped intervals are included byte for byte.
    pub fn new(original_bytes: usize, fields: &[Field]) -> Result<Self> {
        if original_bytes == 0 || original_bytes > MAX_UNIT {
            return Err(invalid("codec unit size outside supported range"));
        }
        let mut plan = Self {
            frames: vec![],
            original_bytes,
            work: 0,
            arena: 0,
        };
        let mut end = 0;
        for field in fields {
            if field.offset < end
                || field.length == 0
                || field
                    .offset
                    .checked_add(field.length)
                    .is_none_or(|n| n > original_bytes)
            {
                return Err(invalid("invalid codec field extent"));
            }
            plan.bytes(end, field.offset - end)?;
            plan.field(*field)?;
            end = field.offset + field.length;
        }
        plan.bytes(end, original_bytes - end)?;
        Ok(plan)
    }

    fn frame(&mut self, base: usize, full: usize, sub: usize, n: usize, kind: i32) -> Result<()> {
        if self.frames.len() >= 65536 {
            return Err(invalid("too many codec frames"));
        }
        let frame = Frame {
            base: base as i32,
            full: full as i32,
            sub: sub as i32,
            n: n as i32,
            kind,
            rank: 0,
            layer: 0,
            page: 0,
            work: self.work as i32,
            arena: self.arena as i32,
            reserved: 0,
        };
        let aligned = frame.raw_bytes().next_multiple_of(4);
        self.work += aligned;
        self.arena += aligned + 4;
        self.frames.push(frame);
        Ok(())
    }

    fn bytes(&mut self, offset: usize, size: usize) -> Result<()> {
        for start in (0..size).step_by(CHUNK) {
            let n = (size - start).min(CHUNK);
            self.frame(offset + start, n, 0, n, 0)?;
        }
        Ok(())
    }

    fn field(&mut self, field: Field) -> Result<()> {
        use Encoding::*;
        let Field {
            offset,
            length,
            encoding,
        } = field;
        let multiple = |size| {
            if length.is_multiple_of(size) {
                Ok(())
            } else {
                Err(invalid("field length disagrees with codec geometry"))
            }
        };
        match encoding {
            Bytes => self.bytes(offset, length)?,
            Float32Matrix128 => {
                multiple(CHUNK)?;
                if offset % 4 != 0 {
                    return Err(invalid("unaligned codec matrix"));
                }
                for sub in (0..length).step_by(CHUNK) {
                    self.frame(offset, length, sub, 16384, 1)?;
                }
            }
            Bfloat16 | Convolution20 | Convolution32 => {
                let width = match encoding {
                    Bfloat16 => 2,
                    Convolution20 => 20,
                    _ => 32,
                };
                multiple(width)?;
                let rows = length / width;
                for lane in 0..width {
                    for sub in (0..rows).step_by(32768) {
                        self.frame(
                            offset,
                            length,
                            lane * rows + sub,
                            (rows - sub).min(32768),
                            width as i32,
                        )?;
                    }
                }
            }
            Latent528 | Index132 | Fp8Page512 => {
                let (stride, n, kind) = match encoding {
                    Latent528 => (135168, 131072, 40),
                    Index132 => (8448, 8192, 41),
                    _ => (32768, 32768, 42),
                };
                multiple(stride)?;
                for sub in (0..length).step_by(stride) {
                    self.frame(offset, length, sub, n, kind)?;
                    if kind != 42 {
                        let count = n / 128;
                        for lane in 0..4 {
                            self.frame(offset, length, sub + lane * count, count, kind + 3)?;
                        }
                    }
                }
            }
            WindowFp4 | LatentFp4 => {
                let (stride, n, kind) = if encoding == WindowFp4 {
                    (33792, 32768, 45)
                } else {
                    (18432, 16384, 47)
                };
                multiple(stride)?;
                for base in (0..length).step_by(stride) {
                    self.frame(offset + base, stride, 0, n, kind)?;
                    self.frame(offset + base, stride, 0, stride - n, kind + 1)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn metadata(&self) -> usize {
        HEADER + self.frames.len() * ENTRY
    }
    fn max_packet(&self) -> usize {
        self.metadata() + self.arena
    }

    fn entries(&self, packet: &[u8], total: usize) -> Result<Vec<std::ops::Range<usize>>> {
        let word = |p| u32::from_le_bytes(packet[p..p + 4].try_into().unwrap());
        if packet.len() < self.metadata()
            || total < self.metadata()
            || total > self.max_packet()
            || word(0) != MAGIC
            || word(4) != 1
            || word(8) as usize != self.frames.len()
            || word(12) as usize != self.original_bytes
        {
            return Err(invalid("invalid codec packet header"));
        }
        let mut pos = self.metadata();
        let mut ranges = Vec::with_capacity(self.frames.len());
        for (frame, entry) in self
            .frames
            .iter()
            .zip(packet[HEADER..self.metadata()].chunks_exact(ENTRY))
        {
            let mut values = entry
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
            for expected in frame.shape() {
                if values.next() != Some(expected as u32) {
                    return Err(invalid("codec frame differs from prepared layout"));
                }
            }
            let start = values.next().unwrap() as usize;
            let len = values.next().unwrap() as usize;
            if start != pos || !(4..=frame.raw_bytes() + 4).contains(&len) || len > total - pos {
                return Err(invalid("invalid codec frame payload extent"));
            }
            ranges.push(pos..pos + len);
            pos += len;
        }
        if pos != total {
            return Err(invalid("trailing codec packet bytes"));
        }
        Ok(ranges)
    }

    pub fn encode(&self, source: &[u8]) -> Result<Vec<u8>> {
        if source.len() != self.original_bytes {
            return Err(invalid("codec input size mismatch"));
        }
        let mut out = vec![0; self.max_packet()];
        for (chunk, word) in out[..HEADER].chunks_exact_mut(4).zip([
            MAGIC,
            1,
            self.frames.len() as u32,
            source.len() as u32,
        ]) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        let mut pos = self.metadata();
        let base = out.as_mut_ptr() as usize + self.metadata();
        let sizes: Vec<usize> = frame_pool(false)?.install(|| {
            self.frames
                .par_iter()
                .map_init(Vec::<u32>::new, |scratch, frame| {
                    let cap = frame.raw_bytes() + 4;
                    scratch.resize(frame.raw_bytes().div_ceil(4), 0);
                    // Plan assigns each frame a disjoint maximum-sized output range.
                    let n = unsafe {
                        kv_cpu_frame(
                            frame,
                            source.as_ptr().cast_mut(),
                            (base + frame.arena as usize) as *mut u8,
                            cap as i32,
                            scratch.as_mut_ptr(),
                            false,
                        )
                    };
                    if n < 4 || n as usize > cap {
                        Err(invalid("codec encode failed"))
                    } else {
                        Ok(n as usize)
                    }
                })
                .collect::<Result<_>>()
        })?;
        for (i, (frame, n)) in self.frames.iter().zip(sizes).enumerate() {
            let start = self.metadata() + frame.arena as usize;
            out.copy_within(start..start + n, pos);
            let dir = &mut out[HEADER + i * ENTRY..HEADER + (i + 1) * ENTRY];
            for (dst, word) in dir.chunks_exact_mut(4).zip(
                frame
                    .shape()
                    .map(|v| v as u32)
                    .into_iter()
                    .chain([pos as u32, n as u32]),
            ) {
                dst.copy_from_slice(&word.to_le_bytes());
            }
            pos += n as usize;
        }
        out.truncate(pos);
        Ok(out)
    }

    pub fn decode(&self, packet: &[u8], dst: &mut [u8]) -> Result<()> {
        self.decode_in(packet, dst, true)
    }

    fn decode_in(&self, packet: &[u8], dst: &mut [u8], restore: bool) -> Result<()> {
        self.decode_chunks(&[packet], dst, restore)
    }

    pub(super) fn decode_chunks(
        &self,
        chunks: &[&[u8]],
        dst: &mut [u8],
        restore: bool,
    ) -> Result<()> {
        if dst.len() != self.original_bytes {
            return Err(invalid("codec destination size mismatch"));
        }
        let input = Source::new(chunks);
        let ranges = self.entries(&input.range(0..self.metadata())?, input.len)?;
        let base = dst.as_mut_ptr() as usize;
        frame_pool(restore)?.install(|| {
            self.frames.par_iter().zip(ranges).try_for_each_init(
                Vec::<u32>::new,
                |scratch, (frame, range)| {
                    let input = input.range(range)?;
                    scratch.resize(frame.raw_bytes().div_ceil(4), 0);
                    // Prepared frames partition output bytes, including transposed fields and padding.
                    let n = unsafe {
                        kv_cpu_frame(
                            frame,
                            base as *mut u8,
                            input.as_ptr().cast_mut(),
                            input.len() as i32,
                            scratch.as_mut_ptr(),
                            true,
                        )
                    };
                    if n < 0 {
                        Err(invalid("invalid entropy data in codec frame"))
                    } else {
                        Ok(())
                    }
                },
            )
        })
    }

    pub fn encode_verified(&self, source: &[u8], verification: &mut [u8]) -> Result<Vec<u8>> {
        let packet = self.encode(source)?;
        self.decode_in(&packet, verification, false)?;
        let exact = frame_pool(false)?.install(|| {
            source
                .par_chunks(CHUNK)
                .zip(verification.par_chunks(CHUNK))
                .all(|(a, b)| a == b)
        });
        if !exact {
            MIRROR_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(invalid("codec round-trip verification failed"));
        }
        Ok(packet)
    }
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
    arena: i32,
    wire: i32,
}

impl Plan {
    fn shape(&self) -> Shape {
        Shape {
            frames: self.frames.len() as i32,
            raw: self.original_bytes as i32,
            work: self.work as i32,
            arena: self.arena as i32,
            wire: self.max_packet() as i32,
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
        arena: i32,
        wire: i32,
        operation: *mut *const std::ffi::c_char,
        code: *mut i32,
        bytes: *mut usize,
    ) -> *mut std::ffi::c_void;
    fn kv_gpu_close(work: *mut std::ffi::c_void);
    fn kv_gpu_transform_ns(work: *mut std::ffi::c_void) -> u64;
    fn kv_gpu_capture(
        work: *mut std::ffi::c_void,
        shape: Shape,
        frames: *const Frame,
        spans: *const DeviceSpan,
        count: i32,
    ) -> i32;
    fn kv_gpu_encode_captured(
        work: *mut std::ffi::c_void,
        shape: Shape,
        out: *mut u8,
        capacity: i32,
        stored_crc: *mut u32,
        original_crc: *mut u32,
    ) -> i32;
    fn kv_gpu_decode_spans(
        work: *mut std::ffi::c_void,
        shape: Shape,
        frames: *const Frame,
        chunks: *const HostSpan,
        nchunks: i32,
        len: i32,
        crc: u32,
        spans: *const DeviceSpan,
        count: i32,
    ) -> i32;
}

struct Source<'a> {
    chunks: &'a [&'a [u8]],
    ends: Vec<usize>,
    len: usize,
}

impl<'a> Source<'a> {
    fn new(chunks: &'a [&'a [u8]]) -> Self {
        let mut len = 0;
        let ends = chunks
            .iter()
            .map(|s| {
                len += s.len();
                len
            })
            .collect();
        Self { chunks, ends, len }
    }
    fn range(&self, range: std::ops::Range<usize>) -> Result<std::borrow::Cow<'a, [u8]>> {
        if range.end > self.len || range.is_empty() {
            return Err(invalid("short codec packet"));
        }
        let i = self.ends.partition_point(|&end| end <= range.start);
        let start = range.start - (self.ends[i] - self.chunks[i].len());
        let first = &self.chunks[i][start..];
        if first.len() >= range.len() {
            return Ok((&first[..range.len()]).into());
        }
        // Only a frame crossing a slab boundary needs local scratch.
        let mut frame = Vec::with_capacity(range.len());
        frame.extend_from_slice(first);
        for chunk in &self.chunks[i + 1..] {
            let n = chunk.len().min(range.len() - frame.len());
            frame.extend_from_slice(&chunk[..n]);
            if frame.len() == range.len() {
                break;
            }
        }
        Ok(frame.into())
    }
}

#[cfg(kv_codec_cuda)]
#[repr(C)]
struct HostSpan {
    data: *const u8,
    length: usize,
}

pub struct GpuCodec {
    #[cfg(kv_codec_cuda)]
    workspace: std::ptr::NonNull<std::ffi::c_void>,
    pub reserved: u64,
}

impl GpuCodec {
    fn limits(plans: &[&Plan]) -> [i32; 5] {
        let mut limits = [0; 5];
        for p in plans {
            let s = p.shape();
            for (limit, v) in limits
                .iter_mut()
                .zip([s.raw, s.frames, s.work, s.arena, s.wire])
            {
                *limit = (*limit).max(v);
            }
        }
        limits
    }
    #[cfg(test)]
    pub(crate) fn workspace_parts(plans: &[&Plan]) -> [u64; 10] {
        let [raw, nf, work, arena, wire] = Self::limits(plans).map(|x| x as u64);
        [44 * nf, raw, raw, work, arena, nf, wire, 4 * nf, 4 * nf, 12]
            .map(|n| n.next_multiple_of(256))
    }
    pub(super) fn transform_ns(&self) -> u64 {
        #[cfg(kv_codec_cuda)]
        {
            unsafe { kv_gpu_transform_ns(self.workspace.as_ptr()) }
        }
        #[cfg(not(kv_codec_cuda))]
        {
            0
        }
    }
    pub fn open(model_context: usize, plans: &[&Plan]) -> Result<Self> {
        #[cfg(kv_codec_cuda)]
        {
            let mut current = 0;
            // The native codec creates its mirror and restore streams in the current context.
            if unsafe { cuCtxGetCurrent(&mut current) } != 0
                || current == 0
                || current != model_context
            {
                return Err(invalid("GPU codec requires the current model context"));
            }
            let [raw, nf, work, arena, wire] = Self::limits(plans);
            let mut operation = std::ptr::null();
            let mut code = 0;
            let mut reserved = 0usize;
            let workspace = std::ptr::NonNull::new(unsafe {
                kv_gpu_open(raw, nf, work, arena, wire, &mut operation, &mut code, &mut reserved)
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

    /// The caller owns each immutable device source until this synchronous call returns.
    pub unsafe fn encode(
        &mut self,
        plan: &Plan,
        spans: &[DeviceSpan],
        out: &mut [u8],
    ) -> Result<Option<(usize, u32, u32)>> {
        unsafe { self.encode_guarded(plan, spans, out, || Ok(())) }
    }

    pub(super) unsafe fn encode_guarded<G>(
        &mut self,
        plan: &Plan,
        spans: &[DeviceSpan],
        out: &mut [u8],
        acquire: impl FnOnce() -> Result<G>,
    ) -> Result<Option<(usize, u32, u32)>> {
        #[cfg(kv_codec_cuda)]
        {
            let (mut stored, mut original) = (0, 0);
            {
                let _source = acquire()?;
                let rc = unsafe {
                    kv_gpu_capture(
                        self.workspace.as_ptr(),
                        plan.shape(),
                        plan.frames.as_ptr(),
                        spans.as_ptr(),
                        i32::try_from(spans.len()).map_err(|_| invalid("too many device spans"))?,
                    )
                };
                if rc != 0 {
                    return Err(invalid("GPU mirror source copy failed"));
                }
            }
            let n = unsafe {
                kv_gpu_encode_captured(
                    self.workspace.as_ptr(),
                    plan.shape(),
                    out.as_mut_ptr(),
                    i32::try_from(out.len()).map_err(|_| invalid("packet output is too large"))?,
                    &mut stored,
                    &mut original,
                )
            };
            match n {
                0 => Ok(None),
                1.. => Ok(Some((n as usize, stored, original))),
                -2 => {
                    MIRROR_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Err(invalid("GPU mirror verification failed"))
                }
                _ => Err(invalid("GPU encode failed")),
            }
        }
        #[cfg(not(kv_codec_cuda))]
        {
            let _ = (plan, spans, out, acquire);
            Err(invalid("GPU KV codec requires the CUDA build"))
        }
    }

    /// The caller holds the destination device ranges exclusively until the copy completes.
    pub unsafe fn decode(
        &mut self,
        plan: &Plan,
        packet: &[u8],
        stored: u32,
        original: u32,
        spans: &[DeviceSpan],
    ) -> Result<()> {
        unsafe { self.decode_chunks(plan, &[packet], stored, original, spans) }
    }

    pub(super) unsafe fn decode_chunks(
        &mut self,
        plan: &Plan,
        chunks: &[&[u8]],
        stored: u32,
        original: u32,
        spans: &[DeviceSpan],
    ) -> Result<()> {
        if flare_selene::shared::packed::crc32_chunks(chunks) != stored {
            return Err(invalid("stored codec checksum mismatch"));
        }
        let src = Source::new(chunks);
        plan.entries(&src.range(0..plan.metadata())?, src.len)?;
        #[cfg(kv_codec_cuda)]
        {
            let host: Vec<_> = chunks
                .iter()
                .map(|c| HostSpan {
                    data: c.as_ptr(),
                    length: c.len(),
                })
                .collect();
            let rc = unsafe {
                kv_gpu_decode_spans(
                    self.workspace.as_ptr(),
                    plan.shape(),
                    plan.frames.as_ptr(),
                    host.as_ptr(),
                    host.len() as i32,
                    src.len as i32,
                    original,
                    spans.as_ptr(),
                    i32::try_from(spans.len()).map_err(|_| invalid("too many device spans"))?,
                )
            };
            if rc == -2 {
                ORIGINAL_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(invalid("GPU original checksum failed"));
            }
            if rc != 0 {
                return Err(invalid("GPU decode failed"));
            }
            Ok(())
        }
        #[cfg(not(kv_codec_cuda))]
        {
            let _ = (original, spans);
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
    #[test]
    fn decode_completes_with_every_encode_worker_busy() {
        use std::sync::{Arc, Condvar, Mutex, mpsc};
        use std::time::Duration;
        let plan = Plan::new(1 << 18, &[]).unwrap();
        let source = vec![47; 1 << 18];
        let packet = plan.encode(&source).unwrap();
        let pool = frame_pool(false).unwrap();
        let workers = pool.current_num_threads();
        assert!(workers > 1);
        assert!(frame_pool(true).unwrap().current_num_threads() > 1);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (entered, ready) = mpsc::channel();
        let blocked = gate.clone();
        pool.spawn_broadcast(move |_| {
            let (m, cv) = &*blocked;
            let closed = m.lock().unwrap();
            entered.send(()).unwrap();
            let _guard = cv
                .wait_timeout_while(closed, Duration::from_secs(10), |open| !*open)
                .unwrap();
        });
        for _ in 0..workers {
            ready.recv_timeout(Duration::from_secs(3)).unwrap();
        }
        let (tx, rx) = mpsc::channel();
        let decode = std::thread::spawn(move || {
            let mut dst = vec![0; source.len()];
            plan.decode(&packet, &mut dst).unwrap();
            tx.send(source == dst).unwrap();
        });
        let restored = rx.recv_timeout(Duration::from_secs(2));
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        decode.join().unwrap();
        assert_eq!(restored, Ok(true));
    }

    #[test]
    fn canonical_packets_round_trip() {
        let fixtures: [&[u8]; 3] = [
            include_bytes!("../../../selene/tests/fixtures/codec/page0.sgc"),
            include_bytes!("../../../selene/tests/fixtures/codec/page1.sgc"),
            include_bytes!("../../../selene/tests/fixtures/codec/row.sgc"),
        ];
        for (i, expected) in fixtures.into_iter().enumerate() {
            let source: Vec<u8> = (0..4096)
                .map(|j| ((j / (8 + 2 * i) + i) % (2 + i)) as u8)
                .collect();
            let mut back = vec![255; 4096];
            let plan = Plan::new(source.len(), &[]).unwrap();
            let bytes = plan.encode_verified(&source, &mut back).unwrap();
            assert_eq!(bytes, expected);
            assert_eq!(source, back);
            let mut damaged = bytes.clone();
            damaged[16] ^= 1;
            assert!(plan.decode(&damaged, &mut back).is_err());
        }
    }

    #[test]
    fn typed_units_preserve_bits_including_padding() {
        for (kind, size) in [
            (Encoding::Float32Matrix128, 65536),
            (Encoding::Latent528, 135168),
            (Encoding::Index132, 8448),
            (Encoding::Fp8Page512, 32768),
            (Encoding::Convolution20, 20000),
            (Encoding::Bfloat16, 6144),
            (Encoding::WindowFp4, 33792),
            (Encoding::LatentFp4, 18432),
        ] {
            let mut state = 2026u32;
            let mut raw: Vec<u8> = (0..size + 16)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 24) as u8
                })
                .collect();
            raw[4..20]
                .copy_from_slice(&[0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 128, 127, 37, 2, 192, 127]);
            let plan = Plan::new(
                raw.len(),
                &[Field {
                    offset: 4,
                    length: size,
                    encoding: kind,
                }],
            )
            .unwrap();
            let mut dst = vec![1; raw.len()];
            plan.encode_verified(&raw, &mut dst).unwrap();
            assert_eq!(raw, dst);
        }
    }
}
