// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::cuda::PinnedHost;
use flare_selene::{
    codec_copy::{DecodeCounters, DecodeStats, Window},
    offload::KvCompression,
    shared::{
        codec::{Checksum, Plan},
        gpu_codec::{DeviceSpan, GpuPlan, GpuRankCodec},
        packed,
        wire::{
            CodecLayout, PackedUnit,
            packed_unit::{Kind, Representation},
        },
    },
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub(super) type SharedGpuCodecs = Arc<Mutex<HashMap<(usize, usize), Arc<Mutex<GpuRankCodec>>>>>;

fn cpu_pool() -> flare_selene::Result<&'static rayon::ThreadPool> {
    static POOL: OnceLock<std::io::Result<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = (std::thread::available_parallelism().map_or(4, usize::from) / 2)
            .clamp(1, 16);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("kv-codec-{i}"))
            .build()
            .map_err(std::io::Error::other)
    })
    .as_ref()
    .map_err(|e| flare_selene::geometry::invalid(e.to_string()))
}

struct CpuCodec {
    page: Plan,
    row: Plan,
    staging: PinnedHost,
    staging_alias: u64,
    encode_buf: Vec<u8>,
}

struct GpuCodec {
    page: GpuPlan,
    row: GpuPlan,
    codec: Arc<Mutex<GpuRankCodec>>,
}

pub(super) struct RankCodec {
    counters: DecodeCounters,
    write_mode: KvCompression,
    cpu_profile: [u8; 32],
    gpu_profile: [u8; 32],
    cpu: CpuCodec,
    gpu: Option<GpuCodec>,
}

impl RankCodec {
    fn original_bytes(&self, row: bool) -> usize {
        if row {
            self.cpu.row.original_bytes()
        } else {
            self.cpu.page.original_bytes()
        }
    }

    pub(super) fn new(
        rank: usize,
        context: usize,
        stream: &Stream,
        geometry: &CacheGeometry,
        counters: DecodeCounters,
        gpu_cubin: Option<&[u8]>,
        shared_gpu: Option<&SharedGpuCodecs>,
    ) -> Result<Self> {
        let cpu_layout = CodecLayout::for_revision(
            geometry,
            flare_selene::shared::codec_layout::REVISION_CPU,
        );
        let ([page, row], cpu_profile) = cpu_layout
            .plans_and_profile()
            .map_err(|e| invalid(e.to_string()))?;
        let gpu_profile = cpu_layout
            .with_revision(flare_selene::shared::codec_layout::REVISION_GPU)
            .profile()
            .map_err(|e| invalid(e.to_string()))?;

        // Both Cpu and Gpu modes keep one CPU pinned staging buffer so Gpu mode can
        // transparently decode existing ZKV1 checkpoints and Literal shares.
        let bytes = page.original_bytes().max(row.original_bytes());
        let staging = stream.mapped_host_buffer(bytes)?;
        let staging_alias = stream.mapped_address(&staging)? as u64;
        let encode_buf = if geometry.placement == KvCompression::Cpu {
            vec![0; page.max_packet().max(row.max_packet())]
        } else {
            Vec::new()
        };

        // Only GPU mode loads nvCOMP and the embedded GPU codec cubin and allocates the
        // shared per-rank GPU workspace; CPU mode allocates no GPU VRAM.
        let gpu = match geometry.placement {
            KvCompression::Gpu => {
                let cubin = gpu_cubin.ok_or_else(|| {
                    invalid(
                        "GPU checkpoint compression requested, but this core was exported without embedded GPU codec kernels; use --lossless-kv-compression cpu or re-export the core",
                    )
                })?;
                let page_gpu = GpuPlan::from_plan(&page).map_err(|e| invalid(e.to_string()))?;
                let row_gpu = GpuPlan::from_plan(&row).map_err(|e| invalid(e.to_string()))?;
                let max_spans = geometry.planes.len().max(1);
                let codec = if let Some(shared) = shared_gpu {
                    let mut map = shared.lock().unwrap();
                    if let Some(existing) = map.get(&(context, rank)) {
                        existing.clone()
                    } else {
                        let created = Arc::new(Mutex::new(
                            GpuRankCodec::new(&page_gpu, &row_gpu, max_spans, cubin, stream.raw())
                                .map_err(|e| invalid(e.to_string()))?,
                        ));
                        map.insert((context, rank), created.clone());
                        created
                    }
                } else {
                    Arc::new(Mutex::new(
                        GpuRankCodec::new(&page_gpu, &row_gpu, max_spans, cubin, stream.raw())
                            .map_err(|e| invalid(e.to_string()))?,
                    ))
                };
                Some(GpuCodec {
                    page: page_gpu,
                    row: row_gpu,
                    codec,
                })
            }
            KvCompression::Cpu | KvCompression::Off => None,
        };

        Ok(Self {
            counters,
            write_mode: geometry.placement,
            cpu_profile,
            gpu_profile,
            cpu: CpuCodec {
                page,
                row,
                staging,
                staging_alias,
                encode_buf,
            },
            gpu,
        })
    }

    pub(super) fn run(
        &mut self,
        rank: usize,
        geometry: &CacheGeometry,
        layout: &RankLayout,
        stream: &Stream,
        event: &Event,
        job: &Job,
        window: &Window,
    ) -> Result<u64> {
        let mut stats = DecodeStats::default();
        let result = match job.direction {
            Direction::In => {
                self.restore_window(rank, geometry, layout, stream, event, job, window, &mut stats)
            }
            Direction::Out => {
                self.capture_window(rank, geometry, layout, stream, event, job, window, &mut stats)
            }
        };
        {
            let mut totals = self.counters.lock().unwrap();
            totals.transform_ns += stats.transform_ns;
            totals.stored_bytes += stats.stored_bytes;
            match &result {
                Ok(_) => {
                    totals.original_bytes +=
                        self.original_bytes(false) as u64 * job.pages.len() as u64;
                    if job.row.is_some() {
                        totals.original_bytes += self.original_bytes(true) as u64;
                    }
                }
                Err(error) => {
                    totals.failed_windows += 1;
                    if let Some(cause) = error.carried() {
                        match cause.downcast_ref::<Checksum>() {
                            Some(Checksum::Stored) => totals.stored_failures += 1,
                            Some(Checksum::Original) => totals.original_failures += 1,
                            None => {}
                        }
                    }
                }
            }
        }
        if result.is_err() {
            stream.synchronize()?;
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_window(
        &mut self,
        rank: usize,
        geometry: &CacheGeometry,
        layout: &RankLayout,
        stream: &Stream,
        event: &Event,
        job: &Job,
        window: &Window,
        stats: &mut DecodeStats,
    ) -> Result<u64> {
        let capture = window
            .capture
            .as_ref()
            .ok_or_else(|| invalid("capture job missing capture target"))?;
        reject_duplicate_pages(job)?;
        let first = window.first;
        let mut transferred = 0u64;
        let mut stored = 0u64;
        match self.write_mode {
            KvCompression::Gpu => {
                let ids = job
                    .pages
                    .iter()
                    .enumerate()
                    .map(|(i, (page, _))| (false, first + i as u64, u64::from(page.0)))
                    .chain(job.row.iter().map(|(slot, _)| (true, 0, u64::from(slot.0))));
                for (row, index, id) in ids {
                    let n = self.original_bytes(row);
                    let spans = device_spans(geometry, layout, row, id)?;
                    let base_unit = PackedUnit {
                        kind: (if row { Kind::Row } else { Kind::Page }) as i32,
                        logical_index: index,
                        stored_rank: rank as u32,
                        ..Default::default()
                    };
                    let start = Instant::now();
                    let gpu = self
                        .gpu
                        .as_mut()
                        .ok_or_else(|| invalid("GPU codec unavailable"))?;
                    let plan = if row { &gpu.row } else { &gpu.page };
                    let stored_len = {
                        let mut guard = gpu.codec.lock().unwrap();
                        let res = guard.encode_from_device_with(
                            plan,
                            base_unit,
                            &self.gpu_profile,
                            &spans,
                            stream.raw(),
                            |unit, bytes| {
                                let slen = unit.stored_length;
                                capture.push_slice(unit, bytes);
                                slen
                            },
                        );
                        if res.is_err() {
                            let _ = stream.synchronize();
                        }
                        res
                    }
                    .map_err(decode_error)?;
                    stats.transform_ns += start.elapsed().as_nanos() as u64;
                    stored += stored_len;
                    transferred += n as u64;
                }
            }
            KvCompression::Cpu | KvCompression::Off => {
                let pool = cpu_pool()?;
                // Batch multiple pages into the existing pinned staging buffer (sized for
                // max(page, row)): one D2H burst + one sync, then parallel Rayon encode.
                let page_bytes = self.cpu.page.original_bytes();
                let batch_cap = (self.cpu.staging.bytes().len() / page_bytes.max(1)).max(1);
                let mut page_offset = 0usize;
                for chunk in job.pages.chunks(batch_cap) {
                    let mut batch_spans = Vec::with_capacity(chunk.len());
                    for (slot_idx, (page, _)) in chunk.iter().enumerate() {
                        let spans = device_spans(geometry, layout, false, u64::from(page.0))?;
                        let slot_start = slot_idx * page_bytes;
                        zero_span_gaps(
                            &mut self.cpu.staging.bytes_mut()[slot_start..slot_start + page_bytes],
                            &spans,
                        );
                        batch_spans.push((slot_start, spans));
                    }
                    copy_spans_batch(
                        stream,
                        event,
                        Direction::Out,
                        &batch_spans,
                        self.cpu.staging_alias,
                    )?;
                    let start = Instant::now();
                    let staging = self.cpu.staging.bytes();
                    let page_plan = &self.cpu.page;
                    let profile = &self.cpu_profile;
                    let encoded: Result<Vec<(PackedUnit, Vec<u8>)>> = if chunk.len() == 1 {
                        let base_unit = PackedUnit {
                            kind: Kind::Page as i32,
                            logical_index: first + page_offset as u64,
                            stored_rank: rank as u32,
                            ..Default::default()
                        };
                        let (unit, used) = page_plan
                            .encode_capture(
                                base_unit,
                                profile,
                                &staging[..page_bytes],
                                &mut self.cpu.encode_buf,
                                Some(pool),
                            )
                            .map_err(decode_error)?;
                        Ok(vec![(unit, self.cpu.encode_buf[..used].to_vec())])
                    } else {
                        let max_pkt = page_plan.max_packet();
                        pool.install(|| {
                            (0..chunk.len())
                                .into_par_iter()
                                .map(|slot_idx| {
                                    let base_unit = PackedUnit {
                                        kind: Kind::Page as i32,
                                        logical_index: first + (page_offset + slot_idx) as u64,
                                        stored_rank: rank as u32,
                                        ..Default::default()
                                    };
                                    let slot_start = slot_idx * page_bytes;
                                    let mut out = vec![0u8; max_pkt];
                                    let (unit, used) = page_plan
                                        .encode_capture(
                                            base_unit,
                                            profile,
                                            &staging[slot_start..slot_start + page_bytes],
                                            &mut out,
                                            None,
                                        )
                                        .map_err(decode_error)?;
                                    out.truncate(used);
                                    Ok((unit, out))
                                })
                                .collect()
                        })
                    };
                    let encoded = encoded?;
                    stats.transform_ns += start.elapsed().as_nanos() as u64;
                    for (unit, bytes) in encoded {
                        stored += unit.stored_length;
                        capture.push(unit, bytes);
                        transferred += page_bytes as u64;
                    }
                    page_offset += chunk.len();
                }
                if let Some((slot, _)) = &job.row {
                    let n = self.cpu.row.original_bytes();
                    let spans = device_spans(geometry, layout, true, u64::from(slot.0))?;
                    zero_span_gaps(&mut self.cpu.staging.bytes_mut()[..n], &spans);
                    copy_spans(stream, event, Direction::Out, &spans, self.cpu.staging_alias)?;
                    let base_unit = PackedUnit {
                        kind: Kind::Row as i32,
                        logical_index: 0,
                        stored_rank: rank as u32,
                        ..Default::default()
                    };
                    let start = Instant::now();
                    let (unit, used) = self
                        .cpu
                        .row
                        .encode_capture(
                            base_unit,
                            &self.cpu_profile,
                            &self.cpu.staging.bytes()[..n],
                            &mut self.cpu.encode_buf,
                            Some(pool),
                        )
                        .map_err(decode_error)?;
                    stats.transform_ns += start.elapsed().as_nanos() as u64;
                    stored += unit.stored_length;
                    capture.push(unit, self.cpu.encode_buf[..used].to_vec());
                    transferred += n as u64;
                }
            }
        }
        stats.stored_bytes = stored;
        Ok(transferred)
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_window(
        &mut self,
        rank: usize,
        geometry: &CacheGeometry,
        layout: &RankLayout,
        stream: &Stream,
        event: &Event,
        job: &Job,
        window: &Window,
        stats: &mut DecodeStats,
    ) -> Result<u64> {
        reject_duplicate_pages(job)?;
        let first = window.first;
        let mut transferred = 0u64;
        let mut stored = 0u64;
        let page_bytes = self.cpu.page.original_bytes();
        let batch_cap = (self.cpu.staging.bytes().len() / page_bytes.max(1)).max(1);
        let all_pages_cpu_or_literal = job.pages.len() > 1
            && job.pages.iter().enumerate().all(|(i, _)| {
                let idx = first + i as u64;
                window.packets.iter().any(|p| {
                    let u = &p.unit;
                    u.kind == Kind::Page as i32
                        && u.logical_index == idx
                        && (u.stored_rank == rank as u32 || u.replicas.contains(&(rank as u32)))
                        && (u.representation == Representation::Literal as i32
                            || u.profile.as_slice() == self.cpu_profile.as_slice())
                })
            });

        let mut remaining_pages_offset = 0usize;
        if all_pages_cpu_or_literal {
            let pool = cpu_pool()?;
            for chunk in job.pages.chunks(batch_cap) {
                let mut items = Vec::with_capacity(chunk.len());
                let mut batch_spans = Vec::with_capacity(chunk.len());
                for (slot_idx, (page, _)) in chunk.iter().enumerate() {
                    let index = first + (remaining_pages_offset + slot_idx) as u64;
                    let spans = device_spans(geometry, layout, false, u64::from(page.0))?;
                    let packet = window
                        .packets
                        .iter()
                        .find(|p| {
                            let u = &p.unit;
                            u.kind == Kind::Page as i32
                                && u.logical_index == index
                                && (u.stored_rank == rank as u32
                                    || u.replicas.contains(&(rank as u32)))
                        })
                        .ok_or_else(|| invalid("missing codec rank share"))?;
                    let u = &packet.unit;
                    // The pool lease holds the source through the all-rank fence.
                    let chunks = unsafe { packet.chunks() };
                    let representation = Representation::try_from(u.representation)
                        .map_err(|_| invalid("unknown codec share representation"))?;
                    if u.original_length != page_bytes as u64
                        || u.stored_length != chunks.iter().map(|c| c.len() as u64).sum::<u64>()
                    {
                        return Err(invalid("incompatible codec share"));
                    }
                    stored += u.stored_length;
                    items.push((u, representation, chunks));
                    batch_spans.push((slot_idx * page_bytes, spans));
                }
                let start = Instant::now();
                let staging_slice =
                    &mut self.cpu.staging.bytes_mut()[..chunk.len() * page_bytes];
                let page_plan = &self.cpu.page;
                let cpu_profile = &self.cpu_profile;
                pool.install(|| -> Result<()> {
                    staging_slice
                        .par_chunks_mut(page_bytes)
                        .zip(items.par_iter())
                        .try_for_each(|(raw, (u, representation, chunks))| -> Result<()> {
                            if *representation == Representation::Literal {
                                let mut pos = 0;
                                for c in chunks {
                                    let end = pos + c.len();
                                    raw.get_mut(pos..end)
                                        .ok_or_else(|| invalid("literal length mismatch"))?
                                        .copy_from_slice(c);
                                    pos = end;
                                }
                                if pos != page_bytes {
                                    return Err(invalid("literal length mismatch"));
                                }
                                packed::verify_units(std::slice::from_ref(*u), raw)?;
                                Ok(())
                            } else if u.profile.as_slice() == cpu_profile.as_slice() {
                                page_plan
                                    .decode_checked(
                                        chunks,
                                        raw,
                                        u.stored_crc,
                                        u.original_crc,
                                        None,
                                    )
                                    .map_err(decode_error)
                            } else {
                                Err(invalid("incompatible codec share"))
                            }
                        })
                })?;
                stats.transform_ns += start.elapsed().as_nanos() as u64;
                copy_spans_batch(
                    stream,
                    event,
                    Direction::In,
                    &batch_spans,
                    self.cpu.staging_alias,
                )?;
                transferred += (chunk.len() * page_bytes) as u64;
                remaining_pages_offset += chunk.len();
            }
        }

        let ids = job
            .pages
            .iter()
            .enumerate()
            .skip(remaining_pages_offset)
            .map(|(i, (page, _))| (false, first + i as u64, u64::from(page.0)))
            .chain(job.row.iter().map(|(slot, _)| (true, 0, u64::from(slot.0))));
        for (row, index, id) in ids {
            let n = self.original_bytes(row);
            let spans = device_spans(geometry, layout, row, id)?;
            let selected = |u: &PackedUnit| {
                u.kind == (if row { Kind::Row } else { Kind::Page }) as i32
                    && u.logical_index == index
                    && (u.stored_rank == rank as u32 || u.replicas.contains(&(rank as u32)))
            };
            let packet = window
                .packets
                .iter()
                .find(|p| selected(&p.unit))
                .ok_or_else(|| invalid("missing codec rank share"))?;
            let u = &packet.unit;
            // The pool lease holds the source through the all-rank fence.
            let chunks = unsafe { packet.chunks() };
            let representation = Representation::try_from(u.representation)
                .map_err(|_| invalid("unknown codec share representation"))?;
            if u.original_length != n as u64
                || u.stored_length != chunks.iter().map(|c| c.len() as u64).sum::<u64>()
            {
                return Err(invalid("incompatible codec share"));
            }
            stored += u.stored_length;
            if representation == Representation::Literal {
                let raw = &mut self.cpu.staging.bytes_mut()[..n];
                let mut pos = 0;
                for chunk in &chunks {
                    let end = pos + chunk.len();
                    raw.get_mut(pos..end)
                        .ok_or_else(|| invalid("literal length mismatch"))?
                        .copy_from_slice(chunk);
                    pos = end;
                }
                if pos != n {
                    return Err(invalid("literal length mismatch"));
                }
                packed::verify_units(std::slice::from_ref(u), raw)?;
                copy_spans(stream, event, Direction::In, &spans, self.cpu.staging_alias)?;
            } else if u.profile.as_slice() == self.cpu_profile.as_slice() {
                // Decode ZKV1 on CPU and upload to VRAM (works in both Cpu and Gpu modes).
                let plan = if row { &self.cpu.row } else { &self.cpu.page };
                let raw = &mut self.cpu.staging.bytes_mut()[..n];
                let start = Instant::now();
                plan.decode_checked(
                    &chunks,
                    raw,
                    u.stored_crc,
                    u.original_crc,
                    Some(cpu_pool()?),
                )
                .map_err(decode_error)?;
                stats.transform_ns += start.elapsed().as_nanos() as u64;
                copy_spans(stream, event, Direction::In, &spans, self.cpu.staging_alias)?;
            } else if u.profile.as_slice() == self.gpu_profile.as_slice() {
                // Decode GKV1 on GPU (active in Gpu mode only).
                let gpu = self
                    .gpu
                    .as_mut()
                    .ok_or_else(|| invalid("incompatible codec share"))?;
                let plan = if row { &gpu.row } else { &gpu.page };
                let start = Instant::now();
                {
                    let mut guard = gpu.codec.lock().unwrap();
                    let res = guard.decode_to_device(
                        plan,
                        &chunks,
                        &spans,
                        u.stored_crc,
                        u.original_crc,
                        stream.raw(),
                    );
                    if res.is_err() {
                        let _ = stream.synchronize();
                    }
                    res
                }
                .map_err(decode_error)?;
                stats.transform_ns += start.elapsed().as_nanos() as u64;
            } else {
                return Err(invalid("incompatible codec share"));
            }
            transferred += n as u64;
        }
        stats.stored_bytes = stored;
        Ok(transferred)
    }
}

fn reject_duplicate_pages(job: &Job) -> Result<()> {
    if job
        .pages
        .iter()
        .enumerate()
        .any(|(i, (page, _))| job.pages[..i].iter().any(|(prior, _)| prior == page))
    {
        return Err(invalid("codec window repeats a device page"));
    }
    Ok(())
}

fn device_spans(
    geometry: &CacheGeometry,
    layout: &RankLayout,
    row: bool,
    id: u64,
) -> Result<Vec<DeviceSpan>> {
    let capacity = if row {
        geometry.row_capacity
    } else {
        geometry.page_capacity
    };
    if id >= capacity {
        return Err(invalid("codec destination id outside cache"));
    }
    if row {
        Ok(vec![DeviceSpan {
            device: geometry.row.unit_address(id),
            offset: 0,
            length: geometry.row.unit_bytes as usize,
        }])
    } else {
        Ok(geometry
            .planes
            .iter()
            .zip(&layout.plane_offsets)
            .map(|(p, off)| DeviceSpan {
                device: p.unit_address(id),
                offset: *off as usize,
                length: p.unit_bytes as usize,
            })
            .collect())
    }
}

fn decode_error(error: std::io::Error) -> Error {
    if let Some(checksum) = error
        .get_ref()
        .and_then(|cause| cause.downcast_ref::<Checksum>())
    {
        Error::carrying(*checksum)
    } else {
        Error::carrying(error)
    }
}

fn zero_span_gaps(buf: &mut [u8], spans: &[DeviceSpan]) {
    let mut covered = 0usize;
    let mut contiguous = true;
    for s in spans {
        if s.offset != covered {
            contiguous = false;
            break;
        }
        covered += s.length;
    }
    if contiguous && covered == buf.len() {
        return;
    }
    buf.fill(0);
}

fn copy_spans_batch(
    stream: &Stream,
    event: &Event,
    direction: Direction,
    batch: &[(usize, Vec<DeviceSpan>)],
    host: u64,
) -> Result<()> {
    struct Fence<'a>(&'a Stream);
    impl Drop for Fence<'_> {
        fn drop(&mut self) {
            let _ = self.0.synchronize();
        }
    }
    let fence = Fence(stream);
    for (slot_start, spans) in batch {
        let base = host + *slot_start as u64;
        for p in spans {
            for start in (0..p.length).step_by(BATCH_BYTES) {
                let n = (p.length - start).min(BATCH_BYTES);
                let dev = p.device + start as u64;
                let hst = base + p.offset as u64 + start as u64;
                let (dst, src) = match direction {
                    Direction::In => (dev, hst),
                    Direction::Out => (hst, dev),
                };
                unsafe {
                    stream.copy_batch(&[dst], &[src], &[n])?;
                }
            }
        }
    }
    stream.record(event)?;
    event.synchronize()?;
    std::mem::forget(fence);
    Ok(())
}

fn copy_spans(
    stream: &Stream,
    event: &Event,
    direction: Direction,
    spans: &[DeviceSpan],
    host: u64,
) -> Result<()> {
    struct Fence<'a>(&'a Stream);
    impl Drop for Fence<'_> {
        fn drop(&mut self) {
            let _ = self.0.synchronize();
        }
    }
    let fence = Fence(stream);
    // Each span is inside this rank's cache. The pinned staging buffer stays mapped until exit.
    for p in spans {
        for start in (0..p.length).step_by(BATCH_BYTES) {
            let n = (p.length - start).min(BATCH_BYTES);
            let dev = p.device + start as u64;
            let hst = host + p.offset as u64 + start as u64;
            let (dst, src) = match direction {
                Direction::In => (dev, hst),
                Direction::Out => (hst, dev),
            };
            unsafe {
                stream.copy_batch(&[dst], &[src], &[n])?;
            }
        }
    }
    stream.record(event)?;
    event.synchronize()?;
    std::mem::forget(fence);
    Ok(())
}

#[cfg(all(test, target_os = "linux", feature = "runtime-core"))]
mod tests {
    use super::*;
    use crate::runtime_core::driver::Driver;
    use flare_selene::{
        PageId, StateSlotId,
        codec_copy::PinnedPacket,
        geometry::{CacheGeometry, CodecField, Plane},
        offload::KvCompression,
        shared::wire::packed_unit::Kind,
    };

    fn run_roundtrip_mode(placement: KvCompression) {
        let driver = Driver::load().unwrap();
        let mut ctx = std::ptr::null_mut();
        unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            driver
                .check(
                    "cuDevicePrimaryCtxRetain",
                    (driver.primary_ctx_retain)(&mut ctx, 0),
                )
                .unwrap();
        }
        struct PrimaryCtx<'a>(&'a Driver);
        impl Drop for PrimaryCtx<'_> {
            fn drop(&mut self) {
                unsafe {
                    let _ = (self.0.primary_ctx_release)(0);
                }
            }
        }
        let _guard = PrimaryCtx(&driver);
        let context = ctx as usize;
        let stream = Stream::open_lowest_priority(context, 0).unwrap();
        let (page_bytes, row_bytes) = (4096usize, 65536usize);
        let mut page_mem = stream.mapped_host_buffer(page_bytes * 2).unwrap();
        let mut row_mem = stream.mapped_host_buffer(row_bytes * 2).unwrap();
        let page_addr = stream.mapped_address(&page_mem).unwrap() as u64;
        let row_addr = stream.mapped_address(&row_mem).unwrap() as u64;
        let mut geometry = CacheGeometry {
            placement,
            codec_fields: vec![
                CodecField {
                    buffer: "kv".into(),
                    offset: 0,
                    length: page_bytes as u64,
                    encoding: 2, // Bfloat16
                },
                CodecField {
                    buffer: "state".into(),
                    offset: 0,
                    length: row_bytes as u64,
                    encoding: 1, // Float32Matrix128
                },
            ],
            snapshot_pages: Vec::new(),
            page_tokens: 64,
            page_capacity: 2,
            row_capacity: 2,
            planes: vec![Plane {
                name: "kv".into(),
                address: page_addr,
                unit_bytes: page_bytes as u64,
            }],
            row: Plane {
                name: "state".into(),
                address: row_addr,
                unit_bytes: row_bytes as u64,
            },
        };

        // Populate source page 0 and row 0 in device-visible memory.
        let page0: Vec<u8> = (0..page_bytes).map(|i| ((i / 7) % 13) as u8).collect();
        let mut row0 = vec![0u8; row_bytes];
        for c in 0..128 {
            for r in 0..128 {
                let v = ((r as f32 * 0.01).sin() * (c as f32 * 0.02).cos()).to_bits();
                let idx = (c * 128 + r) * 4;
                row0[idx..idx + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        page_mem.bytes_mut()[..page_bytes].copy_from_slice(&page0);
        row_mem.bytes_mut()[..row_bytes].copy_from_slice(&row0);
        page_mem.bytes_mut()[page_bytes..].fill(0xa5);
        row_mem.bytes_mut()[row_bytes..].fill(0xa5);

        let mut transfers = if geometry.placement == KvCompression::Gpu {
            let path = std::env::var("FLARE_TEST_KV_CODEC_CUBIN")
                .expect("set FLARE_TEST_KV_CODEC_CUBIN to a compiled kv_codec_gpu.cubin");
            let cubin = Arc::new(std::fs::read(path).unwrap());
            Copiers::new(vec![cubin])
                .start(&[(context, 0)], vec![geometry.clone()], None)
                .unwrap()
        } else {
            Transfers::start_unmapped(&[(context, 0)], vec![geometry.clone()]).unwrap()
        };

        // 1. Capture VRAM -> packed host (Direction::Out)
        let cap_win = Window::capture(0);
        transfers
            .submit(Job {
                id: flare_selene::JobId(1),
                window: 0,
                direction: Direction::Out,
                pages: vec![(PageId(0), 0)],
                row: Some((StateSlotId(0), vec![0])),
                replication: Default::default(),
                codec: Some(cap_win.clone()),
            })
            .unwrap();
        let cap_reply = loop {
            let mut out = Vec::new();
            transfers.poll(&mut out);
            if let Some(reply) = out.pop() {
                break reply;
            }
            std::thread::yield_now();
        };
        cap_reply.result.unwrap();
        let slots = cap_win.capture.as_ref().unwrap().take_sorted();
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].unit.kind, Kind::Page as i32);
        assert_eq!(slots[1].unit.kind, Kind::Row as i32);
        assert_eq!(slots[0].unit.representation, Representation::Packet as i32);
        assert_eq!(slots[1].unit.representation, Representation::Packet as i32);
        assert!(slots[0].bytes.len() < page_bytes);
        assert!(slots[1].bytes.len() < row_bytes);

        // 2. Restore packed host -> VRAM page 1 and row 1 (Direction::In)
        let restore_win = Window::restore_pinned(
            0,
            slots
                .iter()
                .map(|s| unsafe {
                    PinnedPacket::from_registered(
                        s.unit.clone(),
                        vec![(s.bytes.as_ptr() as usize, s.bytes.len())],
                    )
                })
                .collect(),
        );
        transfers
            .submit(Job {
                id: flare_selene::JobId(2),
                window: 0,
                direction: Direction::In,
                pages: vec![(PageId(1), 0)],
                row: Some((StateSlotId(1), vec![0])),
                replication: Default::default(),
                codec: Some(restore_win),
            })
            .unwrap();
        let res_reply = loop {
            let mut out = Vec::new();
            transfers.poll(&mut out);
            if let Some(reply) = out.pop() {
                break reply;
            }
            std::thread::yield_now();
        };
        res_reply.result.unwrap();
        assert_eq!(&page_mem.bytes()[page_bytes..page_bytes * 2], &page0);
        assert_eq!(&row_mem.bytes()[row_bytes..row_bytes * 2], &row0);

        // 3. Corrupt stored CRC must fail cleanly and count a stored verification failure.
        let mut bad_u = slots[0].unit.clone();
        bad_u.stored_crc ^= 1;
        let bad_win = Window::restore_pinned(
            0,
            vec![unsafe {
                PinnedPacket::from_registered(
                    bad_u,
                    vec![(slots[0].bytes.as_ptr() as usize, slots[0].bytes.len())],
                )
            }],
        );
        transfers
            .submit(Job {
                id: flare_selene::JobId(3),
                window: 0,
                direction: Direction::In,
                pages: vec![(PageId(1), 0)],
                row: None,
                replication: Default::default(),
                codec: Some(bad_win),
            })
            .unwrap();
        let bad_reply = loop {
            let mut out = Vec::new();
            transfers.poll(&mut out);
            if let Some(reply) = out.pop() {
                break reply;
            }
            std::thread::yield_now();
        };
        assert!(bad_reply.result.is_err());
        let stats = *transfers.decode_counters().unwrap().lock().unwrap();
        assert_eq!(stats.stored_failures, 1);
        assert_eq!(stats.failed_windows, 1);

        geometry.placement = KvCompression::Off;
        let mut off = Transfers::start_unmapped(&[(context, 0)], vec![geometry]).unwrap();
        assert!(off.codec_profile().is_none());
        transfers.close().unwrap();
        off.close().unwrap();
    }

    #[test]
    #[ignore = "exercises CPU VRAM <-> DRAM capture and restore onto a GPU"]
    fn cpu_vram_capture_and_restore_copies_exact_bytes_and_rejects_corruption() {
        run_roundtrip_mode(KvCompression::Cpu);
    }

    #[test]
    #[ignore = "exercises GPU GKV1 VRAM <-> DRAM capture and restore onto a GPU"]
    fn gpu_vram_capture_and_restore_copies_exact_bytes_and_rejects_corruption() {
        run_roundtrip_mode(KvCompression::Gpu);
    }

    #[test]
    #[ignore = "exercises mixed CPU ZKV1 <-> GPU GKV1 cross-mode restore without abandoning cache"]
    fn mixed_cpu_and_gpu_modes_restore_each_others_packets_and_write_their_own() {
        let driver = Driver::load().unwrap();
        let mut ctx = std::ptr::null_mut();
        unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            driver
                .check(
                    "cuDevicePrimaryCtxRetain",
                    (driver.primary_ctx_retain)(&mut ctx, 0),
                )
                .unwrap();
        }
        struct PrimaryCtx<'a>(&'a Driver);
        impl Drop for PrimaryCtx<'_> {
            fn drop(&mut self) {
                unsafe {
                    let _ = (self.0.primary_ctx_release)(0);
                }
            }
        }
        let _guard = PrimaryCtx(&driver);
        let context = ctx as usize;
        let stream = Stream::open_lowest_priority(context, 0).unwrap();
        let (page_bytes, row_bytes) = (4096usize, 65536usize);
        let mut page_mem = stream.mapped_host_buffer(page_bytes * 2).unwrap();
        let mut row_mem = stream.mapped_host_buffer(row_bytes * 2).unwrap();
        let page_addr = stream.mapped_address(&page_mem).unwrap() as u64;
        let row_addr = stream.mapped_address(&row_mem).unwrap() as u64;
        let mut base_geo = CacheGeometry {
            placement: KvCompression::Cpu,
            codec_fields: vec![
                CodecField {
                    buffer: "kv".into(),
                    offset: 0,
                    length: page_bytes as u64,
                    encoding: 2,
                },
                CodecField {
                    buffer: "state".into(),
                    offset: 0,
                    length: row_bytes as u64,
                    encoding: 1,
                },
            ],
            snapshot_pages: Vec::new(),
            page_tokens: 64,
            page_capacity: 2,
            row_capacity: 2,
            planes: vec![Plane {
                name: "kv".into(),
                address: page_addr,
                unit_bytes: page_bytes as u64,
            }],
            row: Plane {
                name: "state".into(),
                address: row_addr,
                unit_bytes: row_bytes as u64,
            },
        };
        let page0: Vec<u8> = (0..page_bytes).map(|i| ((i / 7) % 13) as u8).collect();
        let mut row0 = vec![0u8; row_bytes];
        for c in 0..128 {
            for r in 0..128 {
                let v = ((r as f32 * 0.01).sin() * (c as f32 * 0.02).cos()).to_bits();
                let idx = (c * 128 + r) * 4;
                row0[idx..idx + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        page_mem.bytes_mut()[..page_bytes].copy_from_slice(&page0);
        row_mem.bytes_mut()[..row_bytes].copy_from_slice(&row0);

        let cpu_profile = CodecLayout::for_revision(
            &base_geo,
            flare_selene::shared::codec_layout::REVISION_CPU,
        )
        .profile()
        .unwrap();
        let gpu_profile = CodecLayout::for_revision(
            &base_geo,
            flare_selene::shared::codec_layout::REVISION_GPU,
        )
        .profile()
        .unwrap();
        assert_ne!(cpu_profile, gpu_profile);

        let test_cubin = || -> Vec<Arc<Vec<u8>>> {
            let path = std::env::var("FLARE_TEST_KV_CODEC_CUBIN")
                .expect("set FLARE_TEST_KV_CODEC_CUBIN to a compiled kv_codec_gpu.cubin");
            vec![Arc::new(std::fs::read(path).unwrap())]
        };
        let mut cpu_xfer = Transfers::start_unmapped(&[(context, 0)], vec![base_geo.clone()]).unwrap();
        base_geo.placement = KvCompression::Gpu;
        let gpu_copiers = Copiers::new(test_cubin());
        let mut gpu_xfer = gpu_copiers.start(&[(context, 0)], vec![base_geo], None).unwrap();

        // CPU mode advertises ZKV1 only (zero GPU VRAM); GPU mode advertises [ZKV1, GKV1].
        assert_eq!(cpu_xfer.codec_receive_profiles(), vec![cpu_profile]);
        assert_eq!(gpu_xfer.codec_receive_profiles(), vec![cpu_profile, gpu_profile]);

        let run_one = |xfer: &mut Transfers, job: Job| {
            xfer.submit(job).unwrap();
            loop {
                let mut out = Vec::new();
                xfer.poll(&mut out);
                if let Some(reply) = out.pop() {
                    reply.result.unwrap();
                    return;
                }
                std::thread::yield_now();
            }
        };

        // 1. Capture in CPU mode (writes ZKV1 / cpu_profile)
        let cpu_cap = Window::capture(0);
        run_one(
            &mut cpu_xfer,
            Job {
                id: flare_selene::JobId(1),
                window: 0,
                direction: Direction::Out,
                pages: vec![(PageId(0), 0)],
                row: Some((StateSlotId(0), vec![0])),
                replication: Default::default(),
                codec: Some(cpu_cap.clone()),
            },
        );
        let cpu_slots = cpu_cap.capture.as_ref().unwrap().take_sorted();
        assert!(cpu_slots.iter().all(|s| s.unit.profile.as_slice() == cpu_profile.as_slice()));

        // 2. Restore CPU ZKV1 packets using GPU-mode Transfers (decodes on CPU -> VRAM)
        page_mem.bytes_mut()[page_bytes..].fill(0xaa);
        row_mem.bytes_mut()[row_bytes..].fill(0xaa);
        let zkv_win = Window::restore_pinned(
            0,
            cpu_slots
                .iter()
                .map(|s| unsafe {
                    PinnedPacket::from_registered(
                        s.unit.clone(),
                        vec![(s.bytes.as_ptr() as usize, s.bytes.len())],
                    )
                })
                .collect(),
        );
        run_one(
            &mut gpu_xfer,
            Job {
                id: flare_selene::JobId(2),
                window: 0,
                direction: Direction::In,
                pages: vec![(PageId(1), 0)],
                row: Some((StateSlotId(1), vec![0])),
                replication: Default::default(),
                codec: Some(zkv_win),
            },
        );
        assert_eq!(&page_mem.bytes()[page_bytes..], &page0);
        assert_eq!(&row_mem.bytes()[row_bytes..], &row0);

        // 3. Re-capture in GPU mode (writes GKV1 / gpu_profile)
        let gpu_cap = Window::capture(0);
        run_one(
            &mut gpu_xfer,
            Job {
                id: flare_selene::JobId(3),
                window: 0,
                direction: Direction::Out,
                pages: vec![(PageId(1), 0)],
                row: Some((StateSlotId(1), vec![0])),
                replication: Default::default(),
                codec: Some(gpu_cap.clone()),
            },
        );
        let gpu_slots = gpu_cap.capture.as_ref().unwrap().take_sorted();
        assert!(gpu_slots.iter().all(|s| s.unit.profile.as_slice() == gpu_profile.as_slice()));

        // 4. CPU-mode Transfers rejects GKV1 cleanly (CPU mode allocates no GPU VRAM; GKV1 is a cache miss).
        let gkv_win = Window::restore_pinned(
            0,
            gpu_slots
                .iter()
                .map(|s| unsafe {
                    PinnedPacket::from_registered(
                        s.unit.clone(),
                        vec![(s.bytes.as_ptr() as usize, s.bytes.len())],
                    )
                })
                .collect(),
        );
        cpu_xfer
            .submit(Job {
                id: flare_selene::JobId(4),
                window: 0,
                direction: Direction::In,
                pages: vec![(PageId(1), 0)],
                row: Some((StateSlotId(1), vec![0])),
                replication: Default::default(),
                codec: Some(gkv_win),
            })
            .unwrap();
        let cpu_reply = loop {
            let mut out = Vec::new();
            cpu_xfer.poll(&mut out);
            if let Some(reply) = out.pop() {
                break reply;
            }
            std::thread::yield_now();
        };
        assert!(cpu_reply.result.is_err());

        cpu_xfer.close().unwrap();
        gpu_xfer.close().unwrap();
    }
}
