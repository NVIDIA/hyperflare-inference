// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    cuda::PinnedHost,
    offload::kv_codec::{DeviceSpan, GpuCodec, Plan, RestoreUnit, restore_pool},
};
use flare_selene::{
    codec_copy::Window,
    offload::KvCompression,
    shared::{packed, wire::CodecLayout, wire::PackedUnit},
};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

struct Totals {
    bytes: AtomicU64,
    stored: AtomicU64,
    transform: AtomicU64,
    failures: AtomicU64,
}
impl Totals {
    const fn new() -> Self {
        Self {
            bytes: AtomicU64::new(0),
            stored: AtomicU64::new(0),
            transform: AtomicU64::new(0),
            failures: AtomicU64::new(0),
        }
    }
}
static TOTAL: [Totals; 2] = [const { Totals::new() }; 2];
#[cfg(test)]
static SCRATCH: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

#[cfg(feature = "server")]
pub(crate) fn codec_metrics(text: &mut String) {
    use std::fmt::Write as _;
    for (stage, value) in [
        ("stored", packed::STORED_FAILURES.load(Relaxed)),
        (
            "original",
            crate::offload::kv_codec::ORIGINAL_FAILURES.load(Relaxed),
        ),
    ] {
        let _ = writeln!(
            text,
            "flare_kv_codec_verification_failures_total{{stage=\"{stage}\"}} {value}"
        );
    }
    for (tier, t) in ["cpu", "gpu"].into_iter().zip(&TOTAL) {
        for (name, value) in [
            ("original_bytes_total", &t.bytes),
            ("stored_bytes_total", &t.stored),
            ("failed_windows_total", &t.failures),
            ("transform_nanoseconds_total", &t.transform),
        ] {
            let _ = writeln!(
                text,
                "flare_kv_codec_{name}{{placement=\"{tier}\",boundary=\"out\"}} {}",
                value.load(Relaxed)
            );
        }
    }
}

pub(super) struct RankCodec {
    pub(super) reserved: [u64; 2],
    profile: [u8; 32],
    page: Plan,
    row: Plan,
    gpu: Option<GpuCodec>,
    restore: PinnedHost,
    restore_alias: u64,
}

#[cfg(all(test, feature = "runtime-core"))]
pub(crate) fn live_scratch() -> [u64; 2] {
    SCRATCH.each_ref().map(|n| n.load(Relaxed))
}

#[cfg(all(test, feature = "runtime-core"))]
pub(crate) fn workspace_parts(g: &CacheGeometry) -> Result<[u64; 6]> {
    let [page, row] = CodecLayout::from(g)
        .plans()
        .map_err(|e| invalid(e.to_string()))?;
    Ok(GpuCodec::workspace_parts(&[&page, &row]))
}

impl RankCodec {
    pub(super) fn new(
        context: usize,
        stream: &Stream,
        geometry: &CacheGeometry,
    ) -> Result<Option<Self>> {
        let Some(profile) = geometry.codec_profile() else {
            return Ok(None);
        };
        let [page, row] = CodecLayout::from(geometry)
            .plans()
            .map_err(|e| invalid(e.to_string()))?;
        let bytes = page.original_bytes().max(row.original_bytes());
        let gpu = (geometry.placement == KvCompression::Gpu)
            .then(|| GpuCodec::open(context, &[&page, &row]))
            .transpose()
            .map_err(|e| invalid(e.to_string()))?;
        let restore = stream.mapped_host_buffer(bytes)?;
        let restore_alias = stream.mapped_address(&restore)? as u64;
        let reserved = [bytes as u64, gpu.as_ref().map_or(0, |g| g.reserved)];
        #[cfg(test)]
        for (i, n) in reserved.into_iter().enumerate() {
            SCRATCH[i].fetch_add(n, Relaxed);
        }
        Ok(Some(Self {
            reserved,
            profile,
            page,
            row,
            gpu,
            restore,
            restore_alias,
        }))
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
        let t = &TOTAL[usize::from(self.gpu.is_some())];
        let result = self.copy_window(rank, geometry, layout, stream, event, job, window);
        match result {
            Ok(_) => {
                t.bytes.fetch_add(
                    self.page.original_bytes() as u64 * job.pages.len() as u64
                        + if job.row.is_some() {
                            self.row.original_bytes() as u64
                        } else {
                            0
                        },
                    Relaxed,
                );
            }
            Err(_) => {
                t.failures.fetch_add(1, Relaxed);
                stream.synchronize()?;
            }
        }
        result
    }
    fn copy_window(
        &mut self,
        rank: usize,
        geometry: &CacheGeometry,
        layout: &RankLayout,
        stream: &Stream,
        event: &Event,
        job: &Job,
        window: &Window,
    ) -> Result<u64> {
        let first = window.0.first;
        let totals = &TOTAL[usize::from(self.gpu.is_some())];
        let mut transferred = 0;
        let mut stored = 0;
        if job
            .pages
            .iter()
            .enumerate()
            .any(|(i, (page, _))| job.pages[..i].iter().any(|(prior, _)| prior == page))
        {
            return Err(invalid("codec window repeats a device page"));
        }
        let ids = job
            .pages
            .iter()
            .enumerate()
            .map(|(i, (page, _))| (false, first + i as u64, u64::from(page.0)))
            .chain(job.row.iter().map(|(slot, _)| (true, 0, u64::from(slot.0))));
        if job.direction != Direction::In {
            return Err(invalid("packed windows require a restore job"));
        }
        let mut pending = Vec::new();
        let flush =
            |gpu: &mut GpuCodec, plan: &Plan, units: &mut Vec<RestoreUnit<'_>>| -> Result<()> {
                // The all-rank reply fences exclusive destination IDs.
                let ns =
                    unsafe { gpu.decode_units(plan, units) }.map_err(|e| invalid(e.to_string()))?;
                totals.transform.fetch_add(ns, Relaxed);
                units.clear();
                Ok(())
            };
        for (row, index, id) in ids {
            let plan = if row { &self.row } else { &self.page };
            let n = plan.original_bytes();
            if id
                >= if row {
                    geometry.row_capacity
                } else {
                    geometry.page_capacity
                }
            {
                return Err(invalid("codec destination id outside cache"));
            }
            let spans: Vec<_> = if row {
                vec![DeviceSpan {
                    device: geometry.row.unit_address(id),
                    offset: 0,
                    length: geometry.row.unit_bytes as i32,
                }]
            } else {
                geometry
                    .planes
                    .iter()
                    .zip(&layout.plane_offsets)
                    .map(|(p, off)| DeviceSpan {
                        device: p.unit_address(id),
                        offset: *off as i32,
                        length: p.unit_bytes as i32,
                    })
                    .collect()
            };

            let selected = |u: &PackedUnit| {
                u.kind == i32::from(row)
                    && u.logical_index == index
                    && (u.stored_rank == rank as u32 || u.replicas.contains(&(rank as u32)))
            };
            let packet = window
                .0
                .packets
                .iter()
                .find(|p| selected(&p.unit))
                .ok_or_else(|| invalid("missing codec rank share"))?;
            let u = &packet.unit;
            // The pool lease holds the source through the all-rank fence.
            let chunks = unsafe { packet.chunks() };
            if u.representation > 1
                || u.original_length != n as u64
                || u.stored_length != chunks.iter().map(|c| c.len() as u64).sum::<u64>()
                || (u.representation == 1 && u.profile != self.profile)
            {
                return Err(invalid("incompatible codec share"));
            }
            stored += u.stored_length;
            if !pending.is_empty()
                && (row || u.representation == 0)
                && let Some(gpu) = self.gpu.as_mut()
            {
                flush(gpu, &self.page, &mut pending)?;
            }
            if u.representation == 1
                && let Some(gpu) = self.gpu.as_mut()
            {
                pending.push(RestoreUnit {
                    chunks,
                    stored: u.stored_crc,
                    original: u.original_crc,
                    spans,
                });
                transferred += u.stored_length;
                if row {
                    flush(gpu, plan, &mut pending)?;
                }
            } else {
                let raw = &mut self.restore.bytes_mut()[..n];
                if u.representation == 0 {
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
                } else {
                    let start = Instant::now();
                    plan.decode_checked(&chunks, raw, u.stored_crc, u.original_crc, Some(restore_pool()?))
                        .map_err(|e| invalid(e.to_string()))?;
                    totals
                        .transform
                        .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
                }
                copy(stream, event, &spans, self.restore_alias)?;
                transferred += n as u64;
            }
        }
        if !pending.is_empty()
            && let Some(gpu) = self.gpu.as_mut()
        {
            flush(gpu, &self.page, &mut pending)?;
        }
        totals.stored.fetch_add(stored, Relaxed);
        Ok(transferred)
    }
}

#[cfg(test)]
impl Drop for RankCodec {
    fn drop(&mut self) {
        for (i, n) in self.reserved.into_iter().enumerate() {
            SCRATCH[i].fetch_sub(n, Relaxed);
        }
    }
}

fn copy(stream: &Stream, event: &Event, spans: &[DeviceSpan], host: u64) -> Result<()> {
    struct Fence<'a>(&'a Stream);
    impl Drop for Fence<'_> {
        fn drop(&mut self) {
            let _ = self.0.synchronize();
        }
    }
    let fence = Fence(stream);
    // Each span is inside this rank's cache. The scratch is mapped until the worker exits.
    for p in spans {
        for start in (0..p.length as usize).step_by(BATCH_BYTES) {
            let n = (p.length as usize - start).min(BATCH_BYTES);
            let (dst, src) = (
                p.device + start as u64,
                host + p.offset as u64 + start as u64,
            );
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(kv_codec_cuda)]
    #[test]
    #[ignore = "requires one free CUDA device"]
    fn packed_suffix_restores_on_each_rank_through_both_placements() {
        use flare_selene::{PageId, StateSlotId, geometry::CodecField, testing::geometry};
        use std::ffi::c_void;
        unsafe extern "C" {
            fn cudaSetDevice(id: i32) -> i32;
            fn cudaMalloc(p: *mut *mut c_void, n: usize) -> i32;
            fn cudaFree(p: *mut c_void) -> i32;
            fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: i32) -> i32;
            fn cuCtxGetCurrent(context: *mut usize) -> i32;
        }
        struct Device(*mut c_void);
        impl Drop for Device {
            fn drop(&mut self) {
                assert_eq!(unsafe { cudaFree(self.0) }, 0);
            }
        }
        let n = 65536;
        let mut ptr = std::ptr::null_mut();
        let mut context = 0;
        unsafe {
            assert_eq!(cudaSetDevice(0), 0);
            assert_eq!(cudaMalloc(&mut ptr, 8 * n), 0);
            assert_eq!(cuCtxGetCurrent(&mut context), 0);
        }
        let memory = Device(ptr);
        let error = GpuCodec::open(context.wrapping_add(1), &[]).err().unwrap();
        assert!(error.to_string().contains("current model context"));
        let mut geometry = geometry(64, &[("kv", n as u64)], n as u64);
        geometry.planes[0].address = memory.0 as u64;
        geometry.row.address = memory.0 as u64 + (2 * n) as u64;
        geometry.page_capacity = 2;
        geometry.row_capacity = 2;
        for buffer in ["kv", "state"] {
            geometry.codec_fields.push(CodecField {
                buffer: buffer.into(),
                offset: 0,
                length: n as u64,
                encoding: 0,
            });
        }
        let mut second = geometry.clone();
        second.planes[0].address += (4 * n) as u64;
        second.row.address += (4 * n) as u64;
        let ranks = [geometry, second];
        let transfers = |placement| {
            let geometry = ranks.clone().map(|mut g| {
                g.placement = placement;
                g
            });
            Transfers::start_unmapped(&[(context, 0); 2], geometry.to_vec()).unwrap()
        };
        let mut cpu = transfers(KvCompression::Cpu);
        let mut gpu = transfers(KvCompression::Gpu);
        fn send(
            t: &mut Transfers,
            stream: &Stream,
            packets: &[flare_selene::testing::Packet],
        ) -> Result<u64> {
            let mut owners = Vec::new();
            let mut spans = Vec::new();
            for packet in packets {
                let mut owner = stream.mapped_host_buffer(packet.data.len())?;
                owner.bytes_mut().copy_from_slice(&packet.data);
                spans.push(unsafe {
                    flare_selene::codec_copy::PinnedPacket::from_registered(
                        packet.unit.clone(),
                        vec![(owner.address(), packet.data.len())],
                    )
                });
                owners.push(owner);
            }
            let window = Window::restore_pinned(12, spans);
            t.submit(Job {
                id: JobId(1),
                window: 0,
                direction: Direction::In,
                codec: Some(window),
                pages: vec![(PageId(1), 0)],
                row: Some((StateSlotId(1), vec![])),
                replication: Default::default(),
            })?;
            let mut replies = Vec::new();
            let end = Instant::now() + std::time::Duration::from_secs(30);
            while replies.is_empty() {
                t.poll(&mut replies);
                assert!(Instant::now() < end);
                std::thread::yield_now();
            }
            let r = replies.pop().unwrap();
            r.result?;
            Ok(r.bytes)
        }
        let stream = Stream::open(context, 0).unwrap();
        let profile = cpu.codec_profile().unwrap();
        let plans = CodecLayout::from(&ranks[0]).plans().unwrap();
        let mut writer = Vec::new();
        let mut originals = Vec::new();
        for (kind, plan) in plans.iter().enumerate() {
            for rank in 0..ranks.len() {
                let mut random = 37u32;
                let bytes: Vec<_> = (0..n).map(|i| {
                    if kind == 1 && rank == 1 {
                        random ^= random << 13;
                        random ^= random >> 17;
                        random ^= random << 5;
                        random as u8
                    } else {
                        ((i / 211 + i / 1024 + rank + kind) % 57) as u8
                    }
                }).collect();
                let unit = PackedUnit {
                    kind: kind as i32,
                    logical_index: if kind == 0 { 12 } else { 0 },
                    stored_rank: rank as u32,
                    ..Default::default()
                };
                let packet = flare_selene::testing::checkpoint_packet(plan, &profile, unit, &bytes);
                assert_eq!(packet.unit.representation, i32::from(kind == 0 || rank == 0));
                writer.push(packet);
                originals.push(bytes);
            }
        }
        for t in [&mut gpu, &mut cpu] {
            let poison = vec![0xa5; n];
            for g in &ranks {
                for plane in [&g.planes[0], &g.row] {
                    let address = plane.unit_address(1) as *mut c_void;
                    assert_eq!(unsafe { cudaMemcpy(address, poison.as_ptr().cast(), n, 1) }, 0);
                }
            }
            send(t, &stream, &writer).unwrap();
            for (i, packet) in writer.iter().enumerate() {
                let g = &ranks[packet.unit.stored_rank as usize];
                let plane = if packet.unit.kind == 0 { &g.planes[0] } else { &g.row };
                let mut output = vec![0u8; n];
                let address = plane.unit_address(1) as *const c_void;
                assert_eq!(unsafe { cudaMemcpy(output.as_mut_ptr().cast(), address, n, 2) }, 0);
                assert!(output == originals[i], "restored share differs: {:?}", packet.unit);
            }
            let mut corrupt = writer.clone();
            corrupt[0].unit.representation = 2;
            assert!(send(t, &stream, &corrupt).is_err());
            assert_eq!(t.in_flight(), 0);
            let mut corrupt = writer.clone();
            corrupt[0].unit.stored_crc ^= 1;
            assert!(send(t, &stream, &corrupt).is_err());
            let mut corrupt = writer.clone();
            corrupt[0].unit.original_crc ^= 1;
            assert!(send(t, &stream, &corrupt).is_err());
            assert_eq!(t.in_flight(), 0);

            // Keep the expected original CRC and give the damaged packet a valid stored CRC.
            let mut corrupt = writer.clone();
            let mut payload = corrupt[0].data.to_vec();
            let frame = u32::from_le_bytes(payload[48..52].try_into().unwrap()) as usize;
            assert_eq!(payload[frame], 2);
            let first_count = frame + 4 + 32;
            payload[first_count] = 0;
            corrupt[0].unit.stored_crc = packed::crc32(&payload);
            corrupt[0].data = payload.into();
            assert!(send(t, &stream, &corrupt).is_err());
            assert_eq!(t.in_flight(), 0);
        }
        for totals in &TOTAL {
            assert!(totals.transform.load(Relaxed) > 0);
        }
        let mut off = transfers(KvCompression::Off);
        assert!(off.codec_profile().is_none());
        assert!(send(&mut off, &stream, &writer).is_err());
        off.close().unwrap();
        cpu.close().unwrap();
        gpu.close().unwrap();
    }
}
