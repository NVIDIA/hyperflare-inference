// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::cuda::PinnedHost;
use flare_selene::{
    codec_copy::{DecodeCounters, DecodeStats, Window},
    shared::{
        codec::{Checksum, Plan},
        packed,
        wire::{
            CodecLayout, PackedUnit,
            packed_unit::{Kind, Representation},
        },
    },
};
use std::sync::OnceLock;

fn restore_pool() -> flare_selene::Result<&'static rayon::ThreadPool> {
    static POOL: OnceLock<std::io::Result<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism()
            .map_or(4, usize::from)
            .clamp(1, 8);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("kv-restore-{i}"))
            .build()
            .map_err(std::io::Error::other)
    })
    .as_ref()
    .map_err(|e| flare_selene::geometry::invalid(e.to_string()))
}

struct DeviceSpan {
    device: u64,
    offset: usize,
    length: usize,
}

pub(super) struct RankCodec {
    counters: DecodeCounters,
    profile: [u8; 32],
    page: Plan,
    row: Plan,
    restore: PinnedHost,
    restore_alias: u64,
}

impl RankCodec {
    pub(super) fn new(
        stream: &Stream,
        geometry: &CacheGeometry,
        counters: DecodeCounters,
    ) -> Result<Self> {
        let ([page, row], profile) = CodecLayout::from(geometry)
            .plans_and_profile()
            .map_err(|e| invalid(e.to_string()))?;
        let bytes = page.original_bytes().max(row.original_bytes());
        let restore = stream.mapped_host_buffer(bytes)?;
        let restore_alias = stream.mapped_address(&restore)? as u64;
        Ok(Self {
            counters,
            profile,
            page,
            row,
            restore,
            restore_alias,
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
        let result = self.copy_window(rank, geometry, layout, stream, event, job, window, &mut stats);
        {
            let mut totals = self.counters.lock().unwrap();
            totals.transform_ns += stats.transform_ns;
            totals.stored_bytes += stats.stored_bytes;
            match &result {
                Ok(_) => {
                    totals.original_bytes +=
                        self.page.original_bytes() as u64 * job.pages.len() as u64;
                    if job.row.is_some() {
                        totals.original_bytes += self.row.original_bytes() as u64;
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

    fn copy_window(
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
        if job.direction != Direction::In {
            return Err(invalid("packed windows require a restore job"));
        }
        if job
            .pages
            .iter()
            .enumerate()
            .any(|(i, (page, _))| job.pages[..i].iter().any(|(prior, _)| prior == page))
        {
            return Err(invalid("codec window repeats a device page"));
        }
        let first = window.first;
        let ids = job
            .pages
            .iter()
            .enumerate()
            .map(|(i, (page, _))| (false, first + i as u64, u64::from(page.0)))
            .chain(job.row.iter().map(|(slot, _)| (true, 0, u64::from(slot.0))));
        let mut transferred = 0u64;
        let mut stored = 0u64;
        for (row, index, id) in ids {
            let plan = if row { &self.row } else { &self.page };
            let n = plan.original_bytes();
            let capacity = if row {
                geometry.row_capacity
            } else {
                geometry.page_capacity
            };
            if id >= capacity {
                return Err(invalid("codec destination id outside cache"));
            }
            let spans: Vec<_> = if row {
                vec![DeviceSpan {
                    device: geometry.row.unit_address(id),
                    offset: 0,
                    length: geometry.row.unit_bytes as usize,
                }]
            } else {
                geometry
                    .planes
                    .iter()
                    .zip(&layout.plane_offsets)
                    .map(|(p, off)| DeviceSpan {
                        device: p.unit_address(id),
                        offset: *off as usize,
                        length: p.unit_bytes as usize,
                    })
                    .collect()
            };
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
                || (representation == Representation::Packet && u.profile != self.profile)
            {
                return Err(invalid("incompatible codec share"));
            }
            stored += u.stored_length;
            let raw = &mut self.restore.bytes_mut()[..n];
            if representation == Representation::Literal {
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
                plan.decode_checked(
                    &chunks,
                    raw,
                    u.stored_crc,
                    u.original_crc,
                    Some(restore_pool()?),
                )
                .map_err(decode_error)?;
                stats.transform_ns += start.elapsed().as_nanos() as u64;
            }
            copy(stream, event, &spans, self.restore_alias)?;
            transferred += n as u64;
        }
        stats.stored_bytes = stored;
        Ok(transferred)
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

fn copy(stream: &Stream, event: &Event, spans: &[DeviceSpan], host: u64) -> Result<()> {
    struct Fence<'a>(&'a Stream);
    impl Drop for Fence<'_> {
        fn drop(&mut self) {
            let _ = self.0.synchronize();
        }
    }
    let fence = Fence(stream);
    // Each span is inside this rank's cache. The pinned restore buffer stays mapped until exit.
    for p in spans {
        for start in (0..p.length).step_by(BATCH_BYTES) {
            let n = (p.length - start).min(BATCH_BYTES);
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

    #[test]
    #[ignore = "restores packed pages and rows onto a GPU"]
    fn cpu_packed_restore_copies_exact_bytes_and_rejects_corruption() {
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
        let (page_bytes, row_bytes) = (4096usize, 8192usize);
        let page_mem = stream.mapped_host_buffer(page_bytes * 2).unwrap();
        let row_mem = stream.mapped_host_buffer(row_bytes).unwrap();
        let page_addr = stream.mapped_address(&page_mem).unwrap() as u64;
        let row_addr = stream.mapped_address(&row_mem).unwrap() as u64;
        let mut geometry = CacheGeometry {
            placement: KvCompression::Cpu,
            codec_fields: vec![
                CodecField {
                    buffer: "kv".into(),
                    offset: 0,
                    length: page_bytes as u64,
                    encoding: 0,
                },
                CodecField {
                    buffer: "state".into(),
                    offset: 0,
                    length: row_bytes as u64,
                    encoding: 0,
                },
            ],
            snapshot_pages: Vec::new(),
            page_tokens: 64,
            page_capacity: 2,
            row_capacity: 1,
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
        let profile = geometry.codec_profile().unwrap();
        let [page_plan, row_plan] = CodecLayout::from(&geometry).plans().unwrap();
        let mut transfers =
            Transfers::start_unmapped(&[(context, 0)], vec![geometry.clone()]).unwrap();

        let page0: Vec<u8> = (0..page_bytes).map(|i| ((i / 7) % 13) as u8).collect();
        let row0: Vec<u8> = (0..row_bytes).map(|i| ((i / 11 + 3) % 9) as u8).collect();
        let mut back_p = vec![0; page_bytes];
        let mut back_r = vec![0; row_bytes];
        let p_pkt = page_plan.encode_verified(&page0, &mut back_p, None).unwrap();
        let r_pkt = row_plan.encode_verified(&row0, &mut back_r, None).unwrap();

        let p_unit = PackedUnit {
            kind: Kind::Page as i32,
            logical_index: 0,
            representation: Representation::Packet as i32,
            profile: profile.to_vec(),
            original_length: page_bytes as u64,
            stored_length: p_pkt.len() as u64,
            stored_crc: packed::crc32(&p_pkt),
            original_crc: packed::crc32(&page0),
            ..Default::default()
        };
        let r_unit = PackedUnit {
            kind: Kind::Row as i32,
            logical_index: 0,
            representation: Representation::Packet as i32,
            profile: profile.to_vec(),
            original_length: row_bytes as u64,
            stored_length: r_pkt.len() as u64,
            stored_crc: packed::crc32(&r_pkt),
            original_crc: packed::crc32(&row0),
            ..Default::default()
        };
        let window = Window::restore_pinned(
            0,
            vec![
                unsafe {
                    PinnedPacket::from_registered(
                        p_unit.clone(),
                        vec![(p_pkt.as_ptr() as usize, p_pkt.len())],
                    )
                },
                unsafe {
                    PinnedPacket::from_registered(
                        r_unit.clone(),
                        vec![(r_pkt.as_ptr() as usize, r_pkt.len())],
                    )
                },
            ],
        );
        let job = Job {
            id: flare_selene::JobId(1),
            window: 0,
            direction: Direction::In,
            pages: vec![(PageId(0), 0)],
            row: Some((StateSlotId(0), vec![0])),
            replication: Default::default(),
            codec: Some(window),
        };
        transfers.submit(job).unwrap();
        let reply = loop {
            let mut out = Vec::new();
            transfers.poll(&mut out);
            if let Some(reply) = out.pop() {
                break reply;
            }
            std::thread::yield_now();
        };
        reply.result.unwrap();
        assert_eq!(&page_mem.bytes()[..page_bytes], &page0);
        assert_eq!(&row_mem.bytes()[..row_bytes], &row0);

        // Corrupt stored CRC must fail cleanly and count a stored verification failure.
        let mut bad_u = p_unit;
        bad_u.stored_crc ^= 1;
        let bad_win = Window::restore_pinned(
            0,
            vec![unsafe {
                PinnedPacket::from_registered(bad_u, vec![(p_pkt.as_ptr() as usize, p_pkt.len())])
            }],
        );
        transfers
            .submit(Job {
                id: flare_selene::JobId(2),
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
}
