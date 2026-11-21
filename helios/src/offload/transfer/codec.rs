// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    cuda::PinnedHost,
    offload::kv_codec::{DeviceSpan, Encoding, Field, GpuCodec, Plan},
};
use flare_selene::{
    codec_copy::{Action, Packet, Window},
    offload::KvCompression,
    shared::{packed, record::checksum, wire::PackedUnit},
};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering::Relaxed},
};

struct Totals {
    bytes: AtomicU64,
    pcie: AtomicU64,
    ns: AtomicU64,
    transform: AtomicU64,
    metadata: AtomicU64,
    failures: AtomicU64,
}
impl Totals {
    const fn new() -> Self {
        Self {
            bytes: AtomicU64::new(0),
            pcie: AtomicU64::new(0),
            ns: AtomicU64::new(0),
            transform: AtomicU64::new(0),
            metadata: AtomicU64::new(0),
            failures: AtomicU64::new(0),
        }
    }
}
static TOTAL: [[Totals; 2]; 2] = [const { [const { Totals::new() }; 2] }; 2];
pub(super) fn count_encoded_input(bytes: [u64; 2]) {
    for (placement, n) in TOTAL.iter().zip(bytes) {
        if n != 0 {
            placement[0].bytes.fetch_add(n, Relaxed);
        }
    }
}

#[cfg(test)]
pub(super) fn encoded_input() -> [u64; 2] {
    std::array::from_fn(|i| TOTAL[i][0].bytes.load(Relaxed))
}

static GPU_MIRRORS: AtomicU64 = AtomicU64::new(0);
static SCRATCH: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static HIGH: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

#[cfg(feature = "server")]
pub(crate) fn codec_metrics(text: &mut String) {
    use std::fmt::Write as _;
    let _ = writeln!(
        text,
        "flare_kv_codec_mirror_stalls_total {}",
        flare_selene::codec_copy::stalled_mirrors()
    );
    for (stage, value) in [
        ("stored", packed::stored_checksum_failures()),
        (
            "in",
            crate::offload::kv_codec::MIRROR_FAILURES.load(Relaxed),
        ),
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
    let _ = writeln!(
        text,
        "flare_kv_codec_gpu_mirror_lanes {}",
        GPU_MIRRORS.load(Relaxed)
    );
    for (tier, ranks) in ["cpu", "gpu"].into_iter().zip(&TOTAL) {
        for (direction, t) in ["in", "out"].into_iter().zip(ranks) {
            for (name, value) in [
                ("original_bytes_total", &t.bytes),
                ("pcie_bytes_total", &t.pcie),
                ("worker_nanoseconds_total", &t.ns),
                ("failed_windows_total", &t.failures),
                ("transform_nanoseconds_total", &t.transform),
                ("packet_header_bytes_total", &t.metadata),
            ] {
                let _ = writeln!(
                    text,
                    "flare_kv_codec_{name}{{placement=\"{tier}\",boundary=\"{direction}\"}} {}",
                    value.load(Relaxed)
                );
            }
        }
    }
    for (i, kind) in ["pinned", "device"].into_iter().enumerate() {
        let _ = writeln!(
            text,
            "flare_kv_codec_scratch_peak_bytes{{kind=\"{kind}\"}} {}",
            HIGH[i].load(Relaxed)
        );
    }
}

pub(super) struct RankCodec {
    pub(super) reserved: [u64; 2],
    gpu_mirror: bool,
    profile: [u8; 32],
    page: Plan,
    row: Plan,
    gpu: Option<GpuCodec>,
    next_gpu_encode: Option<Instant>,
    intake: Option<PinnedHost>,
    restore: PinnedHost,
    restore_alias: u64,
    intake_alias: u64,
}

fn plan(geometry: &CacheGeometry, rank: &RankLayout, row: bool) -> Result<Plan> {
    let mut fields = Vec::new();
    let planes: Vec<_> = if row {
        vec![(&geometry.row, 0)]
    } else {
        geometry
            .planes
            .iter()
            .zip(rank.plane_offsets.iter().copied())
            .collect()
    };
    for (plane, base) in planes {
        for f in geometry
            .codec_fields
            .iter()
            .filter(|f| f.buffer == plane.name)
        {
            let encoding = match f.encoding {
                0 => Encoding::Bytes,
                1 => Encoding::Float32Matrix128,
                2 => Encoding::Bfloat16,
                20 => Encoding::Convolution20,
                32 => Encoding::Convolution32,
                40 => Encoding::Latent528,
                41 => Encoding::Index132,
                42 => Encoding::Fp8Page512,
                45 => Encoding::WindowFp4,
                47 => Encoding::LatentFp4,
                _ => return Err(invalid("unqualified KV field")),
            };
            fields.push(Field {
                offset: (base + f.offset) as usize,
                length: f.length as usize,
                encoding,
            });
        }
    }
    let length = if row {
        geometry.row_record_bytes()
    } else {
        geometry.page_record_bytes()
    };
    Plan::new(length as usize, &fields).map_err(|e| invalid(e.to_string()))
}

#[cfg(all(test, feature = "runtime-core"))]
pub(crate) fn live_scratch() -> [u64; 2] {
    SCRATCH.each_ref().map(|n| n.load(Relaxed))
}

#[cfg(all(test, feature = "runtime-core"))]
pub(crate) fn workspace_parts(g: &CacheGeometry, rank: &RankLayout) -> Result<[u64; 10]> {
    Ok(GpuCodec::workspace_parts(&[
        &plan(g, rank, false)?,
        &plan(g, rank, true)?,
    ]))
}

#[cfg(test)]
#[test]
fn unqualified_layout_is_raw_only_for_an_explicit_codec_request() {
    let mut g = flare_selene::testing::geometry(64, &[("kv", 4096)], 4096);
    for placement in [KvCompression::Cpu, KvCompression::Gpu, KvCompression::Off] {
        g.placement = placement;
        let geom = std::slice::from_ref(&g);
        let policy = mirror_policy(geom, &Layout::new(geom).unwrap()).unwrap();
        assert_eq!(policy.unqualified_layout, placement != KvCompression::Off);
        assert!(policy.backstop.is_none());
    }
}

#[derive(Debug)]
struct CpuBackstop(SyncSender<flare_selene::codec_copy::BackstopJob>);

impl flare_selene::codec_copy::BackstopEncoder for CpuBackstop {
    fn submit(&self, job: flare_selene::codec_copy::BackstopJob) -> Result<bool> {
        match self.0.try_send(job) {
            Ok(()) => Ok(true),
            Err(TrySendError::Full(_)) => Ok(false),
            Err(TrySendError::Disconnected(_)) => Err(invalid("CPU backstop stopped")),
        }
    }
}

pub(super) fn mirror_policy(
    geometry: &[CacheGeometry],
    layout: &Layout,
) -> Result<flare_selene::codec_copy::MirrorPolicy> {
    use flare_selene::codec_copy::{BackstopJob, MirrorPolicy};
    let gpu = geometry.iter().all(|g| g.placement == KvCompression::Gpu);
    let mut policy = MirrorPolicy {
        unqualified_layout: false,
        workspace_unavailable: false,
        gpu,
        backstop: None,
    };
    let Some(profile) = flare_selene::codec_copy::profile(geometry) else {
        policy.unqualified_layout = geometry.iter().any(|g| g.placement != KvCompression::Off);
        return Ok(policy);
    };
    if !gpu {
        return Ok(policy);
    }
    let plans = geometry
        .iter()
        .zip(&layout.ranks)
        .map(|(g, r)| Ok([plan(g, r, false)?, plan(g, r, true)?]))
        .collect::<Result<Vec<_>>>()?;
    let (tx, rx) = mpsc::sync_channel::<BackstopJob>(QUEUE_DEPTH);
    std::thread::Builder::new()
        .name("kv-backstop".into())
        .spawn(move || {
            lower_priority();
            let mut scratch = Vec::new();
            for job in rx {
                let done = job.run(|raw| {
                    let plan = &plans[raw.unit.stored_rank as usize][raw.unit.kind as usize];
                    scratch.resize(plan.original_bytes, 0);
                    let started = Instant::now();
                    let totals = &TOTAL[0][0];
                    let source = raw.data.contiguous();
                    let bytes = plan.encode_verified(&source, &mut scratch).map_err(|e| {
                        totals.failures.fetch_add(1, Relaxed);
                        invalid(e.to_string())
                    })?;
                    let elapsed = started.elapsed().as_nanos() as u64;
                    totals.transform.fetch_add(elapsed, Relaxed);
                    totals.ns.fetch_add(elapsed, Relaxed);
                    if bytes.len() >= raw.data.len() {
                        return Ok(raw.clone());
                    }
                    totals.metadata.fetch_add(plan.metadata() as u64, Relaxed);
                    let mut unit = raw.unit.clone();
                    unit.representation = 1;
                    unit.profile = profile.to_vec();
                    unit.raw_xxh3 = 0;
                    unit.original_crc = packed::crc32(&source);
                    unit.stored_crc = packed::crc32(&bytes);
                    unit.stored_length = bytes.len() as u64;
                    Ok(Packet {
                        unit,
                        data: raw.data.verified(bytes)?,
                    })
                });
                if let Some(bytes) = done {
                    count_encoded_input([bytes, 0]);
                }
            }
        })
        .map_err(|e| Error::caused_by("cannot start the CPU backstop", e))?;
    policy.backstop = Some(Arc::new(CpuBackstop(tx)));
    Ok(policy)
}

impl RankCodec {
    pub(super) fn new(
        context: usize,
        stream: &Stream,
        geometry: &CacheGeometry,
        rank: &RankLayout,
        mirror: bool,
    ) -> Result<Option<Self>> {
        let Some(profile) = geometry.codec_profile() else {
            return Ok(None);
        };
        let page = plan(geometry, rank, false)?;
        let row = plan(geometry, rank, true)?;
        let bytes = page.original_bytes.max(row.original_bytes);
        let gpu = (geometry.placement == KvCompression::Gpu)
            // The grid-limited stream stays in this rank's model context.
            .then(|| GpuCodec::open(context, &[&page, &row]))
            .transpose()
            .map_err(|e| invalid(e.to_string()))?;
        let restore = stream.mapped_host_buffer(bytes)?;
        let restore_alias = stream.mapped_address(&restore)? as u64;
        let intake = (mirror && geometry.placement == KvCompression::Cpu)
            .then(|| stream.mapped_host_buffer(bytes))
            .transpose()?;
        let intake_alias = intake
            .as_ref()
            .map(|h| stream.mapped_address(h))
            .transpose()?
            .unwrap_or(0) as u64;
        let reserved = [
            (bytes * (1 + usize::from(intake.is_some()))) as u64,
            gpu.as_ref().map_or(0, |g| g.reserved),
        ];
        for (i, n) in reserved.into_iter().enumerate() {
            HIGH[i].fetch_max(SCRATCH[i].fetch_add(n, Relaxed) + n, Relaxed);
        }
        let gpu_mirror = mirror && gpu.is_some();
        if gpu_mirror {
            GPU_MIRRORS.fetch_add(1, Relaxed);
        }
        Ok(Some(Self {
            reserved,
            gpu_mirror,
            profile,
            page,
            row,
            gpu,
            next_gpu_encode: None,
            intake,
            restore,
            restore_alias,
            intake_alias,
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
        let direction = usize::from(job.direction == Direction::In);
        let t = &TOTAL[usize::from(self.gpu.is_some())][direction];
        let started = std::time::Instant::now();
        let result = self.copy_window(rank, geometry, layout, stream, event, job, window);
        t.ns.fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
        match result {
            Ok(bytes) => {
                t.pcie.fetch_add(bytes, Relaxed);
                if job.direction == Direction::In {
                    t.bytes.fetch_add(
                        self.page.original_bytes as u64 * job.pages.len() as u64
                            + if job.row.is_some() {
                                self.row.original_bytes as u64
                            } else {
                                0
                            },
                        Relaxed,
                    );
                }
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
        let first = match window.0.as_ref() {
            Action::Mirror { first, .. }
            | Action::RestorePinned { first, .. }
            | Action::Restore { first, .. } => *first,
        };
        let totals =
            &TOTAL[usize::from(self.gpu.is_some())][usize::from(job.direction == Direction::In)];
        let mut emitted = Vec::new();
        let mut transferred = 0;
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
        let mirror = matches!(&*window.0, Action::Mirror { .. });
        for (row, index, id) in ids {
            let plan = if row { &self.row } else { &self.page };
            let n = plan.original_bytes;
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

            if mirror != (job.direction == Direction::Out) {
                return Err(invalid("codec job direction mismatch"));
            }
            if let Action::Mirror { progress, .. } = window.0.as_ref() {
                let mut unit = PackedUnit {
                    kind: i32::from(row),
                    logical_index: index,
                    stored_rank: rank as u32,
                    original_length: n as u64,
                    ..Default::default()
                };
                let literal = matches!(&*window.0, Action::Mirror { literal: true, .. });
                let data: flare_selene::codec_copy::PacketBytes = if literal {
                    self.restore.bytes_mut()[..n].fill(0);
                    {
                        let _pin = progress.source()?;
                        copy(stream, event, &spans, self.restore_alias, true)?;
                    }
                    progress.literal_packet(&self.restore.bytes()[..n])?
                } else if let Some(gpu) = self.gpu.as_mut() {
                    let duty = geometry.gpu_mirror_duty.unwrap_or(25);
                    if !(1..=100).contains(&duty) {
                        return Err(invalid("GPU mirror duty must be in 1..=100"));
                    }
                    if let Some(next) = self.next_gpu_encode {
                        while let Some(remaining) = next.checked_duration_since(Instant::now()) {
                            if progress.cancelled() {
                                return Err(invalid("mirror cancelled"));
                            }
                            std::thread::sleep(remaining.min(std::time::Duration::from_millis(50)));
                        }
                    }
                    let started = Instant::now();
                    // The store pins the source IDs until every rank returns.
                    let result = unsafe {
                        gpu.encode_guarded(plan, &spans, &mut self.restore.bytes_mut()[..n], || {
                            progress.source().map_err(std::io::Error::other)
                        })
                    };
                    let finished = Instant::now();
                    let busy = finished.duration_since(started);
                    let idle = busy * u32::from(100 - duty) / u32::from(duty);
                    self.next_gpu_encode = Some(finished + idle);
                    let result = result.map_err(|e| invalid(e.to_string()))?;
                    totals.transform.fetch_add(gpu.transform_ns(), Relaxed);
                    match result {
                        Some((length, stored, original)) => {
                            unit.stored_crc = stored;
                            unit.original_crc = original;
                            unit.representation = 1;
                            (&self.restore.bytes()[..length]).into()
                        }
                        None => {
                            self.restore.bytes_mut()[..n].fill(0);
                            {
                                let _pin = progress.source()?;
                                copy(stream, event, &spans, self.restore_alias, true)?;
                            }
                            (&self.restore.bytes()[..n]).into()
                        }
                    }
                } else {
                    let intake = self
                        .intake
                        .as_mut()
                        .ok_or_else(|| invalid("encoding is disabled"))?;
                    intake.bytes_mut()[..n].fill(0);
                    {
                        let _pin = progress.source()?;
                        copy(stream, event, &spans, self.intake_alias, true)?;
                    }
                    let raw = &intake.bytes()[..n];
                    let start = Instant::now();
                    let packet = plan
                        .encode_verified(raw, &mut self.restore.bytes_mut()[..n])
                        .map_err(|e| invalid(e.to_string()))?;
                    totals
                        .transform
                        .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
                    if packet.len() < n {
                        unit.representation = 1;
                        unit.original_crc = packed::crc32(raw);
                        unit.stored_crc = packed::crc32(&packet);
                        packet.into()
                    } else {
                        raw.into()
                    }
                };
                unit.stored_length = data.len() as u64;
                if unit.representation == 1 {
                    totals.metadata.fetch_add(plan.metadata() as u64, Relaxed);
                    unit.profile = self.profile.to_vec();
                } else {
                    unit.raw_xxh3 = if literal {
                        checksum(&self.restore.bytes()[..n])
                    } else {
                        checksum(&data.contiguous())
                    };
                }
                transferred += if self.gpu.is_some() {
                    unit.stored_length
                } else {
                    n as u64
                };
                emitted.push(Packet { unit, data });
                progress.advanced();
            } else {
                let selected = |u: &PackedUnit| {
                    u.kind == i32::from(row)
                        && u.logical_index == index
                        && (u.stored_rank == rank as u32 || u.replicas.contains(&(rank as u32)))
                };
                let (u, chunks) = match window.0.as_ref() {
                    Action::Restore { packets, .. } => packets
                        .iter()
                        .find(|p| selected(&p.unit))
                        .map(|p| (&p.unit, p.data.chunks().collect())),
                    Action::RestorePinned { packets, .. } => packets
                        .iter()
                        .find(|p| selected(&p.unit))
                        // The caller owns the slab or network window through the all-rank fence.
                        .map(|p| (&p.unit, unsafe { p.chunks() })),
                    _ => unreachable!(),
                }
                .ok_or_else(|| invalid("missing codec rank share"))?;
                if u.representation > 1
                    || u.original_length != n as u64
                    || u.stored_length != chunks.iter().map(|c| c.len() as u64).sum::<u64>()
                    || (u.representation == 1 && u.profile != self.profile)
                {
                    return Err(invalid("incompatible codec share"));
                }
                if u.representation == 1 {
                    totals.metadata.fetch_add(plan.metadata() as u64, Relaxed);
                }
                if u.representation == 1
                    && let Some(gpu) = self.gpu.as_mut()
                {
                    // The all-rank reply fences exclusive destination IDs.
                    unsafe {
                        gpu.decode_chunks(plan, &chunks, u.stored_crc, u.original_crc, &spans)
                    }
                    .map_err(|e| invalid(e.to_string()))?;
                    totals.transform.fetch_add(gpu.transform_ns(), Relaxed);
                    transferred += u.stored_length;
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
                        if packed::crc32_chunks(&chunks) != u.stored_crc {
                            return Err(invalid("stored codec checksum mismatch"));
                        }
                        plan.decode_chunks(&chunks, raw, true)
                            .map_err(|e| invalid(e.to_string()))?;
                        if packed::crc32(raw) != u.original_crc {
                            crate::offload::kv_codec::ORIGINAL_FAILURES.fetch_add(1, Relaxed);
                            return Err(invalid("original codec checksum mismatch"));
                        }
                        totals
                            .transform
                            .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
                    }
                    copy(stream, event, &spans, self.restore_alias, false)?;
                    transferred += n as u64;
                }
            }
        }
        if let Action::Mirror { ranks, .. } = window.0.as_ref() {
            ranks
                .get(rank)
                .ok_or_else(|| invalid("unknown codec rank"))?
                .set(emitted)
                .map_err(|_| invalid("duplicate rank mirror"))?;
        }
        Ok(transferred)
    }
}

impl Drop for RankCodec {
    fn drop(&mut self) {
        if self.gpu_mirror {
            GPU_MIRRORS.fetch_sub(1, Relaxed);
        }
        for (i, n) in self.reserved.into_iter().enumerate() {
            SCRATCH[i].fetch_sub(n, Relaxed);
        }
    }
}

fn copy(stream: &Stream, event: &Event, spans: &[DeviceSpan], host: u64, out: bool) -> Result<()> {
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
            let (a, b) = (
                p.device + start as u64,
                host + p.offset as u64 + start as u64,
            );
            let (dst, src) = if out { (b, a) } else { (a, b) };
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
    fn placements_round_trip_through_the_rank_worker_and_fence() {
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
            assert_eq!(cudaMalloc(&mut ptr, 4 * n), 0);
            assert_eq!(cuCtxGetCurrent(&mut context), 0);
        }
        let memory = Device(ptr);
        let error = GpuCodec::open(context.wrapping_add(1), &[]).err().unwrap();
        assert!(error.to_string().contains("current model context"));
        let data: Vec<u8> = (0..2 * n)
            .map(|i| ((i / 211 + i / 1024) % 57) as u8)
            .collect();
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
        for (i, address) in [geometry.planes[0].address, geometry.row.address]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                unsafe { cudaMemcpy(address as *mut c_void, data[i * n..].as_ptr().cast(), n, 1) },
                0
            );
        }
        let transfers = |placement| {
            let mut g = geometry.clone();
            g.placement = placement;
            Transfers::start_unmapped(&[(context, 0)], vec![g]).unwrap()
        };
        let mut cpu = transfers(KvCompression::Cpu);
        let mut gpu = transfers(KvCompression::Gpu);
        fn send(t: &mut Transfers, direction: Direction, window: Window) -> Result<u64> {
            let id = u32::from(direction == Direction::In);
            t.submit(Job {
                id: JobId(1),
                window: 0,
                direction,
                codec: Some(window),
                pages: vec![(PageId(id), 0)],
                row: Some((StateSlotId(id), vec![])),
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
        let mut captured = Vec::new();
        for t in [&mut cpu, &mut gpu] {
            let window = Window::mirror(12, 1);
            let moved = send(t, Direction::Out, window.clone()).unwrap();
            let packets = window.captured().unwrap();
            assert!(packets.iter().all(|p| p.unit.representation == 1));
            assert_eq!(
                moved,
                if captured.is_empty() {
                    (2 * n) as u64
                } else {
                    packets.iter().map(|p| p.unit.stored_length).sum()
                }
            );
            captured.push(packets);
        }
        assert_eq!(captured[0], captured[1]);
        for (writer, t) in [(&captured[0], &mut gpu), (&captured[1], &mut cpu)] {
            send(t, Direction::In, Window::restore(12, writer.clone())).unwrap();
            for (i, address) in [geometry.planes[0].address, geometry.row.address]
                .into_iter()
                .enumerate()
            {
                let mut output = vec![0; n];
                assert_eq!(
                    unsafe {
                        cudaMemcpy(
                            output.as_mut_ptr().cast(),
                            (address + n as u64) as *const c_void,
                            n,
                            2,
                        )
                    },
                    0
                );
                assert_eq!(output, &data[i * n..(i + 1) * n]);
            }
            let mut corrupt = writer.clone();
            corrupt[0].unit.representation = 2;
            assert!(send(t, Direction::In, Window::restore(12, corrupt)).is_err());
            assert_eq!(t.in_flight(), 0);
            let mut corrupt = writer.clone();
            corrupt[0].unit.original_crc ^= 1;
            assert!(send(t, Direction::In, Window::restore(12, corrupt)).is_err());
            assert_eq!(t.in_flight(), 0);

            // Keep the expected original CRC and give the damaged packet a valid stored CRC.
            let mut corrupt = writer.clone();
            let mut payload = corrupt[0].data.contiguous().into_owned();
            let frame = u32::from_le_bytes(payload[48..52].try_into().unwrap()) as usize;
            assert_eq!(payload[frame], 2);
            let first_count = frame + 4 + 32;
            payload[first_count] = 0;
            corrupt[0].unit.stored_crc = packed::crc32(&payload);
            corrupt[0].data = payload.into();
            assert!(send(t, Direction::In, Window::restore(12, corrupt)).is_err());
            assert_eq!(t.in_flight(), 0);
        }
        for placement in &TOTAL {
            for direction in placement {
                assert!(direction.transform.load(Relaxed) > 0);
                assert!(direction.metadata.load(Relaxed) > 0);
            }
        }
        let mut off = transfers(KvCompression::Off);
        assert!(off.codec_profile().is_none());
        assert!(off.mirrors.is_empty());
        assert!(off.backstops.is_empty());
        assert!(
            send(
                &mut off,
                Direction::In,
                Window::restore(12, captured[0].clone())
            )
            .is_err()
        );
        let literal = Window::literal(12, 1);
        let encoded_input = TOTAL[0][0].bytes.load(Relaxed);
        send(&mut cpu, Direction::Out, literal.clone()).unwrap();
        assert_eq!(TOTAL[0][0].bytes.load(Relaxed), encoded_input);
        let packets = literal.captured().unwrap();
        for (i, p) in packets.iter().enumerate() {
            assert_eq!(p.unit.representation, 0);
            assert_eq!(p.data.contiguous().as_ref(), &data[i * n..(i + 1) * n]);
        }
        off.close().unwrap();
        cpu.close().unwrap();
        gpu.close().unwrap();
    }
}
