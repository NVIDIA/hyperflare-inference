// SPDX-License-Identifier: Apache-2.0

//! Stored rank shares passed between a host store and its device copier.
use crate::{
    Result,
    geometry::invalid,
    shared::{
        packed,
        wire::{PackedUnit, PackedWindow},
    },
};
use std::sync::{Arc, OnceLock};

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
mod backstop;
use crate::StoreStats;
pub use backstop::Backstop;
pub(crate) use backstop::BackstopUsage;
pub use backstop::{BackstopEncoder, BackstopJob, PacketBytes};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default)]
pub struct MirrorPolicy {
    pub unqualified_layout: bool,
    pub workspace_unavailable: bool,
    pub gpu: bool,
    pub backstop: Option<Arc<dyn BackstopEncoder>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Encoder {
    Gpu,
    Backstop,
    Cpu,
    #[default]
    Raw,
}

pub type PeerCapability = Arc<std::sync::atomic::AtomicBool>;

/// The first capture owns the lane. Only captures waiting behind it consume the queue budget.
pub fn queue_fallback(
    pending: impl Iterator<Item = (std::time::Instant, u64)>,
    incoming: u64,
    now: std::time::Instant,
    ranks: usize,
) -> Option<usize> {
    queue_limit(
        pending,
        incoming,
        now,
        MAX_PENDING_BYTES.saturating_mul(ranks as u64),
        MAX_AGE,
    )
}

fn queue_limit(
    pending: impl Iterator<Item = (Instant, u64)>,
    incoming: u64,
    now: Instant,
    bytes: u64,
    age: Duration,
) -> Option<usize> {
    let mut first = None;
    let mut second = None;
    let mut total = 0u64;
    for (time, bytes) in pending {
        total = total.saturating_add(bytes);
        if first.is_none_or(|(t, _)| time < t) {
            second = first.map(|(t, _)| t);
            first = Some((time, bytes));
        } else if second.is_none_or(|t| time < t) {
            second = Some(time);
        }
    }
    let (_, active) = first?;
    if total.saturating_sub(active).saturating_add(incoming) > bytes {
        return Some(0);
    }
    second
        .filter(|t| now.saturating_duration_since(*t) >= age)
        .map(|_| 1)
}

/// Reserve row entries even for a page-only mirror so a later row refill fits.
/// Each share needs at most 96 protobuf bytes including its containing field.
/// Replica entries replace full share entries and cannot increase this bound.
pub fn capture_fits(pages: usize, ranks: usize) -> bool {
    let units = pages.saturating_add(1).saturating_mul(ranks);
    units <= packed::MAX_UNITS
        && units.saturating_mul(96).saturating_add(11) <= packed::MAX_DIRECTORY
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RawMirror {
    QueueBytes,
    QueueAge,
    IntakeSlots,
    RankFiles,
    PeerCapability,
    OfferBounds,
    RawRefill,
    BackstopRaw,
    UnqualifiedLayout,
    BackstopBudget,
    WorkspaceUnavailable,
}

/// Original bytes in the GPU windows of one capture, before replica or codec compression.
#[derive(Debug)]
pub(crate) struct MirrorBytes {
    pages: usize,
    page_bytes: u64,
    row_bytes: u64,
    pub fallback: Option<RawMirror>,
    counted: bool,
    pub encoder: Encoder,
    backstop: Option<Backstop>,
    copied: u64,
}

impl MirrorBytes {
    pub fn new(pages: usize, page_bytes: u64, row_bytes: u64) -> Self {
        Self {
            pages,
            page_bytes,
            row_bytes,
            fallback: None,
            counted: false,
            encoder: Encoder::Raw,
            backstop: None,
            copied: 0,
        }
    }

    pub fn with_policy(mut self, policy: &MirrorPolicy) -> Self {
        if policy.workspace_unavailable {
            self.fallback = Some(RawMirror::WorkspaceUnavailable);
        } else if policy.unqualified_layout {
            self.fallback = Some(RawMirror::UnqualifiedLayout);
        }
        self
    }

    pub fn total(&self) -> u64 {
        self.pages as u64 * self.page_bytes + self.row_bytes
    }

    pub fn choose(
        &mut self,
        peer: Option<&PeerCapability>,
        ranks: usize,
        policy: &MirrorPolicy,
        pending: impl Iterator<Item = (Instant, u64, Encoder)> + Clone,
        now: std::time::Instant,
    ) {
        self.fallback = if !capture_fits(self.pages, ranks) {
            Some(RawMirror::OfferBounds)
        } else if peer.is_some_and(|p| !p.load(std::sync::atomic::Ordering::Acquire)) {
            Some(RawMirror::PeerCapability)
        } else {
            self.encoder = if policy.gpu {
                Encoder::Gpu
            } else {
                Encoder::Cpu
            };
            let matching = |lane: Encoder| {
                pending
                    .clone()
                    .filter(move |p| p.2 == lane)
                    .map(|(t, b, _)| (t, b))
            };
            let mut refused = queue_fallback(matching(self.encoder), self.total(), now, ranks);
            if refused.is_some()
                && let Some(encoder) = &policy.backstop
            {
                refused = queue_limit(
                    matching(Encoder::Backstop),
                    self.total(),
                    now,
                    BACKSTOP_BYTES,
                    BACKSTOP_AGE,
                );
                if refused.is_none() {
                    self.encoder = Encoder::Backstop;
                    self.backstop = Some(Backstop::new(encoder.clone(), now));
                }
            }
            refused.map(|i| {
                if i == 0 {
                    RawMirror::QueueBytes
                } else {
                    RawMirror::QueueAge
                }
            })
        };
        if self.fallback.is_some() {
            self.encoder = Encoder::Raw;
        }
    }

    pub(crate) fn record_pin_time(&self, since: Instant, stats: &mut StoreStats) {
        let elapsed = since.elapsed();
        stats.codec_pin_ns += elapsed.as_nanos() as u64;
        for (bucket, ms) in stats.codec_pin_buckets.iter_mut().zip(PIN_MS) {
            *bucket += u64::from(elapsed <= Duration::from_millis(ms));
        }
        stats.codec_pin_buckets[PIN_MS.len()] += 1;
    }

    pub(crate) fn copy_progress(
        &mut self,
        pool: &crate::slab::SlabPool,
        allocate: impl FnOnce(usize) -> Option<Vec<crate::slab::Slot>>,
    ) -> Arc<MirrorProgress> {
        if self.backstop.is_none() {
            return Arc::default();
        }
        let slots = self
            .total()
            .checked_mul(2)
            .and_then(|bytes| usize::try_from(bytes.div_ceil(pool.page_bytes() as u64)).ok())
            .and_then(allocate);
        if let Some(slots) = slots {
            Arc::new(MirrorProgress(
                Default::default(),
                Some(backstop::BackstopSlots::new(pool, slots)),
            ))
        } else {
            self.fallback = Some(RawMirror::BackstopBudget);
            self.intake_full();
            Arc::default()
        }
    }

    pub fn intake_full(&mut self) {
        self.backstop = None;
        self.encoder = Encoder::Raw;
        self.fallback.get_or_insert(RawMirror::IntakeSlots);
    }

    pub(crate) fn abandon_backstop(&mut self, stats: &mut StoreStats) -> bool {
        let abandoned = self.backstop.take().is_some();
        stats.backstop.abandoned += u64::from(abandoned);
        abandoned
    }

    pub fn admitted(&self, stats: &mut StoreStats) {
        if self.backstop.is_some() {
            stats.backstop.events += 1;
            stats.backstop.bytes += self.total();
        }
    }

    pub fn ready(&self, windows: &[Window], now: Instant) -> bool {
        self.backstop.as_ref().is_none_or(|b| b.ready(windows, now))
    }

    pub fn packets(&mut self, windows: &[Window], stats: &mut StoreStats) -> Result<Vec<Packet>> {
        if let Some(b) = &self.backstop {
            let packets = b.result()?;
            if packets.is_none() {
                self.fallback = Some(RawMirror::BackstopRaw);
            }
            self.account(stats, self.copied);
            if let Some(packets) = packets {
                return Ok(packets);
            }
        }
        windows.iter().try_fold(Vec::new(), |mut all, w| {
            all.extend(w.captured()?);
            Ok(all)
        })
    }

    pub fn published(&self, stats: &mut StoreStats) {
        if let Some(backstop) = &self.backstop {
            let elapsed = backstop.started.elapsed();
            stats.backstop.ready_ns += elapsed.as_nanos() as u64;
            for (i, bound) in READY_MS.iter().enumerate() {
                stats.backstop.ready_buckets[i] +=
                    u64::from(elapsed <= Duration::from_millis(*bound));
            }
            stats.backstop.ready_buckets[READY_MS.len()] += 1;
        }
    }

    pub fn completed(&mut self, stats: &mut StoreStats, window: u32) {
        let first = window as usize * crate::slab::WINDOW_PAGES;
        let end = (first + crate::slab::WINDOW_PAGES).min(self.pages);
        let bytes = (end - first) as u64 * self.page_bytes
            + if end == self.pages { self.row_bytes } else { 0 };
        if self.encoder == Encoder::Backstop {
            if let Some(b) = &self.backstop {
                b.copied(Instant::now());
                self.copied += bytes;
            }
        } else {
            self.account(stats, bytes);
        }
    }

    fn account(&mut self, stats: &mut StoreStats, bytes: u64) {
        let encoder = if self.fallback.is_some() {
            Encoder::Raw
        } else {
            self.encoder
        };
        stats.backstop.by_encoder[encoder as usize] += bytes;
        stats.codec_mirrored_original_bytes += bytes;
        if let Some(cause) = self.fallback {
            let i = cause as usize;
            stats.codec_fallback_bytes[i] += bytes;
            stats.codec_fallback_events[i] += u64::from(!self.counted);
            self.counted = true;
        }
    }
}

/// Queued allowance per rank, beyond the capture already being served.
pub const MAX_PENDING_BYTES: u64 = 384 << 20;
pub const MAX_AGE: std::time::Duration = std::time::Duration::from_secs(8);
// The sixteen IN workers share this allowance across all ranks of one copier.
pub const BACKSTOP_BYTES: u64 = 256 << 20;
pub const BACKSTOP_AGE: Duration = Duration::from_secs(4);
pub const STALL_LIMIT: std::time::Duration = std::time::Duration::from_secs(30);
static STALLS: AtomicU64 = AtomicU64::new(0);
pub fn stalled_mirrors() -> u64 {
    STALLS.load(Relaxed)
}

#[derive(Debug, Default)]
pub struct MirrorProgress(
    std::sync::Mutex<Progress>,
    Option<Arc<backstop::BackstopSlots>>,
);
#[derive(Debug, Default)]
struct Progress {
    last: Option<std::time::Instant>,
    readers: usize,
    cancelled: bool,
}
impl MirrorProgress {
    pub fn literal_packet(&self, bytes: &[u8]) -> Result<PacketBytes> {
        if let Some(charge) = &self.1 {
            let packet = charge.append(bytes)?;
            charge.copied(bytes.len() as u64);
            Ok(packet)
        } else {
            Ok(bytes.into())
        }
    }

    pub fn submitted(&self) {
        self.0
            .lock()
            .unwrap()
            .last
            .get_or_insert_with(std::time::Instant::now);
    }
    pub fn source(&self) -> Result<SourceRead<'_>> {
        let mut state = self.0.lock().unwrap();
        if state.cancelled {
            return Err(invalid("mirror stalled"));
        }
        state.readers += 1;
        Ok(SourceRead(self))
    }
    pub fn cancelled(&self) -> bool {
        self.0.lock().unwrap().cancelled
    }
    pub fn advanced(&self) {
        self.0.lock().unwrap().last = Some(std::time::Instant::now());
    }
    pub fn can_release(&self, now: std::time::Instant) -> bool {
        let mut state = self.0.lock().unwrap();
        if !state.cancelled
            && state
                .last
                .is_some_and(|t| now.saturating_duration_since(t) >= STALL_LIMIT)
        {
            state.cancelled = true;
            STALLS.fetch_add(1, Relaxed);
        }
        state.cancelled && state.readers == 0
    }
}
pub struct SourceRead<'a>(&'a MirrorProgress);
impl Drop for SourceRead<'_> {
    fn drop(&mut self) {
        self.0.0.lock().unwrap().readers -= 1;
    }
}

pub const READY_MS: [u64; 6] = [50, 250, 1000, 4000, 8000, 30000];
pub const PIN_MS: [u64; 6] = [1, 10, 50, 100, 500, 1000];

#[derive(Clone, Debug, PartialEq)]
pub struct Packet {
    pub unit: PackedUnit,
    pub data: PacketBytes,
}

impl Packet {
    pub fn verified_window(
        expected: &[PackedUnit],
        window: &PackedWindow,
        bytes: &[u8],
    ) -> Result<Vec<Self>> {
        packed::verify_window(expected, window, bytes)?;
        Self::from_bytes(expected, bytes)
    }
    pub(crate) fn from_bytes(expected: &[PackedUnit], bytes: &[u8]) -> Result<Vec<Self>> {
        let mut rest = bytes;
        let packets = expected
            .iter()
            .map(|unit| {
                let length = usize::try_from(unit.stored_length)
                    .map_err(|_| invalid("oversized codec unit"))?;
                let (data, tail) = rest
                    .split_at_checked(length)
                    .ok_or_else(|| invalid("short codec window"))?;
                rest = tail;
                Ok(Self {
                    unit: unit.clone(),
                    data: data.into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if !rest.is_empty() {
            return Err(invalid("trailing codec window bytes"));
        }
        Ok(packets)
    }
}

#[derive(Clone, Debug)]
pub struct PinnedPacket {
    pub unit: PackedUnit,
    pub(crate) spans: Vec<(usize, usize)>,
}

impl PinnedPacket {
    /// # Safety
    /// The ranges must stay registered and immutable through the last copier fence.
    pub unsafe fn from_registered(unit: PackedUnit, spans: Vec<(usize, usize)>) -> Self {
        Self { unit, spans }
    }
    /// # Safety
    /// The copier must still own the immutable slab pin or transport window lease.
    pub unsafe fn chunks(&self) -> Vec<&[u8]> {
        self.spans
            .iter()
            .map(|&(address, length)| unsafe {
                std::slice::from_raw_parts(address as *const u8, length)
            })
            .collect()
    }
}

#[derive(Debug)]
pub enum Action {
    Mirror {
        literal: bool,
        backstop: bool,
        progress: Arc<MirrorProgress>,
        first: u64,
        ranks: Vec<OnceLock<Vec<Packet>>>,
    },
    RestorePinned {
        first: u64,
        packets: Vec<PinnedPacket>,
    },
    Restore {
        first: u64,
        packets: Vec<Packet>,
    },
}

#[derive(Clone, Debug)]
pub struct Window(pub Arc<Action>);
impl PartialEq for Window {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Window {}
impl Window {
    pub fn mirror(first: u64, ranks: usize) -> Self {
        Self::capture(first, ranks, false, Arc::default())
    }
    pub fn literal(first: u64, ranks: usize) -> Self {
        Self::capture(first, ranks, true, Arc::default())
    }
    pub(crate) fn capture(
        first: u64,
        ranks: usize,
        literal: bool,
        progress: Arc<MirrorProgress>,
    ) -> Self {
        Self::capture_on(first, ranks, literal, progress, false)
    }
    pub fn capture_on(
        first: u64,
        ranks: usize,
        literal: bool,
        progress: Arc<MirrorProgress>,
        backstop: bool,
    ) -> Self {
        Self(Arc::new(Action::Mirror {
            backstop,
            literal: literal || backstop,
            progress,
            first,
            ranks: (0..ranks).map(|_| OnceLock::new()).collect(),
        }))
    }
    pub fn restore(first: u64, packets: Vec<Packet>) -> Self {
        Self(Arc::new(Action::Restore { first, packets }))
    }
    pub(crate) fn restore_pinned(first: u64, packets: Vec<PinnedPacket>) -> Self {
        Self(Arc::new(Action::RestorePinned { first, packets }))
    }
    pub fn captured(&self) -> Result<Vec<Packet>> {
        let Action::Mirror { ranks, .. } = self.0.as_ref() else {
            return Err(invalid("expected a mirror window"));
        };
        let mut packets = Vec::new();
        for rank in ranks {
            packets.extend(
                rank.get()
                    .ok_or_else(|| invalid("missing rank mirror"))?
                    .iter()
                    .cloned(),
            );
        }
        packets.sort_by_key(|p| (p.unit.kind, p.unit.logical_index, p.unit.stored_rank));
        Ok(packets)
    }
}

pub fn profile(geometry: &[crate::CacheGeometry]) -> Option<[u8; 32]> {
    if geometry
        .iter()
        .any(|rank| rank.placement == crate::offload::KvCompression::Off)
    {
        return None;
    }
    checked_profile(geometry).ok()
}

pub fn checked_profile(geometry: &[crate::CacheGeometry]) -> std::result::Result<[u8; 32], String> {
    let mut expected = None;
    for (rank, g) in geometry.iter().enumerate() {
        let actual = g
            .checked_codec_profile()
            .map_err(|e| format!("rank {rank}: {e}"))?;
        if expected.is_some_and(|p| p != actual) {
            return Err(format!(
                "codec field layout on rank {rank} differs from rank 0"
            ));
        }
        expected = Some(actual);
    }
    expected.ok_or_else(|| "no cache ranks".to_owned())
}

pub fn reader_caps(profile: Option<[u8; 32]>, rdma: bool) -> crate::shared::wire::CodecCaps {
    let mut caps = packed::raw_reader();
    if let Some(profile) = profile.filter(|_| !rdma) {
        caps.receive_profiles.push(profile.to_vec());
    }
    caps
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    #[test]
    fn even_an_empty_backstop_lane_must_fit_its_heap_reservation() {
        #[derive(Debug)]
        struct EncoderStub;
        impl BackstopEncoder for EncoderStub {
            fn submit(&self, _: BackstopJob) -> Result<bool> {
                Ok(false)
            }
        }
        let pool = crate::slab::SlabPool::new(4096, 2, 0).unwrap();
        let heap = &pool.backstop_usage;
        let admit = |bytes| {
            let mut t = MirrorBytes::new(0, 1, bytes);
            t.encoder = Encoder::Backstop;
            t.backstop = Some(Backstop::new(Arc::new(EncoderStub), Instant::now()));
            let progress = t.copy_progress(&pool, |count| pool.allocate(count));
            (t, progress)
        };
        assert_eq!(admit(4097).0.fallback, Some(RawMirror::BackstopBudget));
        let (t, rank) = admit(4095);
        assert_eq!(t.fallback, None);
        let packet = rank.literal_packet(&[7; 100]).unwrap();
        drop(packet);
        drop(t);
        assert_eq!(admit(1).0.fallback, Some(RawMirror::BackstopBudget));
        assert_eq!(pool.free_slots(), 0);
        assert_eq!(heap.peak.load(Relaxed), 8192);
        assert_eq!(heap.intake.load(Relaxed), 100);
        drop(rank);
        assert_eq!(heap.intake.load(Relaxed), 0);
        assert_eq!(pool.free_slots(), 2);
        assert_eq!(heap.peak.load(Relaxed), 8192);
        assert_eq!(admit(4096).0.fallback, None);
    }

    #[test]
    fn raw_and_verified_packets_occupy_the_borrowed_slots_until_the_last_reader() {
        use crate::slab::SlabPool;
        let pool = SlabPool::new(4096, 8, 0).unwrap();
        let all = pool.allocate(8).unwrap();
        let loan = [all[6], all[1], all[4], all[0]];
        pool.release(loan);
        let progress = MirrorProgress(
            Default::default(),
            Some(backstop::BackstopSlots::new(
                &pool,
                pool.allocate(4).unwrap(),
            )),
        );
        let original: Vec<_> = (0..8192).map(|i| (i % 239) as u8).collect();
        let raw = progress.literal_packet(&original).unwrap();
        let verified = raw.verified(vec![0x5d; 5000]).unwrap();
        let addresses =
            |p: &PacketBytes| p.chunks().map(|c| c.as_ptr() as usize).collect::<Vec<_>>();
        assert_eq!(addresses(&raw), pool.addresses(&loan[..2]));
        assert_eq!(addresses(&verified), pool.addresses(&loan[2..]));
        assert_eq!(raw.contiguous().as_ref(), original);
        assert!(verified.chunks().flatten().all(|&byte| byte == 0x5d));
        assert!(raw.verified(vec![1; 8193]).is_err());
        drop(raw);
        drop(progress);
        assert_eq!(pool.free_slots(), 0);
        drop(verified);
        assert_eq!(pool.free_slots(), 4);
        assert_eq!(pool.backstop_usage.peak.load(Relaxed), 16384);
    }

    #[test]
    fn oversized_first_capture_and_progressing_head_are_admitted_whole() {
        let t = std::time::Instant::now();
        let large = MAX_PENDING_BYTES * 3;
        assert_eq!(queue_fallback(std::iter::empty(), large, t, 1), None);
        assert_eq!(
            queue_fallback([(t, large)].into_iter(), 1, t + MAX_AGE * 10, 1),
            None
        );
        assert_eq!(
            queue_fallback([(t, large), (t, 1)].into_iter(), 1, t + MAX_AGE * 10, 1),
            Some(1)
        );
    }
    #[test]
    fn backstop_bytes_and_age_are_independent_of_the_gpu_lane() {
        #[derive(Debug)]
        struct EncoderStub;
        impl BackstopEncoder for EncoderStub {
            fn submit(&self, _: BackstopJob) -> Result<bool> {
                unreachable!()
            }
        }
        let policy = MirrorPolicy {
            unqualified_layout: false,
            workspace_unavailable: false,
            gpu: true,
            backstop: Some(Arc::new(EncoderStub)),
        };
        let start = Instant::now();
        let now = start + MAX_AGE;
        let choose = |pending: Vec<(Instant, u64, Encoder)>, size| {
            let mut t = MirrorBytes::new(0, 1, size);
            t.choose(None, 1, &policy, pending.into_iter(), now);
            (t.encoder, t.fallback)
        };
        let gpu = (start, MAX_PENDING_BYTES, Encoder::Gpu);
        let cpu = (start, BACKSTOP_BYTES, Encoder::Backstop);
        let disabled = MirrorPolicy {
            unqualified_layout: false,
            workspace_unavailable: false,
            gpu: true,
            backstop: None,
        };
        let mut raw = MirrorBytes::new(0, 1, 1);
        raw.choose(
            None,
            1,
            &disabled,
            [gpu, (start, 1, Encoder::Gpu)].into_iter(),
            now,
        );
        assert_eq!(
            (raw.encoder, raw.fallback),
            (Encoder::Raw, Some(RawMirror::QueueAge))
        );
        assert_eq!(
            choose(vec![cpu, cpu], MAX_PENDING_BYTES + 1),
            (Encoder::Gpu, None)
        );
        assert_eq!(
            choose(vec![gpu], MAX_PENDING_BYTES + 1),
            (Encoder::Backstop, None)
        );
        assert_eq!(
            choose(vec![gpu, gpu, cpu], BACKSTOP_BYTES),
            (Encoder::Backstop, None)
        );
        assert_eq!(
            choose(vec![gpu, gpu, cpu], BACKSTOP_BYTES + 1),
            (Encoder::Raw, Some(RawMirror::QueueBytes))
        );
        assert_eq!(
            choose(
                vec![gpu, gpu, cpu, (now - BACKSTOP_AGE, 1, Encoder::Backstop)],
                1
            ),
            (Encoder::Raw, Some(RawMirror::QueueAge))
        );
    }

    #[test]
    fn a_stalled_mirror_closes_new_source_reads_and_waits_for_existing_dma() {
        let state = MirrorProgress::default();
        state.submitted();
        let dma = state.source().unwrap();
        let expired = std::time::Instant::now() + STALL_LIMIT;
        assert!(!state.can_release(expired));
        assert!(state.source().is_err());
        drop(dma);
        assert!(state.can_release(expired));
    }
    #[test]
    fn capture_checks_the_directory_before_issuing_codec_jobs() {
        assert!(capture_fits(4095, 4));
        assert!(!capture_fits(4096, 4));
        assert!(!capture_fits(usize::MAX, 4));
    }
}
