// SPDX-License-Identifier: Apache-2.0

use super::{Action, Packet, Window};
use crate::{
    Result,
    slab::{SlabPool, Slot},
};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

/// Slot capacity borrowed by active CPU backstops in this pool.
#[derive(Debug, Default)]
pub(crate) struct BackstopUsage {
    pub reserved: AtomicU64,
    pub peak: AtomicU64,
    pub intake: AtomicU64,
    pub releases: AtomicU64,
}

pub(super) struct BackstopSlots {
    pool: SlabPool,
    slots: Vec<Slot>,
    copied: AtomicU64,
    next: Mutex<usize>,
}

impl std::fmt::Debug for BackstopSlots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackstopSlots")
            .field("slots", &self.slots.len())
            .finish()
    }
}

impl BackstopSlots {
    pub(super) fn new(pool: &SlabPool, slots: Vec<Slot>) -> Arc<Self> {
        assert!(
            pool.rank_parts().is_none(),
            "backstop slots must be CPU-addressable"
        );
        let bytes = slots.len() as u64 * pool.page_bytes() as u64;
        let used = pool.backstop_usage.reserved.fetch_add(bytes, Relaxed) + bytes;
        pool.backstop_usage.peak.fetch_max(used, Relaxed);
        Arc::new(Self {
            pool: pool.clone(),
            slots,
            copied: AtomicU64::new(0),
            next: Default::default(),
        })
    }

    pub(super) fn append(self: &Arc<Self>, bytes: &[u8]) -> Result<PacketBytes> {
        let capacity = self.slots.len() * self.pool.page_bytes();
        let start = {
            let mut next = self.next.lock().unwrap();
            let start = *next;
            *next = start
                .checked_add(bytes.len())
                .filter(|end| *end <= capacity)
                .ok_or_else(|| crate::geometry::invalid("backstop packet exceeds its slots"))?;
            start
        };
        let mut offset = start;
        let mut rest = bytes;
        while !rest.is_empty() {
            let (address, len) = self.span(offset, rest.len());
            // Each successful reservation has a disjoint range, written before publication.
            unsafe { std::ptr::copy_nonoverlapping(rest.as_ptr(), address as *mut u8, len) };
            offset += len;
            rest = &rest[len..];
        }
        Ok(PacketBytes(Bytes::Slots {
            charge: self.clone(),
            start,
            len: bytes.len(),
        }))
    }

    fn span(&self, offset: usize, len: usize) -> (usize, usize) {
        let width = self.pool.page_bytes();
        let slot = self.slots[offset / width];
        let within = offset % width;
        (
            self.pool.slot_address(slot) + within,
            len.min(width - within),
        )
    }

    pub(super) fn copied(&self, bytes: u64) {
        self.copied.fetch_add(bytes, Relaxed);
        self.pool.backstop_usage.intake.fetch_add(bytes, Relaxed);
    }
}

impl Drop for BackstopSlots {
    fn drop(&mut self) {
        let budget = &self.pool.backstop_usage;
        let bytes = self.slots.len() as u64 * self.pool.page_bytes() as u64;
        budget.intake.fetch_sub(*self.copied.get_mut(), Relaxed);
        budget.reserved.fetch_sub(bytes, Relaxed);
        self.pool.release(self.slots.drain(..));
        budget
            .releases
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}

/// Immutable bytes of a rank packet. A backstop packet retains its borrowed slots.
#[derive(Clone, Debug)]
pub struct PacketBytes(Bytes);

#[derive(Clone, Debug)]
enum Bytes {
    Owned(Arc<[u8]>),
    Slots {
        charge: Arc<BackstopSlots>,
        start: usize,
        len: usize,
    },
}

impl From<Arc<[u8]>> for PacketBytes {
    fn from(bytes: Arc<[u8]>) -> Self {
        Self(Bytes::Owned(bytes))
    }
}
impl From<Vec<u8>> for PacketBytes {
    fn from(bytes: Vec<u8>) -> Self {
        Arc::<[u8]>::from(bytes).into()
    }
}
impl From<&[u8]> for PacketBytes {
    fn from(bytes: &[u8]) -> Self {
        Arc::<[u8]>::from(bytes).into()
    }
}

impl PacketBytes {
    pub fn len(&self) -> usize {
        match &self.0 {
            Bytes::Owned(bytes) => bytes.len(),
            Bytes::Slots { len, .. } => *len,
        }
    }

    fn chunk_at(&self, offset: usize) -> &[u8] {
        match &self.0 {
            Bytes::Owned(bytes) => &bytes[offset..],
            Bytes::Slots { charge, start, len } => {
                if offset == *len {
                    return &[];
                }
                let (address, count) = charge.span(start + offset, len - offset);
                // append initialized this range, which stays immutable until the last packet drops.
                unsafe { std::slice::from_raw_parts(address as *const u8, count) }
            }
        }
    }

    pub fn chunks(&self) -> impl Iterator<Item = &[u8]> {
        let mut offset = 0;
        std::iter::from_fn(move || {
            if offset == self.len() {
                return None;
            }
            let chunk = self.chunk_at(offset);
            offset += chunk.len();
            Some(chunk)
        })
    }

    pub fn contiguous(&self) -> std::borrow::Cow<'_, [u8]> {
        let first = self.chunk_at(0);
        if first.len() == self.len() {
            return std::borrow::Cow::Borrowed(first);
        }
        let mut bytes = Vec::with_capacity(self.len());
        for chunk in self.chunks() {
            bytes.extend_from_slice(chunk);
        }
        std::borrow::Cow::Owned(bytes)
    }

    pub(crate) fn copy_to(&self, mut offset: usize, mut out: &mut [u8]) {
        assert!(out.len() <= self.len() - offset);
        while !out.is_empty() {
            let chunk = self.chunk_at(offset);
            let n = chunk.len().min(out.len());
            out[..n].copy_from_slice(&chunk[..n]);
            offset += n;
            out = &mut out[n..];
        }
    }

    /// Keep verified output in the same capture's allowance. Incompressible units reuse the input.
    pub fn verified(&self, bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() > self.len() {
            return Err(crate::geometry::invalid("backstop output grew"));
        }
        match &self.0 {
            Bytes::Slots { charge, .. } => charge.append(&bytes),
            Bytes::Owned(_) => Ok(bytes.into()),
        }
    }
}

impl PartialEq for PacketBytes {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        let mut offset = 0;
        while offset < self.len() {
            let a = self.chunk_at(offset);
            let b = other.chunk_at(offset);
            let n = a.len().min(b.len());
            if a[..n] != b[..n] {
                return false;
            }
            offset += n;
        }
        true
    }
}

pub trait BackstopEncoder: std::fmt::Debug + Send + Sync {
    /// A full queue leaves the capture with its caller for the next poll.
    fn submit(&self, job: BackstopJob) -> Result<bool>;
}

#[derive(Debug)]
enum State {
    Waiting,
    Encoding(Vec<Packet>),
    Done(Result<Vec<Packet>>),
    Raw,
}

#[derive(Debug)]
struct Progress {
    last: Instant,
    state: State,
}

#[derive(Debug)]
pub struct Backstop {
    pub(super) started: Instant,
    encoder: Arc<dyn BackstopEncoder>,
    progress: Arc<Mutex<Progress>>,
}

impl Backstop {
    pub fn new(encoder: Arc<dyn BackstopEncoder>, now: Instant) -> Self {
        Self {
            started: now,
            encoder,
            progress: Arc::new(Mutex::new(Progress {
                last: now,
                state: State::Waiting,
            })),
        }
    }

    pub fn copied(&self, now: Instant) {
        self.progress.lock().unwrap().last = now;
    }

    /// Called after every rank's raw D2H fence. Timeout can now publish a complete literal mirror.
    pub fn ready(&self, windows: &[Window], now: Instant) -> bool {
        let mut p = self.progress.lock().unwrap();
        if matches!(p.state, State::Done(_) | State::Raw) {
            return true;
        }
        if now.saturating_duration_since(p.last) >= super::STALL_LIMIT {
            p.state = State::Raw;
            return true;
        }
        if matches!(p.state, State::Waiting) {
            let job = BackstopJob {
                windows: windows.iter().map(|w| Arc::downgrade(&w.0)).collect(),
                progress: Arc::downgrade(&self.progress),
            };
            match self.encoder.submit(job) {
                Ok(true) => p.state = State::Encoding(Vec::new()),
                Ok(false) => {}
                Err(e) => {
                    p.state = State::Done(Err(e));
                    return true;
                }
            }
        }
        false
    }

    pub fn result(&self) -> Result<Option<Vec<Packet>>> {
        match &mut self.progress.lock().unwrap().state {
            State::Raw => Ok(None),
            State::Done(result) => result
                .as_mut()
                .map(|v| Some(std::mem::take(v)))
                .map_err(|e| e.clone()),
            _ => Err(crate::geometry::invalid(
                "backstop mirror is still encoding",
            )),
        }
    }
}

impl Drop for Backstop {
    fn drop(&mut self) {
        self.progress.lock().unwrap().state = State::Raw;
    }
}

/// The worker borrows owned host bytes. A late reply cannot write a published or recycled slab.
#[derive(Debug)]
pub struct BackstopJob {
    windows: Vec<Weak<Action>>,
    progress: Weak<Mutex<Progress>>,
}

impl BackstopJob {
    pub fn run(self, mut encode: impl FnMut(&Packet) -> Result<Packet>) -> Option<u64> {
        let progress = self.progress.upgrade()?;
        let mut bytes = 0;
        for window in self.windows {
            let raw = Window(window.upgrade()?).captured();
            let packets = match raw {
                Ok(packets) => packets,
                Err(e) => {
                    finish_error(&progress, e);
                    return None;
                }
            };
            for packet in packets {
                if !matches!(progress.lock().unwrap().state, State::Encoding(_)) {
                    return None;
                }
                let verified = match encode(&packet) {
                    Ok(packet) => packet,
                    Err(e) => {
                        finish_error(&progress, e);
                        return None;
                    }
                };
                let mut p = progress.lock().unwrap();
                let State::Encoding(output) = &mut p.state else {
                    return None;
                };
                bytes += packet.unit.original_length;
                output.push(verified);
                p.last = Instant::now();
            }
        }
        let mut p = progress.lock().unwrap();
        let State::Encoding(output) = &mut p.state else {
            return None;
        };
        p.state = State::Done(Ok(std::mem::take(output)));
        Some(bytes)
    }
}

fn finish_error(progress: &Mutex<Progress>, error: crate::Error) {
    let mut p = progress.lock().unwrap();
    if matches!(p.state, State::Encoding(_)) {
        p.state = State::Done(Err(error));
    }
}
