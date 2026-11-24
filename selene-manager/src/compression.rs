// SPDX-License-Identifier: Apache-2.0

use crate::{
    Key,
    arena::Extent,
    record::{self, Descriptor, NamespaceId},
    store::Namespace,
};
use flare_selene::geometry::padded_bytes;
use flare_selene::shared::{
    codec::Plan,
    packed,
    record::{HEADER_BYTES, Header},
    wire::packed_unit::{Kind, Representation},
    wire::{CodecCaps, CodecLayout, PackedOffer, PackedUnit},
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fmt::Write,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc,
    },
    time::{Duration, Instant},
};

type Profile = [u8; 32];
type Plans = Arc<[Plan; 2]>;
pub(crate) const WAIT: Duration = Duration::from_millis(250);
pub(crate) const MIN_RATE: f64 = 500_000_000.0; // Raw record bytes per second.

#[derive(Default)]
pub(crate) struct Profiles {
    plans: HashMap<Profile, Plans>,
    sessions: HashMap<u64, (NamespaceId, Profile)>,
    last: HashMap<NamespaceId, Profile>,
}

impl Profiles {
    #[cfg(test)]
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    pub fn offer(
        &mut self,
        session: u64,
        descriptor: &Descriptor,
        layout: CodecLayout,
        caps: Option<&CodecCaps>,
    ) -> Result<(), String> {
        if layout.revision != flare_selene::shared::codec_layout::REVISION_CPU {
            // Non-CPU codec profiles (such as GKV1 GPU-edge codecs) are encoded by the
            // writer on VRAM -> DRAM capture and stored verbatim in DRAM/NVMe; manager
            // CPU write-back only encodes ZKV1 records.
            return Ok(());
        }
        let (plans, profile) = layout.plans_and_profile().map_err(|error| error.to_string())?;
        if !caps.is_some_and(|c| {
            c.envelope_revision == packed::REVISION
                && c.receive_profiles.iter().any(|p| p.as_slice() == profile)
        }) {
            return Err("the session does not offer the layout's profile".into());
        }
        if layout.page_tokens != descriptor.page_tokens {
            return Err(format!(
                "layout page tokens {} differ from the namespace's {}",
                layout.page_tokens, descriptor.page_tokens
            ));
        }
        // Records store each plane and the row padded to a block. Flare's session geometry is already
        // padded while the layout carries unpadded unit sizes, so compare the padded sizes.
        if let Some(rank) = descriptor.ranks.iter().position(|rank| {
            padded_bytes(rank.row_bytes) != layout.row.as_ref().and_then(|p| padded_bytes(p.unit_bytes))
                || !rank
                    .planes
                    .iter()
                    .map(|&bytes| padded_bytes(bytes))
                    .eq(layout.planes.iter().map(|p| padded_bytes(p.unit_bytes)))
        }) {
            return Err(format!(
                "layout unit sizes differ from rank {rank}'s geometry"
            ));
        }
        if let std::collections::hash_map::Entry::Vacant(entry) = self.plans.entry(profile) {
            entry.insert(Arc::new(plans));
        }
        let namespace = descriptor.namespace_id();
        let previous = self.last.insert(namespace, profile);
        let session_offer = self.sessions.insert(session, (namespace, profile));
        for retired in previous.into_iter().chain(session_offer.map(|(_, p)| p)) {
            if retired != profile {
                self.release_if_unused(retired);
            }
        }
        Ok(())
    }

    pub fn withdraw(&mut self, session: u64) -> Option<NamespaceId> {
        let (namespace, profile) = self.sessions.remove(&session)?;
        self.release_if_unused(profile);
        Some(namespace)
    }

    pub fn release_if_empty(&mut self, id: NamespaceId, namespace: &Namespace) {
        // A write in flight pins its record. Eviction cancels queued writes for removed records.
        if namespace.tree.is_empty()
            && !self.sessions.values().any(|(ns, _)| *ns == id)
            && let Some(profile) = self.last.remove(&id)
        {
            self.release_if_unused(profile);
        }
    }

    fn release_if_unused(&mut self, profile: Profile) {
        if !self.last.values().any(|p| *p == profile)
            && !self.sessions.values().any(|(_, p)| *p == profile)
        {
            self.plans.remove(&profile);
        }
    }

    pub fn for_chain(
        &self,
        namespace: &Namespace,
        id: NamespaceId,
        key: Key,
    ) -> Option<(Profile, Plans)> {
        let tree = &namespace.tree;
        let ancestors = |parent| std::iter::successors(parent, |k| tree[k].parent);
        let mut required = None;
        let mut constrain = |layout: &packed::Layout| {
            for unit in &layout.offer().units {
                if unit.representation == Representation::Packet as i32 {
                    let profile: Profile = unit.profile.as_slice().try_into().ok()?;
                    if required.replace(profile).is_some_and(|p| p != profile) {
                        return None;
                    }
                }
            }
            Some(())
        };
        let node = &tree[&key];
        if let Some(layout) = ancestors(node.parent)
            .find_map(|k| tree[&k].location.disk_header.as_ref()?.packed.as_deref())
        {
            constrain(layout)?;
        }
        if node.children() > 0 {
            // Replacing a durable record makes its whole subtree undurable.
            for descendant in namespace.undurable.iter().filter_map(|k| tree.node(k)) {
                let location = &descendant.location;
                let layout = location
                    .disk_header
                    .as_ref()
                    .and_then(|h| h.packed.as_deref());
                if (layout.is_some() || (location.writing && location.disk_header.is_none()))
                    && ancestors(descendant.parent)
                        .take_while(|p| p.consumed_tokens >= key.consumed_tokens)
                        .any(|p| p == key)
                {
                    // An in-flight write can still publish a packed file.
                    constrain(layout?)?;
                }
            }
        }
        let profile = self
            .sessions
            .values()
            .filter(|(ns, p)| *ns == id && required.is_none_or(|r| r == *p))
            .map(|(_, p)| *p)
            .min()
            .or_else(|| {
                self.last
                    .get(&id)
                    .copied()
                    .filter(|p| required.is_none_or(|r| r == *p))
            })?;
        Some((profile, self.plans.get(&profile)?.clone()))
    }
}

pub(crate) struct PackedRecord {
    pub header: Header,
    storage: Reservation,
    length: usize,
}

impl PackedRecord {
    pub fn bytes(&self) -> &[u8] {
        &self.storage.buffer[..self.length]
    }
}

struct TempBudget {
    available: usize,
    cached: Vec<Vec<u8>>,
    cached_bytes: usize,
}

struct Reservation {
    budget: Arc<Mutex<TempBudget>>,
    charge: usize,
    buffer: Vec<u8>,
}

impl Reservation {
    fn take(
        budget: &Arc<Mutex<TempBudget>>,
        charge: usize,
        length: usize,
    ) -> Result<Self, &'static str> {
        debug_assert!(charge >= length);
        let mut state = budget.lock().unwrap();
        state.available = state.available.checked_sub(charge).ok_or("temp_budget")?;
        let index = state.cached.iter().position(|b| b.capacity() == length);
        let cached = index.map(|i| {
            let buffer = state.cached.swap_remove(i);
            state.cached_bytes -= buffer.capacity();
            buffer
        });
        // Idle buffers occupy only the part of the budget that no job has reserved.
        while state.cached_bytes > state.available {
            state.cached_bytes -= state.cached.pop().unwrap().capacity();
        }
        drop(state);
        let buffer = cached.unwrap_or_else(|| vec![0; length]);
        Ok(Self {
            budget: budget.clone(),
            charge,
            buffer,
        })
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut state = self.budget.lock().unwrap();
        state.available += self.charge;
        state.cached_bytes += self.buffer.capacity();
        state.cached.push(std::mem::take(&mut self.buffer));
    }
}

pub(crate) struct Ready {
    pub id: u64,
    pub result: Result<PackedRecord, &'static str>,
    pub elapsed: Duration,
}

pub(crate) struct Encoder {
    pool: Arc<rayon::ThreadPool>,
    budget: Arc<Mutex<TempBudget>>,
    sender: mpsc::Sender<Ready>,
    ready: mpsc::Receiver<Ready>,
    pending: usize,
    #[cfg(test)]
    pub before_unit: Option<Arc<dyn Fn(Instant) -> Instant + Send + Sync>>,
}

impl Encoder {
    pub fn new(arena_bytes: usize) -> std::io::Result<Self> {
        let workers = (std::thread::available_parallelism().map_or(1, usize::from) / 4).clamp(1, 16);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|_| "manager-encode".into())
            .build()
            .map_err(std::io::Error::other)?;
        let (sender, ready) = mpsc::channel();
        eprintln!("selene: CPU write-back compression uses {workers} workers");
        Ok(Self {
            pool: Arc::new(pool),
            budget: Arc::new(Mutex::new(TempBudget {
                available: arena_bytes / 8,
                cached: Vec::new(),
                cached_bytes: 0,
            })),
            sender,
            ready,
            pending: 0,
            #[cfg(test)]
            before_unit: None,
        })
    }

    pub fn start(
        &mut self,
        id: u64,
        memory: Arc<Extent>,
        header: Header,
        row: bool,
        profile: Profile,
        plans: Plans,
        shutdown_raw: Arc<AtomicBool>,
    ) {
        let pool = self.pool.clone();
        let sender = self.sender.clone();
        let budget = self.budget.clone();
        #[cfg(test)]
        let before_unit = self.before_unit.clone();
        self.pending += 1;
        self.pool.spawn(move || {
            let start = Instant::now();
            let deadline =
                start + WAIT + Duration::from_secs_f64(header.length(row) as f64 / MIN_RATE);
            let result = encode(
                header,
                row,
                memory.bytes(),
                profile,
                &plans,
                &pool,
                &budget,
                || {
                    if shutdown_raw.load(Relaxed) {
                        return Err("shutdown");
                    }
                    let now = Instant::now();
                    #[cfg(test)]
                    let now = before_unit.as_ref().map_or(now, |clock| clock(start));
                    if now >= deadline {
                        return Err("deadline");
                    }
                    Ok(())
                },
            );
            let _ = sender.send(Ready {
                id,
                result,
                elapsed: start.elapsed(),
            });
        });
    }

    pub fn poll(&mut self) -> Option<Ready> {
        let ready = self.ready.try_recv().ok()?;
        self.pending -= 1;
        Some(ready)
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // Finish reading arena inputs before the store releases its shared-region lock.
        for _ in 0..self.pending {
            let _ = self.ready.recv();
        }
    }
}

fn encode(
    header: Header,
    row: bool,
    memory: &[u8],
    profile: Profile,
    plans: &[Plan; 2],
    pool: &rayon::ThreadPool,
    budget: &Arc<Mutex<TempBudget>>,
    before_unit: impl Fn() -> Result<(), &'static str> + Sync,
) -> Result<PackedRecord, &'static str> {
    let length = header.length(row) as usize;
    let raw = memory.get(..length).ok_or("raw_checksum")?;
    record::verify_record(raw, &header, row).map_err(|_| "raw_checksum")?;
    let ranks = header.ranks.len();
    let replication = header.replication();
    let page_ranks = if replication.pages { 1 } else { ranks };
    let row_ranks = if replication.row { 1 } else { ranks };
    let units = header.tail_pages as usize * page_ranks + usize::from(row) * row_ranks;
    // A raw header's rank geometry fits a block. Each descriptor fits 96 bytes
    // plus up to two bytes per additional replica in the packed repeated field.
    let replicas =
        header.tail_pages as usize * (ranks - page_ranks) + usize::from(row) * (ranks - row_ranks);
    let directory_bound = HEADER_BYTES + 16 + units * 96 + replicas * 2;
    if ranks > packed::MAX_RANKS
        || units > packed::MAX_UNITS
        || directory_bound > packed::MAX_DIRECTORY
    {
        return Err("v4_limits");
    }
    // Serialization clones the offer, including each profile and replica list.
    let descriptors =
        units * (size_of::<PackedUnit>() + profile.len()) + replicas * size_of::<u32>();
    // Both growable layout indexes can double, including their end sentinels.
    let indexes = 2 * (units + 1).max(2) * (size_of::<u64>() + size_of::<usize>());
    // The serialized directory overlaps the header-plus-directory buffer.
    // The fixed blocks also cover the cloned raw rank geometry.
    let allowance = 2 * (descriptors + directory_bound) + indexes + 4 * HEADER_BYTES;
    let first_page = header
        .parent
        .map_or(0, |p| p.consumed_tokens / header.page_tokens);
    // The directory precedes the packet slots so that every final copy moves toward offset zero.
    let prefix = HEADER_BYTES + directory_bound;
    let payload = &raw[header.page_segment as usize..];
    let groups = [
        (header.tail_pages as usize * page_ranks, page_ranks),
        (usize::from(row) * row_ranks, row_ranks),
    ];
    let size = groups
        .iter()
        .zip(plans)
        .try_fold(prefix, |total, ((count, _), plan)| {
            total.checked_add(count.checked_mul(plan.max_packet())?)
        })
        .ok_or("temp_budget")?;
    let charge = length
        .max(size)
        .checked_add(allowance)
        .ok_or("temp_budget")?;
    let mut storage = Reservation::take(budget, charge, size)?;
    let mut offer = PackedOffer {
        first_page,
        units: vec![PackedUnit::default(); units],
    };
    let mut sources = payload;
    let mut slots = &mut storage.buffer[prefix..];
    let mut descriptors = offer.units.as_mut_slice();
    for (kind, (count, shares)) in [Kind::Page, Kind::Row].into_iter().zip(groups) {
        if count == 0 {
            continue;
        }
        let plan = &plans[kind as usize];
        let n = plan.original_bytes();
        let (input, rest) = sources
            .split_at_checked(count.checked_mul(n).ok_or("verify_failed")?)
            .ok_or("verify_failed")?;
        sources = rest;
        let packet = plan.max_packet();
        let (output, rest) = slots.split_at_mut(count * packet);
        slots = rest;
        let (units, rest) = descriptors.split_at_mut(count);
        descriptors = rest;
        let workspace = plan.verification_bytes();
        let scratch = workspace + plan.metadata();
        // Keep processing in fewer lanes when the budget cannot fund every rank at once.
        let mut buffers = vec![Reservation::take(budget, scratch, workspace)?];
        while buffers.len() < count.min(pool.current_num_threads()) {
            let Ok(buffer) = Reservation::take(budget, scratch, workspace) else {
                break;
            };
            buffers.push(buffer);
        }
        let batch = count.div_ceil(buffers.len());
        pool.install(|| {
            (
                output.par_chunks_mut(batch * packet),
                units.par_chunks_mut(batch),
                buffers,
            )
                .into_par_iter()
                .enumerate()
                .try_for_each(|(group, (slots, units, mut scratch))| {
                    for (j, (slot, unit)) in slots.chunks_exact_mut(packet).zip(units).enumerate() {
                        before_unit()?;
                        let i = group * batch + j;
                        let source = &input[i * n..(i + 1) * n];
                        let descriptor = PackedUnit {
                            kind: kind as i32,
                            logical_index: if kind == Kind::Page {
                                first_page + (i / shares) as u64
                            } else {
                                0
                            },
                            stored_rank: (i % shares) as u32,
                            replicas: if shares == 1 {
                                (1..ranks as u32).collect()
                            } else {
                                Vec::new()
                            },
                            ..Default::default()
                        };
                        *unit = plan
                            .encode_unit(
                                descriptor,
                                &profile,
                                source,
                                &mut scratch.buffer,
                                slot,
                                pool,
                            )
                            .map_err(|_| "verify_failed")?;
                    }
                    Ok::<_, &'static str>(())
                })
        })?;
    }
    if !sources.is_empty() {
        return Err("verify_failed");
    }
    if !offer.units.iter().any(|unit| unit.representation == Representation::Packet as i32) {
        return Err("literal_only");
    }
    let header = header.with_packed(offer).map_err(|_| "v4_limits")?;
    let stored = header.length(row) as usize;
    if stored >= length {
        return Err("not_smaller");
    }
    let metadata = header.packed_metadata().map_err(|_| "v4_limits")?;
    let bytes = &mut storage.buffer;
    if metadata.len() > prefix {
        return Err("v4_limits");
    }
    bytes[..metadata.len()].copy_from_slice(&metadata);
    let mut source = prefix;
    let mut dest = metadata.len();
    let mut raw_offset = 0;
    for unit in &header.packed.as_ref().unwrap().offer().units {
        let n = unit.stored_length as usize;
        if unit.representation == Representation::Packet as i32 {
            bytes.copy_within(source..source + n, dest);
        } else {
            bytes[dest..dest + n].copy_from_slice(&payload[raw_offset..raw_offset + n]);
        }
        dest += n;
        source += plans[unit.kind as usize].max_packet();
        raw_offset += unit.original_length as usize;
    }
    Ok(PackedRecord {
        header,
        storage,
        length: stored,
    })
}

const BUCKETS: [f64; 7] = [0.001, 0.01, 0.05, 0.1, 0.25, 1.0, 5.0];

#[derive(Default)]
pub(crate) struct Metrics {
    pub original: u64,
    pub stored: u64,
    pub rejected: u64,
    raw: HashMap<&'static str, (u64, u64)>,
    histogram: [u64; 7],
    pub seconds: f64,
    count: u64,
}

impl Metrics {
    pub fn raw(&mut self, cause: &'static str, length: u64) {
        let totals = self.raw.entry(cause).or_default();
        totals.0 += 1;
        totals.1 += length;
    }

    pub fn observe(&mut self, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        self.seconds += seconds;
        self.count += 1;
        for (bucket, count) in BUCKETS.iter().zip(&mut self.histogram) {
            *count += u64::from(seconds <= *bucket);
        }
    }

    pub fn render(&self, text: &mut String) {
        writeln!(
            text,
            "selene_codec_encoded_original_bytes_total {}",
            self.original
        )
        .unwrap();
        writeln!(
            text,
            "selene_codec_encoded_stored_bytes_total {}",
            self.stored
        )
        .unwrap();
        writeln!(text, "selene_codec_layout_rejected_total {}", self.rejected).unwrap();
        for cause in [
            "deadline",
            "temp_budget",
            "v4_limits",
            "literal_only",
            "not_smaller",
            "no_plan",
            "raw_checksum",
            "verify_failed",
            "streamed",
            "write_retry",
            "shutdown",
        ] {
            let (count, bytes) = self.raw.get(cause).copied().unwrap_or_default();
            writeln!(
                text,
                "selene_codec_writeback_raw_total{{cause=\"{cause}\"}} {count}"
            )
            .unwrap();
            writeln!(
                text,
                "selene_codec_writeback_raw_bytes_total{{cause=\"{cause}\"}} {bytes}"
            )
            .unwrap();
        }
        writeln!(text, "# TYPE selene_codec_encode_seconds histogram").unwrap();
        for (le, count) in BUCKETS.iter().zip(self.histogram) {
            writeln!(
                text,
                "selene_codec_encode_seconds_bucket{{le=\"{le}\"}} {count}"
            )
            .unwrap();
        }
        writeln!(
            text,
            "selene_codec_encode_seconds_bucket{{le=\"+Inf\"}} {}",
            self.count
        )
        .unwrap();
        writeln!(text, "selene_codec_encode_seconds_count {}", self.count).unwrap();
        writeln!(text, "selene_codec_encode_seconds_sum {}", self.seconds).unwrap();
    }
}
