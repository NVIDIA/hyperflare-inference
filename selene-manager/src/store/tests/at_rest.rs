// SPDX-License-Identifier: Apache-2.0

use super::*;
use flare_selene::shared::{
    record::checksum,
    wire::{CodecField, CodecLayout, CodecPlane},
};

struct Fixture {
    store: Store,
    id: NamespaceId,
    descriptor: Descriptor,
    layout: CodecLayout,
    root: PathBuf,
    shm: PathBuf,
}

impl Fixture {
    fn new(label: &str, memory: usize, descriptor: Descriptor) -> Self {
        Self::with_compression(label, memory, descriptor, true)
    }

    fn with_compression(label: &str, memory: usize, descriptor: Descriptor, enabled: bool) -> Self {
        let (store, root, shm) = store_with_root(label, memory);
        let mut config = store.config.clone();
        drop(store);
        config.compress_at_rest = enabled;
        let mut store = Store::open(config).unwrap();
        store.hello(1, 0, None, descriptor.clone()).unwrap();
        let id = store.sessions[&1].namespace;
        let rank = &descriptor.ranks[0];
        let plane = |name: &str, n| CodecPlane {
            name: name.into(),
            unit_bytes: n,
            fields: vec![CodecField {
                offset: 0,
                length: n,
                encoding: 0,
            }],
        };
        let layout = CodecLayout {
            revision: "ZKV1/v1/64-pivots".into(),
            page_tokens: 64,
            planes: vec![plane("kv", rank.planes[0])],
            row: Some(plane("row", rank.row_bytes)),
        };
        Self {
            store,
            id,
            descriptor,
            layout,
            root,
            shm,
        }
    }

    fn offer(&mut self, session: u64, layout: CodecLayout, profile: [u8; 32]) {
        self.store.handle(StoreCommand::OfferLayout {
            session,
            namespace: self.id,
            layout,
            caps: Some(flare_selene::codec_copy::reader_caps(Some(profile), false)),
        });
    }

    fn enable(&mut self) -> [u8; 32] {
        let profile = self.layout.profile().unwrap();
        self.offer(1, self.layout.clone(), profile);
        profile
    }

    fn open_namespace(&mut self, session: u8) {
        self.descriptor.identity[0] = session;
        self.store
            .hello(u64::from(session), 0, None, self.descriptor.clone())
            .unwrap();
        self.id = self.descriptor.namespace_id();
    }

    fn insert(&mut self, pages: u64, parent: Option<Key>, fill: impl Fn(usize, &mut [u8])) -> Key {
        let key = page_key(pages);
        let tail = (pages - parent.map_or(0, |p| p.consumed_tokens / 64)) as u32;
        let mut header = record::new_header(&self.descriptor, key, parent, [0; 16], tail)
            .unwrap()
            .with_capture_role(CaptureRole::Endpoint);
        let mut extent = self
            .store
            .memory
            .allocate(header.length(true) as usize)
            .unwrap();
        let bytes = extent.bytes_mut();
        let page_bytes = header.page_record_bytes() as usize;
        for i in 0..tail as usize {
            let start = header.page_segment as usize + i * page_bytes;
            let page = &mut bytes[start..start + page_bytes];
            fill(i, page);
            let hash = checksum(page);
            bytes[4096 + 8 * i..4104 + 8 * i].copy_from_slice(&hash.to_le_bytes());
        }
        let row = &mut bytes[header.row_segment as usize..];
        fill(tail as usize, row);
        header.row_checksum = checksum(row);
        bytes[..4096].copy_from_slice(&header.encode().unwrap());
        insert_memory_node(
            &mut self.store,
            self.id,
            &self.descriptor,
            key,
            parent,
            tail,
            extent,
        );
        self.store.namespaces.get_mut(&self.id).unwrap().tree[&key]
            .location
            .header = header;
        self.store.refresh_durability(self.id, Some(key));
        key
    }

    fn write(&mut self, key: Key) -> Header {
        self.store.write_ancestor(self.id, key, "dependency");
        finish_jobs(&mut self.store);
        self.store.namespaces[&self.id].tree[&key]
            .location
            .disk_header
            .clone()
            .unwrap()
    }

    fn verify_file(&self, key: Key, row: bool) -> Header {
        let bytes = fs::read(self.store.namespaces[&self.id].path(key).unwrap()).unwrap();
        let header = Header::decode(&bytes).unwrap();
        record::verify_record(&bytes, &header, row).unwrap();
        header
    }

    fn submit(
        &mut self,
        session: u64,
        pages: u64,
        role: CaptureRole,
    ) -> tokio::sync::mpsc::Receiver<Result<PutAnswer>> {
        let (events, answers) = tokio::sync::mpsc::channel(4);
        self.store.submit_put(
            session,
            1,
            PutRequest {
                namespace: self.id,
                key: page_key(pages),
                published_parent: None,
                ancestor_keys: vec![],
                bucket: [0; 16],
                chain_pages: pages as u32,
                published_tail_pages: pages as u32,
                capture_role: role,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
            },
            events,
        );
        answers
    }

    fn raw_count(&self, cause: &str, count: u64) {
        assert!(self.store.metrics_text().contains(&format!(
            "selene_codec_writeback_raw_total{{cause=\"{cause}\"}} {count}\n"
        )));
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
        let _ = fs::remove_dir_all(&self.shm);
    }
}

fn zeros(_: usize, bytes: &mut [u8]) {
    bytes.fill(0);
}

fn one_packet(i: usize, bytes: &mut [u8]) {
    if i == 0 {
        return zeros(i, bytes);
    }
    // Pseudorandom high-entropy bytes force literal storage under Zstandard.
    let mut state = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xd1b5_4a32_d192_ed03;
    for chunk in bytes.chunks_mut(8) {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let raw = z.to_le_bytes();
        chunk.copy_from_slice(&raw[..chunk.len()]);
    }
}

#[test]
fn eviction_releases_the_plans_of_closed_namespaces() {
    let mut f = Fixture::new("codec-plan-eviction", 1 << 20, descriptor());
    let mut records = Vec::new();
    for session in 2..=5 {
        f.open_namespace(session);
        let session = u64::from(session);
        let mut layout = f.layout.clone();
        layout.planes[0].name = format!("kv-{session}");
        let profile = layout.profile().unwrap();
        f.offer(session, layout, profile);
        let key = f.insert(1, None, zeros);
        f.store.connection_closed(session);
        f.store.handle(StoreCommand::WithdrawLayout { session });
        records.push((f.id, key));
    }
    assert_eq!(f.store.profiles.plan_count(), records.len());
    let (id, key) = records[0];
    f.store.request_writeback(id, key);
    assert_eq!(f.store.evict_accounted(id, key), 0);
    finish_jobs(&mut f.store);
    for (id, key) in records {
        assert_eq!(f.store.evict_accounted(id, key), 1);
    }
    assert_eq!(f.store.profiles.plan_count(), 0);
}

#[test]
fn empty_namespaces_share_plans_until_the_last_offer_closes() {
    let mut f = Fixture::new("codec-plan-withdraw", 1 << 20, descriptor());
    let profile = f.enable();
    f.open_namespace(2);
    f.offer(2, f.layout.clone(), profile);
    let key = f.insert(1, None, zeros);
    assert_eq!(f.store.evict_accounted(f.id, key), 1);
    f.store.handle(StoreCommand::WithdrawLayout { session: 2 });
    assert_eq!(f.store.profiles.plan_count(), 1);
    f.store.handle(StoreCommand::WithdrawLayout { session: 1 });
    assert_eq!(f.store.profiles.plan_count(), 0);
}

#[test]
fn compression_off_ignores_layout_offers() {
    let mut f = Fixture::with_compression("codec-offer-off", 1 << 20, descriptor(), false);
    f.enable();
    assert_eq!(f.store.profiles.plan_count(), 0);
    f.offer(1, f.layout.clone(), [0; 32]);
    assert_eq!(f.store.metrics.codec.rejected, 0);
}

#[test]
fn compaction_releases_the_resident_copy_at_the_packed_row_offset() {
    let mut f = Fixture::new("codec-compact", 1 << 20, descriptor());
    f.enable();
    let root = f.insert(1, None, zeros);
    let middle = f.insert(2, Some(root), zeros);
    let leaf = f.insert(3, Some(middle), zeros);
    f.write(root);
    let disk = f.write(middle);
    f.write(leaf);
    let (tx, mut feed) = tokio::sync::mpsc::unbounded_channel();
    f.store.catalog(f.descriptor.clone(), tx).unwrap();
    let tree = &mut f.store.namespaces.get_mut(&f.id).unwrap().tree;
    tree.pin(&root).unwrap();
    tree.pin(&leaf).unwrap();
    assert!(tree[&middle].location.memory.is_some());
    let before = f.store.charged;
    let resident = tree[&middle].location.bytes;
    f.store.config.nvme = before;
    assert!(!f.store.make_room_evict(1));
    finish_jobs(&mut f.store);
    let node = &f.store.namespaces[&f.id].tree[&middle];
    assert_eq!(node.location.bytes, disk.row_segment);
    assert_eq!(f.store.charged, before - resident + disk.row_segment);
    assert!(node.location.memory.is_none());
    assert!(!node.row);
    assert!(f.verify_file(middle, false).packed.is_some());
    let mut changed = Vec::new();
    while let Ok(proto::CatalogEvent::Changed(entry)) = feed.try_recv() {
        changed.push((entry.key, entry.packed, entry.row));
    }
    assert_eq!(changed, [(middle, false, false), (middle, true, false)]);
}

#[test]
fn a_32_mib_put_reclaims_the_existing_64_mib_charge_without_deletion() {
    let mut descriptor = descriptor();
    descriptor.ranks[0].planes = vec![128 << 10];
    let mut f = Fixture::new("codec-budget", 512 << 20, descriptor);
    f.enable();
    f.store.encoder.as_mut().unwrap().before_unit = Some(Arc::new(|start| start));
    let key = f.insert(256, None, one_packet);
    let disk = f.write(key);
    let length = disk.length(true);
    assert!(disk.packed.is_some() && (31 << 20..=32 << 20).contains(&length));
    assert_eq!(f.store.charged, 64 << 20);
    let mut answers = f.submit(1, 129, CaptureRole::Legacy);
    f.store.pump();
    assert!(matches!(
        answers.try_recv().unwrap().unwrap(),
        PutAnswer::Accept(_)
    ));
    assert_eq!(f.store.charged, length + (32 << 20));
    assert_eq!(f.store.metrics.evictions, 0);
    assert!(f.verify_file(key, true).packed.is_some());
}

#[test]
fn slow_record_stops_at_its_deadline() {
    let mut f = Fixture::new("codec-deadline", 4 << 20, descriptor());
    f.enable();
    let key = f.insert(32, None, zeros);
    let bytes = f.store.namespaces[&f.id].tree[&key].location.header.length(true);
    let deadline = WAIT + Duration::from_secs_f64(bytes as f64 / compression::MIN_RATE);
    f.store.encoder.as_mut().unwrap().before_unit = Some(Arc::new(move |start| start + deadline));
    let disk = f.write(key);
    assert!(disk.packed.is_none() && f.verify_file(key, true).packed.is_none());
    f.raw_count("deadline", 1);
    assert!(f.store.metrics_text().contains(&format!(
        "selene_codec_writeback_raw_bytes_total{{cause=\"deadline\"}} {bytes}\n"
    )));
}

#[test]
fn record_can_finish_within_its_size_allowance() {
    let mut f = Fixture::new("codec-size-allowance", 4 << 20, descriptor());
    f.enable();
    let key = f.insert(8, None, zeros);
    f.store.encoder.as_mut().unwrap().before_unit =
        Some(Arc::new(|start| start + WAIT + Duration::from_nanos(1)));
    assert!(f.write(key).packed.is_some());
    assert!(f.verify_file(key, true).packed.is_some());
    f.raw_count("deadline", 0);
}

#[test]
fn serial_ancestors_survive_repeated_arena_reclaims() {
    let mut f = Fixture::new("codec-reclaim", 1 << 20, descriptor());
    f.enable();
    let mut keys = Vec::new();
    for i in 1..=5 {
        keys.push(f.insert(i * 8, keys.last().copied(), zeros));
    }
    f.store.encoder.as_mut().unwrap().before_unit =
        Some(Arc::new(|_| {
            std::thread::sleep(Duration::from_millis(20));
            Instant::now()
        }));
    let mut answers = f.submit(1, 128, CaptureRole::Legacy);
    let start = Instant::now();
    loop {
        f.store.pump();
        if let Ok(answer) = answers.try_recv() {
            assert!(matches!(answer.unwrap(), PutAnswer::Accept(_)));
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "put still waiting"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(start.elapsed() > WAIT);
    assert!(f.store.metrics.put_writeback_waits > 1);
    for key in keys {
        let location = &f.store.namespaces[&f.id].tree[&key].location;
        assert!(location.durable && location.memory.is_none());
        let disk = f.verify_file(key, true);
        assert_eq!(disk.key, key);
        assert!(disk.packed.is_some());
    }
}

#[test]
fn shutdown_writes_running_and_new_encodes_raw() {
    let mut f = Fixture::new("codec-shutdown", 1 << 20, descriptor());
    f.enable();
    let root = f.insert(8, None, zeros);
    let child = f.insert(16, Some(root), zeros);
    f.store.encoder.as_mut().unwrap().before_unit =
        Some(Arc::new(|_| {
            std::thread::sleep(Duration::from_millis(40));
            Instant::now()
        }));
    f.store.write_ancestor(f.id, root, "dependency");
    assert!(f.store.drain_until(Instant::now() + WAIT));
    for key in [root, child] {
        assert!(f.verify_file(key, true).packed.is_none());
    }
    f.raw_count("shutdown", 2);
}

#[test]
fn v4_unit_limit_and_whole_file_growth_preserve_readable_raw_files() {
    let mut f = Fixture::new("codec-limits", 256 << 20, descriptor());
    f.enable();
    let oversized = f.insert(16384, None, zeros);
    f.write(oversized);
    f.raw_count("v4_limits", 1);
    let expanding = f.insert(1024, None, one_packet);
    f.write(expanding);
    f.raw_count("not_smaller", 1);
    for key in [oversized, expanding] {
        f.store.evict_memory(f.id, key);
        assert!(f.verify_file(key, true).packed.is_none());
        assert!(!Store::entry(&f.store.namespaces[&f.id], key).packed);
    }
}

#[test]
fn a_resident_raw_parent_binds_children_to_its_disk_profile() {
    let mut f = Fixture::new("codec-chain", 1 << 20, descriptor());
    let p = f.enable();
    let parent = f.insert(1, None, zeros);
    assert!(f.write(parent).packed.is_some());
    assert!(!Store::entry(&f.store.namespaces[&f.id], parent).packed);
    f.store.handle(StoreCommand::WithdrawLayout { session: 1 });
    let mut q_layout = f.layout.clone();
    q_layout.planes[0].name = "other".into();
    let q = q_layout.profile().unwrap();
    assert_ne!(p, q);
    f.offer(2, q_layout, q);
    let child = f.insert(2, Some(parent), zeros);
    assert!(f.write(child).packed.is_none());
    f.raw_count("no_plan", 1);
    f.enable();
    let grandchild = f.insert(3, Some(child), zeros);
    let packed = f.write(grandchild).packed.unwrap();
    assert_eq!(packed.required_profiles(true), [p.to_vec()]);
}

#[test]
fn a_mismatched_digest_or_nonfirst_rank_geometry_supplies_no_plan() {
    let mut descriptor = descriptor();
    descriptor.ranks.push(descriptor.ranks[0].clone());
    // Records store planes padded to a block, so only a padded difference is a mismatch.
    descriptor.ranks[1].planes[0] += 4096;
    let mut f = Fixture::new("codec-layout", 1 << 20, descriptor);
    let layout = f.layout.clone();
    let profile = layout.profile().unwrap();
    f.offer(1, layout, profile);
    assert_eq!(f.store.metrics.codec.rejected, 1);
    let key = f.insert(1, None, zeros);
    assert!(f.write(key).packed.is_none());
    f.raw_count("no_plan", 1);

    let mut matched = Fixture::new("codec-digest", 1 << 20, super::descriptor());
    matched.offer(1, matched.layout.clone(), [0; 32]);
    assert_eq!(matched.store.metrics.codec.rejected, 1);
    let key = matched.insert(1, None, zeros);
    assert!(matched.write(key).packed.is_none());
    matched.raw_count("no_plan", 1);
}

#[test]
fn a_layout_with_unpadded_unit_sizes_matches_the_padded_session_geometry() {
    // Flare's session geometry pads each plane and the row to a block; real layouts carry the
    // unpadded unit sizes (GLM planes are not block multiples). The offer must compare padded sizes.
    let mut f = Fixture::new("codec-unpadded", 1 << 20, descriptor());
    for plane in f.layout.planes.iter_mut().chain(f.layout.row.iter_mut()) {
        plane.unit_bytes -= 100;
        plane.fields[0].length = plane.unit_bytes;
    }
    assert!(f.layout.planes[0].unit_bytes % 4096 != 0);
    let profile = f.enable();
    assert_eq!(f.store.metrics.codec.rejected, 0);
    let key = f.insert(2, None, zeros);
    let disk = f.write(key);
    assert!(disk.packed.is_some(), "a compressible record is written packed");
    assert_eq!(disk.packed.unwrap().required_profiles(true), vec![profile.to_vec()]);
}

#[test]
fn compression_off_deletes_the_raw_leaf_under_budget_pressure() {
    let mut f = Fixture::with_compression("raw-budget", 1 << 20, descriptor(), false);
    let leaf = f.insert(2, None, zeros);
    let disk = f.write(leaf);
    let node = &f.store.namespaces[&f.id].tree[&leaf];
    let path = f.store.namespaces[&f.id].path(leaf).unwrap();
    assert_eq!(disk, node.location.header);
    assert_eq!(
        fs::read(&path).unwrap(),
        node.location.memory.as_ref().unwrap().bytes()
    );
    assert_eq!(disk.length(true), 20_480);
    assert_eq!(f.store.charged, 32_768);
    let (tx, mut feed) = tokio::sync::mpsc::unbounded_channel();
    f.store.catalog(f.descriptor.clone(), tx).unwrap();
    f.store.config.nvme = 53_248;
    let mut answers = f.submit(1, 3, CaptureRole::Legacy);
    f.store.pump();
    assert!(matches!(
        answers.try_recv().unwrap().unwrap(),
        PutAnswer::Accept(_)
    ));
    assert!(
        !f.store.namespaces[&f.id].tree.contains(&leaf),
        "raw leaf survived"
    );
    assert!(!path.exists());
    assert_eq!(f.store.charged, 32_768);
    assert_eq!(f.store.metrics.evictions, 1);
    assert!(matches!(feed.try_recv().unwrap(), proto::CatalogEvent::Removed(key) if key == leaf));
    assert!(feed.try_recv().is_err());
}

#[test]
fn packed_metadata_exceeding_the_temp_budget_keeps_the_file_raw() {
    let arena_bytes = 1 << 20;
    let mut f = Fixture::new("codec-temp", arena_bytes, descriptor());
    f.enable();
    let key = f.insert(29, None, zeros);
    let raw_length = f.store.namespaces[&f.id].tree[&key]
        .location
        .header
        .length(true);
    assert_eq!(raw_length, (arena_bytes / 8) as u64);
    f.write(key);
    assert!(
        f.verify_file(key, true).packed.is_none(),
        "record was encoded"
    );
    f.raw_count("temp_budget", 1);
}

#[test]
fn a_replaced_root_respects_its_retained_childs_disk_profile() {
    for offer_p in [false, true] {
        let mut f = Fixture::new("codec-replace", 1 << 20, descriptor());
        let p = f.enable();
        let root = f.insert(1, None, zeros);
        let child = f.insert(2, Some(root), zeros);
        for key in [root, child] {
            assert_eq!(
                f.write(key).packed.unwrap().required_profiles(true),
                [p.to_vec()]
            );
        }
        f.store.connection_closed(1);
        f.store.handle(StoreCommand::WithdrawLayout { session: 1 });
        f.store.hello(2, 0, None, f.descriptor.clone()).unwrap();
        let mut other = f.layout.clone();
        other.planes[0].name = "other".into();
        let q = other.profile().unwrap();
        assert_ne!(p, q);
        f.offer(2, other, q);
        if offer_p {
            f.store.hello(3, 0, None, f.descriptor.clone()).unwrap();
            f.offer(3, f.layout.clone(), p);
        }

        // Re-put the same page and row with the stronger capture role.
        let mut answers = f.submit(2, 1, CaptureRole::Message);
        f.store.pump();
        assert!(
            matches!(answers.try_recv().unwrap().unwrap(), PutAnswer::Accept(a) if a.capture_role == CaptureRole::Message)
        );
        let (put, _) = put_mut(&mut f.store.sessions, 2, 1).unwrap();
        Arc::get_mut(put.memory.as_mut().unwrap())
            .unwrap()
            .bytes_mut()
            .fill(0);
        let hash = checksum(&[0; 4096]);
        let mut header = put.header.clone();
        header.row_checksum = hash;
        let mut metadata = vec![0; header.page_segment as usize];
        metadata[..4096].copy_from_slice(&header.encode().unwrap());
        metadata[4096..4104].copy_from_slice(&hash.to_le_bytes());
        put.requested += 1;
        f.store.pump();
        let PutAnswer::Grant(grant) = answers.try_recv().unwrap().unwrap() else {
            panic!("expected a replacement window")
        };
        let index = grant.index;
        drop(grant);
        f.store
            .window_filled(2, 1, index, vec![hash], Some(hash))
            .unwrap();
        f.store.put_commit(2, 1, 0, metadata).unwrap();
        let disk = f.write(root);
        assert_eq!(disk.capture_role(), CaptureRole::Message);

        for key in [root, child] {
            assert!(f.store.evict_memory(f.id, key));
        }
        let (records, _) = f
            .store
            .plan_get(
                f.id,
                &GetRequest {
                    namespace: f.id,
                    keys: vec![root, child],
                    skip_pages: 0,
                    expected_pages: 2,
                    row: true,
                },
            )
            .unwrap();
        let profiles: HashSet<_> = records
            .iter()
            .filter_map(|r| r.header.packed.as_ref())
            .flat_map(|layout| layout.required_profiles(true))
            .collect();
        assert_eq!(
            profiles,
            HashSet::from([p.to_vec()]),
            "profiles required from page zero"
        );
        assert_eq!(disk.packed.is_some(), offer_p);
        f.raw_count("no_plan", u64::from(!offer_p));
    }
}
