/*
 * SPDX-License-Identifier: Apache-2.0
 */

mod support;

use flare_selene::shared::{
    frame::{ToInstance, ToManager},
    packed,
    record::Header,
};
use selene_manager::{
    pool::key_to,
    proto::v2::{self, client_message::M as C, server_message::M},
};
use support::{
    Host,
    channel::{self, Security},
    key,
    pool::{self, Ident, Pool, Session},
};
use tokio::io::AsyncWriteExt;

const PAGE0: &[u8] = include_bytes!("../../selene/tests/fixtures/codec/page0.sgc");
const PAGE1: &[u8] = include_bytes!("../../selene/tests/fixtures/codec/page1.sgc");
const ROW: &[u8] = include_bytes!("../../selene/tests/fixtures/codec/row.sgc");

fn payload(unit: &v2::PackedUnit) -> &'static [u8] {
    match unit.original_crc {
        0x476cc8d3 => PAGE0,
        0xed7164fa => PAGE1,
        0xb2b29b40 => ROW,
        _ => panic!("unexpected unit"),
    }
}

fn offer() -> v2::PackedOffer {
    let units = [(0, 0, 0x476cc8d3), (0, 1, 0xed7164fa), (1, 0, 0xb2b29b40)]
        .into_iter()
        .map(|(kind, logical_index, original_crc)| {
            let mut u = v2::PackedUnit {
                kind,
                logical_index,
                original_crc,
                representation: 1,
                profile: vec![3; 32],
                original_length: 4096,
                ..Default::default()
            };
            let bytes = payload(&u);
            u.stored_crc = packed::crc32(bytes);
            u.stored_length = bytes.len() as u64;
            u
        })
        .collect();
    v2::PackedOffer {
        first_page: 0,
        units,
    }
}

async fn open(pool: &mut Pool, name: &str, capable: bool) -> Session {
    let mut ident = Ident::new(name);
    ident.codec = capable.then(|| v2::CodecCaps {
        envelope_revision: 4,
        receive_profiles: vec![vec![3; 32]],
    });
    let mut session = pool.open(&ident).await;
    assert_eq!(session.opened.codec, ident.codec);
    session.snapshot(vec![]).await;
    session
}

fn begin(tcp: bool) -> v2::PutBegin {
    v2::PutBegin {
        put_id: 1,
        key: Some(key_to(key(128))),
        row: true,
        chain_pages: 2,
        published_tail_pages: 2,
        bucket: vec![4; 16],
        mode: if tcp {
            v2::CopyMode::DataChannel
        } else {
            v2::CopyMode::SharedWindows
        } as i32,
        packed: Some(offer()),
        ..Default::default()
    }
}

async fn put(writer: &mut Session, tcp: bool, request: v2::PutBegin) -> u64 {
    writer.send(C::PutBegin(request)).await;
    let accept = writer
        .take(|m| match m {
            M::PutAccept(a) => Some(a.clone()),
            M::PutRefused(r) => panic!("{r:?}"),
            _ => None,
        })
        .await;
    let physical = accept.packed.unwrap().physical_record_bytes;
    assert_eq!(accept.record_bytes, 5 * 4096);
    let mut conn = if tcp {
        Some(channel::dial(accept.channel.as_ref().unwrap(), &Security::Plain).await)
    } else {
        None
    };
    for _ in 0..2 {
        if let Some(c) = &mut conn {
            let ToInstance::PackedGrant(shape, w) = ToInstance::read(c, 64).await.unwrap() else {
                panic!("missing packed grant")
            };
            let data: Vec<_> = w
                .units
                .iter()
                .flat_map(|u| payload(u).iter().copied())
                .collect();
            assert_eq!(shape.length, data.len() as u64);
            ToManager::PackedFilled(shape.index).write(c).await.unwrap();
            c.write_all(&data).await.unwrap();
            c.flush().await.unwrap();
        } else {
            let w = writer
                .take(|m| match m {
                    M::PutGrant(g) => g.window.clone(),
                    _ => None,
                })
                .await;
            let data: Vec<_> = w
                .packed
                .as_ref()
                .unwrap()
                .units
                .iter()
                .flat_map(|u| payload(u).iter().copied())
                .collect();
            writer.shm.as_ref().unwrap().write(&w, &data);
            writer
                .send(C::PutFilled(v2::PutFilled {
                    put_id: 1,
                    index: w.index,
                    ..Default::default()
                }))
                .await;
        }
    }
    writer
        .send(C::PutCommit(v2::PutCommit {
            put_id: 1,
            ..Default::default()
        }))
        .await;
    writer
        .take(|m| match m {
            M::Stored(_) => Some(()),
            M::PutRefused(r) => panic!("{r:?}"),
            _ => None,
        })
        .await;
    drop(conn);
    writer
        .send(C::FenceAck(v2::FenceAck {
            subject: Some(v2::fence_ack::Subject::PutId(1)),
            outcome: v2::FenceOutcome::Drained as i32,
        }))
        .await;
    physical
}

async fn read(reader: &mut Session, id: u64, tcp: bool) {
    let candidate = reader.manager_candidate(id, 128).await;
    let mut claim = reader.claim_for(id, &candidate, 64, true);
    if tcp {
        claim.mode = v2::CopyMode::DataChannel as i32
    }
    reader.claim(claim).await;
    let granted = reader.claim_granted(id).await;
    let selected = granted.records[0].packed.as_ref().unwrap();
    assert_eq!(selected.units, &offer().units[1..]);
    let mut conn = if tcp {
        Some(channel::dial(granted.channel.as_ref().unwrap(), &Security::Plain).await)
    } else {
        None
    };
    for expected in &selected.units {
        let (index, units, data) = if let Some(c) = &mut conn {
            let ToInstance::PackedWindow(s, w) = ToInstance::read(c, 64).await.unwrap() else {
                panic!("missing packed window")
            };
            let mut data = vec![0; s.length as usize];
            flare_selene::shared::frame::read_packed_payload(
                c,
                std::slice::from_ref(expected),
                &w,
                &mut data,
            )
            .await
            .unwrap();
            (s.index, w.units, data)
        } else {
            let w = reader.window_ready(id).await.window.unwrap();
            let data = reader.shm.as_ref().unwrap().read(&w);
            packed::verify_window(
                std::slice::from_ref(expected),
                w.packed.as_ref().unwrap(),
                &data,
            )
            .unwrap();
            (w.index, w.packed.unwrap().units, data)
        };
        assert_eq!(units, std::slice::from_ref(expected));
        packed::verify_units(&units, &data).unwrap();
        assert_eq!(data, payload(expected));
        reader.released(id, index).await;
    }
    reader.copy_result(id, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        reader.claim_terminal(id).await.state,
        v2::ClaimState::Copied as i32
    );
    reader.fence(id, v2::FenceOutcome::Drained).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn canonical_packets_survive_shared_windows_tcp_drain_and_recovery() {
    let base = support::base("packed");
    let root = base.join("records");
    let mut config = selene_manager::store::Config::new(
        root.clone(),
        base.join("run/m.sock"),
        8 << 20,
        64 << 20,
    );
    config.name = support::instance_name("packed");
    config.listen = Some(support::net::free_address());
    config.listen_data = Some(support::net::free_address());
    let mut host = Host::open_config(config.clone());
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut legacy = open(&mut pool, "legacy", false).await;
    legacy.send(C::PutBegin(begin(false))).await;
    legacy
        .take(|m| match m {
            M::PutRefused(r) => Some(assert_eq!(r.reason, v2::Reason::Invalid as i32)),
            _ => None,
        })
        .await;
    let mut writer = open(&mut pool, "writer", true).await;
    let physical = put(&mut writer, true, begin(true)).await;
    let mut reader = open(&mut pool, "reader", true).await;
    read(&mut reader, 1, false).await;
    let reply = legacy
        .lookup_request(v2::LookupRequest {
            lookup_id: 1,
            boundaries: vec![v2::PrefixKey {
                consumed_tokens: 0,
                digest: vec![4; 16],
            }],
            before_tokens: 129,
            ..Default::default()
        })
        .await;
    assert!(reply.candidates.is_empty());
    assert!(host.metrics().contains(&format!(
        "selene_packed_arena_buddy_slack_bytes {}",
        physical.next_power_of_two() - physical
    )));
    drop((legacy, writer, reader, pool));
    host.stop();
    let dir = root.join(flare_selene::record::directory_name(&[7; 32], &[3; 16]));
    let file = dir.join(flare_selene::record::file_name(&key(128)));
    let bytes = std::fs::read(&file).unwrap();
    let h = Header::decode(&bytes).unwrap();
    assert_eq!(bytes.len() as u64, physical);
    assert_eq!(
        &bytes[h.page_segment as usize..],
        [PAGE0, PAGE1, ROW].concat()
    );
    selene_manager::record::verify_record(&bytes, &h, true).unwrap();
    eprintln!(
        "packed fixture: physical {physical}, buddy {}, slack {}",
        physical.next_power_of_two(),
        physical.next_power_of_two() - physical
    );
    let mut second = Host::open_config(config);
    second.run();
    let mut pool = Pool::connect(&second.config.socket).await;
    let mut reader = open(&mut pool, "reopen", true).await;
    read(&mut reader, 2, true).await;
    drop((pool, reader));
    second.stop();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compacted_endpoint_flushes_only_its_pages_and_keeps_the_directory() {
    let base = support::base("packed-cut");
    let root = base.join("records");
    let mut config = selene_manager::store::Config::new(
        root.clone(),
        base.join("run/m.sock"),
        8 << 20,
        64 << 20,
    );
    config.name = support::instance_name("packed-cut");
    let mut host = Host::open_config(config);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut raw = open(&mut pool, "raw", false).await;
    assert!(matches!(
        raw.put_record(1, key(64), None, Default::default()).await,
        support::pool::PutResult::Stored(_)
    ));
    let mut writer = open(&mut pool, "packed", true).await;
    let mut req = begin(false);
    req.key = Some(key_to(key(192)));
    req.chain_pages = 3;
    req.has_published_parent = true;
    req.published_parent = Some(key_to(key(64)));
    req.capture_role = v2::CaptureRole::Endpoint as i32;
    let offer = req.packed.as_mut().unwrap();
    offer.first_page = 1;
    for u in offer.units.iter_mut().filter(|u| u.kind == 0) {
        u.logical_index += 1
    }
    let offer = offer.clone();
    let full = put(&mut writer, false, req).await;
    assert!(matches!(
        raw.put_record(2, key(256), Some(key(192)), Default::default())
            .await,
        support::pool::PutResult::Stored(_)
    ));
    drop((writer, raw, pool));
    host.stop();
    let dir = root.join(flare_selene::record::directory_name(&[7; 32], &[3; 16]));
    let path = dir.join(flare_selene::record::file_name(&key(192)));
    let bytes = std::fs::read(&path).unwrap();
    let h = Header::decode(&bytes).unwrap();
    assert_eq!(h.packed.as_ref().unwrap().offer(), &offer);
    assert_eq!(h.row_segment, bytes.len() as u64);
    assert_eq!(full - h.row_segment, ROW.len() as u64);
    assert_eq!(&bytes[h.page_segment as usize..], [PAGE0, PAGE1].concat());
    assert_eq!(
        &bytes[..h.page_segment as usize],
        h.packed_metadata().unwrap()
    );
    // A partially present row is equally unusable after a restart. The raw
    // child links through this packed parent and keeps it reachable.
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&ROW[..37])
        .unwrap();
    let lock = flare_selene::record::DirectoryLock::acquire(&dir).unwrap();
    let records =
        flare_selene::record::scan(&lock, &[7; 32], &[3; 16], 64, &support::descriptor().ranks)
            .unwrap();
    assert_eq!(
        records
            .iter()
            .map(|r| (r.header.key, r.row))
            .collect::<Vec<_>>(),
        [(key(64), true), (key(192), false), (key(256), true)]
    );
}

#[test]
fn directory_bounds_integrity_and_physical_window_limits() {
    let ranks = support::descriptor().ranks;
    let offer = offer();
    for mutate in [
        |o: &mut v2::PackedOffer| o.units[0].stored_length = u64::MAX,
        |o: &mut v2::PackedOffer| o.units[0].original_length = 17,
        |o: &mut v2::PackedOffer| o.units[0].replicas.push(0),
        |o: &mut v2::PackedOffer| o.units.swap(0, 2),
        |o: &mut v2::PackedOffer| o.units[0].profile.clear(),
        |o: &mut v2::PackedOffer| o.units[0].raw_xxh3 = 1,
    ] {
        let mut bad = offer.clone();
        mutate(&mut bad);
        assert!(packed::Layout::new(bad, &ranks, 0, 2).is_err());
    }
    let layout = packed::Layout::new(offer.clone(), &ranks, 0, 2).unwrap();
    assert_eq!(
        layout
            .windows(0, true, 1000)
            .iter()
            .map(|w| w.length)
            .collect::<Vec<_>>(),
        [704, 994, 1208]
    );
    assert_eq!(layout.windows(1, true, 2000)[0].offset, 704);
    let header =
        selene_manager::record::new_header(&support::descriptor(), key(128), None, [4; 16], 2)
            .unwrap()
            .with_packed(offer.clone())
            .unwrap();
    let metadata = header.packed_metadata().unwrap();
    assert!(Header::decode(&metadata[..metadata.len() - 1]).is_err());
    let mut corrupt = metadata.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(Header::decode(&corrupt).is_err());
    let mut huge = metadata;
    huge[181..185].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Header::read_metadata(&*huge).is_err());
    let mut damaged = PAGE1.to_vec();
    damaged[70] ^= 1;
    assert!(packed::verify_units(&offer.units[1..2], &damaged).is_err());
    let mut literal = offer.units[0].clone();
    literal.representation = 0;
    literal.profile.clear();
    literal.stored_crc = 0;
    literal.original_crc = 0;
    literal.stored_length = 4096;
    let bytes = vec![17; 4096];
    literal.raw_xxh3 = flare_selene::record::checksum(&bytes);
    let mut mixed = offer;
    mixed.units[0] = literal;
    let layout = packed::Layout::new(mixed, &ranks, 0, 2).unwrap();
    assert!(packed::verify_units(layout.units(0, 1, false).unwrap(), &bytes).is_ok());
    assert!(packed::verify_units(layout.units(0, 1, false).unwrap(), &vec![18; 4096]).is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arena_pressure_and_streaming_write_the_same_bytes() {
    for memory in [4096, 8192] {
        let base = support::base("packed-pressure");
        let root = base.join("records");
        let mut cfg = selene_manager::store::Config::new(
            root.clone(),
            base.join("run/m.sock"),
            memory,
            64 << 20,
        );
        cfg.name = support::instance_name("packed-pressure");
        cfg.staging = 8 << 20;
        let mut host = Host::open_config(cfg);
        host.run();
        let mut pool = Pool::connect(&host.config.socket).await;
        let mut first = open(&mut pool, "pressure-first", true).await;
        let size = put(&mut first, false, begin(false)).await;
        let file = root
            .join(flare_selene::record::directory_name(&[7; 32], &[3; 16]))
            .join(flare_selene::record::file_name(&key(128)));
        if memory == 8192 {
            assert!(!file.exists(), "the first record is dirty in the arena");
            let mut second = open(&mut pool, "pressure-second", true).await;
            let mut next = begin(false);
            next.key = Some(key_to(key(129)));
            put(&mut second, false, next).await;
        }
        let bytes = std::fs::read(file).unwrap();
        let h = Header::decode(&bytes).unwrap();
        assert_eq!(bytes.len() as u64, size);
        assert_eq!(
            &bytes[h.page_segment as usize..],
            [PAGE0, PAGE1, ROW].concat()
        );
        drop((pool, first));
        host.stop();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn incompatible_suffix_and_bad_unit_are_refused_without_publishing() {
    let mut host = Host::dram_only(1 << 20);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut writer = open(&mut pool, "bad", true).await;
    let mut short = begin(false);
    short.has_published_parent = true;
    short.published_parent = Some(key_to(key(64)));
    short.published_tail_pages = 1;
    let suffix = short.packed.as_mut().unwrap();
    suffix.first_page = 1;
    suffix.units.remove(0);
    writer.send(C::PutBegin(short)).await;
    let refusal = writer
        .take(|m| match m {
            M::PutRefused(r) => Some(r.clone()),
            _ => None,
        })
        .await;
    assert_eq!(refusal.detail, packed::RAW_REQUIRED);
    let mut req = begin(false);
    req.put_id = 2;
    writer.send(C::PutBegin(req)).await;
    let w = writer
        .take(|m| match m {
            M::PutGrant(g) => g.window.clone(),
            _ => None,
        })
        .await;
    let mut data: Vec<_> = w
        .packed
        .as_ref()
        .unwrap()
        .units
        .iter()
        .flat_map(|u| payload(u).iter().copied())
        .collect();
    data[30] ^= 1;
    writer.shm.as_ref().unwrap().write(&w, &data);
    writer
        .send(C::PutFilled(v2::PutFilled {
            put_id: 2,
            index: w.index,
            ..Default::default()
        }))
        .await;
    let refusal = writer
        .take(|m| match m {
            M::PutRefused(r) => Some(r.clone()),
            _ => None,
        })
        .await;
    assert_eq!(refusal.reason, v2::Reason::Invalid as i32);
    assert!(refusal.detail.contains("unit"), "{refusal:?}");
    writer
        .send(C::FenceAck(v2::FenceAck {
            subject: Some(v2::fence_ack::Subject::PutId(2)),
            outcome: v2::FenceOutcome::Drained as i32,
        }))
        .await;
    assert!(matches!(
        writer
            .put_record(3, key(128), None, Default::default())
            .await,
        support::pool::PutResult::Stored(_)
    ));
    drop((writer, pool));
    host.stop();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn notices_snapshots_and_stale_claims_filter_the_whole_chain() {
    let mut host = Host::dram_only(1 << 20);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut source = open(&mut pool, "source", true).await;
    let mut legacy = open(&mut pool, "legacy", false).await;
    let mut modern = open(&mut pool, "modern", true).await;
    let parent = pool::entry(128, None, v2::Tier::Dram);
    let child = pool::entry(192, Some(128), v2::Tier::Dram);
    source.snapshot(vec![parent.clone(), child.clone()]).await;
    source.barrier().await;
    let lookup = |id| v2::LookupRequest {
        lookup_id: id,
        boundaries: vec![v2::PrefixKey {
            consumed_tokens: 0,
            digest: pool::bucket_of(192),
        }],
        before_tokens: 193,
        ..Default::default()
    };
    let candidate = legacy
        .lookup_request(lookup(1))
        .await
        .candidates
        .into_iter()
        .find(|c| c.key == child.key)
        .expect("raw chain visible to legacy");
    let mut packed_parent = parent.clone();
    let mut raw_ident = Ident::new("raw-only");
    raw_ident.codec = Some(packed::raw_reader());
    let mut raw = pool.open(&raw_ident).await;
    raw.snapshot(vec![]).await;
    assert!(!raw.lookup_request(lookup(1)).await.candidates.is_empty());
    packed_parent.packed = true;
    source
        .notices(vec![pool::saved(packed_parent.clone())])
        .await;
    source.barrier().await;
    assert!(raw.lookup_request(lookup(2)).await.candidates.is_empty());
    assert!(!modern.lookup_request(lookup(9)).await.candidates.is_empty());
    let mut put = begin(false);
    for unit in &mut put.packed.as_mut().unwrap().units {
        unit.representation = 0;
        unit.profile.clear();
        unit.stored_length = unit.original_length;
        unit.stored_crc = 0;
        unit.original_crc = 0;
    }
    raw.send(C::PutBegin(put)).await;
    let reply = raw.take(|m| match m {
        M::PutRefused(r) => Some(r.detail.as_str() == "packed put requires a negotiated profile"),
        M::PutAccept(_) => Some(false),
        _ => None,
    }).await;
    assert!(reply);
    drop(raw);
    packed_parent.required_profiles = vec![vec![3; 32]];
    source
        .notices(vec![pool::saved(packed_parent.clone())])
        .await;
    source.barrier().await;
    assert!(legacy.lookup_request(lookup(2)).await.candidates.is_empty());
    assert!(
        modern
            .lookup_request(lookup(1))
            .await
            .candidates
            .iter()
            .any(|c| c.key == child.key)
    );
    legacy.claim(legacy.claim_for(1, &candidate, 0, true)).await;
    assert_eq!(
        legacy.claim_terminal(1).await.reason,
        v2::ClaimReason::ChainGap as i32
    );
    // A legacy source cannot install a packed-only entry by sending a new-format snapshot.
    legacy.snapshot(vec![packed_parent.clone()]).await;
    legacy.barrier().await;
    source.snapshot(vec![]).await;
    source.barrier().await;
    assert!(modern.lookup_request(lookup(2)).await.candidates.is_empty());
    drop((source, modern, legacy, pool));
    host.stop();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_peer_relays_just_the_selected_opaque_units() {
    let mut host = Host::dram_only(1 << 20);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut source = open(&mut pool, "source", true).await;
    let mut reader = open(&mut pool, "reader", true).await;
    let mut entry = pool::entry(128, None, v2::Tier::Dram);
    entry.required_profiles = vec![vec![3; 32]];
    source.snapshot(vec![entry]).await;
    source.barrier().await;
    let candidate = reader
        .lookup(1, 193, 64)
        .await
        .candidates
        .into_iter()
        .next()
        .unwrap();
    reader
        .claim(reader.claim_for(1, &candidate, 64, true))
        .await;
    let request = source.export_request().await;
    assert_eq!(request.receiver_codec, reader.opened.codec);
    let mut suffix = offer();
    suffix.units.remove(0);
    suffix.first_page = 1;
    let record = v2::Record {
        key: candidate.key,
        first_page: 1,
        pages: 1,
        row: true,
        packed: Some(suffix.clone()),
    };
    source.pin_ok(&request, vec![record], true).await;
    let granted = reader.claim_granted(1).await;
    assert_eq!(granted.records[0].packed.as_ref(), Some(&suffix));
    let mut seen = 0;
    while seen < 2 {
        for w in source.export_fill().await.windows {
            let manifest = w.packed.as_ref().unwrap();
            assert_eq!(manifest.units, std::slice::from_ref(&suffix.units[seen]));
            let data = payload(&manifest.units[0]);
            source.shm.as_ref().unwrap().write(&w, data);
            source
                .send(C::WindowFilled(v2::WindowFilled {
                    export_id: request.export_id,
                    index: w.index,
                    status: v2::FillStatus::FillOk as i32,
                    ..Default::default()
                }))
                .await;
            let got = reader.window_ready(1).await.window.unwrap();
            let bytes = reader.shm.as_ref().unwrap().read(&got);
            packed::verify_window(&manifest.units, got.packed.as_ref().unwrap(), &bytes).unwrap();
            assert_eq!(bytes, data);
            reader.released(1, got.index).await;
            seen += 1;
        }
    }
    reader.copy_result(1, v2::CopyOutcome::CopyOk).await;
    source
        .export_end(
            request.export_id,
            v2::ExportEndStatus::ExportComplete,
            v2::FenceOutcome::Drained,
        )
        .await;
    assert_eq!(
        reader.claim_terminal(1).await.state,
        v2::ClaimState::Copied as i32
    );
    reader.fence(1, v2::FenceOutcome::Drained).await;
    drop((source, reader, pool));
    host.stop();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_row_larger_than_staging_uses_only_its_stored_length() {
    let bytes = include_bytes!("../../selene/tests/fixtures/codec/large-row.sgc");
    let base = support::base("packed-staging");
    let mut cfg = selene_manager::store::Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        4096,
        2 << 20,
    );
    cfg.name = support::instance_name("packed-staging");
    cfg.staging = 8 << 20;
    let mut host = Host::open_config(cfg);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut id = Ident::new("large-row");
    let geom = id.scope.geometry.as_mut().unwrap();
    geom.ranks[0].row_bytes = 9 << 20;
    geom.row_record_bytes = 9 << 20;
    id.codec = Some(v2::CodecCaps {
        envelope_revision: 4,
        receive_profiles: vec![vec![3; 32]],
    });
    let mut writer = pool.open(&id).await;
    writer.snapshot(vec![]).await;
    let mut row = offer().units.pop().unwrap();
    row.stored_length = bytes.len() as u64;
    row.original_length = 9 << 20;
    row.stored_crc = 0x96bd12e4;
    row.original_crc = 0x63214bad;
    let req = v2::PutBegin {
        key: Some(key_to(key(17))),
        put_id: 1,
        mode: v2::CopyMode::SharedWindows as i32,
        row: true,
        bucket: vec![4; 16],
        packed: Some(v2::PackedOffer {
            first_page: 0,
            units: vec![row.clone()],
        }),
        ..Default::default()
    };
    writer.send(C::PutBegin(req)).await;
    let grant = writer
        .take(|m| match m {
            M::PutGrant(g) => g.window.clone(),
            M::PutRefused(r) => panic!("{r:?}"),
            _ => None,
        })
        .await;
    assert_eq!(grant.length, 6352);
    assert_eq!(grant.region, v2::Region::Staging as i32);
    assert!(grant.row);
    writer.shm.as_ref().unwrap().write(&grant, bytes);
    writer
        .send(C::PutFilled(v2::PutFilled {
            put_id: 1,
            index: grant.index,
            ..Default::default()
        }))
        .await;
    writer
        .send(C::PutCommit(v2::PutCommit {
            put_id: 1,
            ..Default::default()
        }))
        .await;
    writer
        .take(|m| match m {
            M::Stored(_) => Some(()),
            M::PutRefused(r) => panic!("{r:?}"),
            _ => None,
        })
        .await;
    writer
        .send(C::FenceAck(v2::FenceAck {
            subject: Some(v2::fence_ack::Subject::PutId(1)),
            outcome: v2::FenceOutcome::Drained as i32,
        }))
        .await;
    let candidate = writer.manager_candidate(4, 17).await;
    writer.claim(writer.claim_for(1, &candidate, 0, true)).await;
    let granted = writer.claim_granted(1).await;
    assert_eq!(granted.total_pages, 0);
    let w = writer.window_ready(1).await.window.unwrap();
    let data = writer.shm.as_ref().unwrap().read(&w);
    packed::verify_window(&[row], w.packed.as_ref().unwrap(), &data).unwrap();
    assert_eq!(data, bytes);
    writer.released(1, w.index).await;
    writer.copy_result(1, v2::CopyOutcome::CopyOk).await;
    writer.claim_terminal(1).await;
    writer.fence(1, v2::FenceOutcome::Drained).await;
    drop((writer, pool));
    host.stop();
}
