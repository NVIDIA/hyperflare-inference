/*
 * SPDX-License-Identifier: Apache-2.0
 */

#[path = "packed/at_rest.rs"]
mod at_rest;
mod support;

use flare_selene::shared::{codec::Plan, frame::ToInstance, packed, record::Header};
use selene_manager::proto::v2;
use support::{
    Host,
    channel::{self, Security},
    key,
    pool::{self, Ident, Pool, Session},
};
use std::sync::OnceLock;
use tokio::io::AsyncReadExt;

fn fixture_packets() -> &'static [(Vec<u8>, u32); 3] {
    static PACKETS: OnceLock<[(Vec<u8>, u32); 3]> = OnceLock::new();
    PACKETS.get_or_init(|| {
        let plan = Plan::new(4096, &[]).unwrap();
        let mut back = vec![0u8; 4096];
        std::array::from_fn(|i| {
            let raw: Vec<u8> = (0..4096)
                .map(|j| ((j / (8 + 2 * i) + i) % (2 + i)) as u8)
                .collect();
            let pkt = plan.encode_verified(&raw, &mut back, None).unwrap();
            (pkt, packed::crc32(&raw))
        })
    })
}

fn page0() -> &'static [u8] {
    &fixture_packets()[0].0
}

fn page1() -> &'static [u8] {
    &fixture_packets()[1].0
}

fn row() -> &'static [u8] {
    &fixture_packets()[2].0
}

fn payload(unit: &v2::PackedUnit) -> &'static [u8] {
    let packets = fixture_packets();
    packets
        .iter()
        .find(|(_, crc)| *crc == unit.original_crc)
        .map(|(pkt, _)| pkt.as_slice())
        .expect("unexpected unit")
}

fn offer() -> v2::PackedOffer {
    let packets = fixture_packets();
    let units = [(0, 0, packets[0].1), (0, 1, packets[1].1), (1, 0, packets[2].1)]
        .into_iter()
        .map(|(kind, logical_index, original_crc)| {
            let mut u = v2::PackedUnit {
                kind,
                logical_index,
                original_crc,
                representation: 1,
                profile: ident("fixture", 4096)
                    .codec
                    .unwrap()
                    .receive_profiles
                    .remove(0),
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

fn ident(name: &str, row_bytes: u64) -> Ident {
    let mut ident = Ident::new(name);
    let mut g = flare_selene::testing::geometry(64, &[("kv", 4096)], row_bytes);
    for plane in g.planes.iter().chain([&g.row]) {
        g.codec_fields.push(flare_selene::geometry::CodecField {
            buffer: plane.name.clone(),
            offset: 0,
            length: plane.unit_bytes,
            encoding: 0,
        });
    }
    ident.codec = Some(flare_selene::codec_copy::reader_caps(
        g.codec_profile(),
        false,
    ));
    ident.codec_layout = Some((&g).into());
    let geometry = ident.scope.geometry.as_mut().unwrap();
    geometry.ranks[0].row_bytes = row_bytes;
    geometry.row_record_bytes = row_bytes;
    ident
}

async fn open(pool: &mut Pool, name: &str, capable: bool) -> Session {
    let ident = if capable {
        ident(name, 4096)
    } else {
        Ident::new(name)
    };
    let mut session = pool.open(&ident).await;
    assert_eq!(session.opened.codec, ident.codec);
    session.snapshot(vec![]).await;
    session
}

async fn put_fixture(
    writer: &mut Session, tokens: u64, parent: Option<u64>, payload: &[u8], row_bytes: usize,
) -> v2::WindowSpec {
    let mut descriptor = support::descriptor();
    descriptor.ranks[0].row_bytes = row_bytes as u64;
    let options = support::store_client::PutOptions {
        role: flare_selene::CaptureRole::Endpoint,
        ..Default::default()
    };
    let outcome = support::put::put(writer, &descriptor, 1, key(tokens), parent.map(key),
        options, None, Some(payload)).await;
    let support::put::Outcome::Stored { first_window: Some(window), .. } = outcome else {
        panic!("fixture put did not store its record");
    };
    support::put::fence(writer, 1).await;
    window
}

fn fixture() -> Vec<u8> {
    (0..3)
        .flat_map(|i| (0..4096).map(move |j| ((j / (8 + 2 * i) + i) % (2 + i)) as u8))
        .collect()
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
            let ToInstance::Window(window) = ToInstance::read(c, 64).await.unwrap() else {
                panic!("missing packed window")
            };
            let mut data = vec![0; window.shape.length as usize];
            c.read_exact(&mut data).await.unwrap();
            (window.shape.index, window.packed.unwrap().units, data)
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
        1 << 20,
        64 << 20,
    );
    config.compress_at_rest = true;
    config.name = support::instance_name("packed");
    let trace = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    config.faults.trace = Some(trace.clone());
    config.listen = Some(support::net::free_address());
    config.listen_data = Some(support::net::free_address());
    let mut host = Host::open_config(config.clone());
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut legacy = open(&mut pool, "legacy", false).await;
    let mut writer = open(&mut pool, "writer", true).await;
    put_fixture(&mut writer, 128, None, &fixture(), 4096).await;
    at_rest::force_eviction(&mut writer).await;
    let mut rdma_ident = Ident::new("rdma-reader");
    rdma_ident.codec = Some(flare_selene::codec_copy::reader_caps(Some([3; 32]), true));
    let mut rdma = pool.open(&rdma_ident).await;
    rdma.snapshot(vec![]).await;
    let miss = rdma
        .lookup_request(v2::LookupRequest {
            lookup_id: 8,
            boundaries: vec![v2::PrefixKey {
                consumed_tokens: 0,
                digest: vec![4; 16],
            }],
            before_tokens: 129,
            ..Default::default()
        })
        .await;
    assert!(miss.candidates.is_empty());
    drop(rdma);
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
    drop((legacy, writer, reader, pool));
    host.stop();
    let dir = root.join(flare_selene::record::directory_name(&[7; 32], &[3; 16]));
    let file = dir.join(flare_selene::record::file_name(&key(128)));
    let bytes = std::fs::read(&file).unwrap();
    let h = Header::decode(&bytes).unwrap();
    assert_eq!(bytes.len() as u64, h.length(true));
    assert_eq!(
        &bytes[h.page_segment as usize..],
        [page0(), page1(), row()].concat()
    );
    selene_manager::record::verify_record(&bytes, &h, true).unwrap();
    let mut second = Host::open_config(config);
    second.run();
    let mut pool = Pool::connect(&second.config.socket).await;
    let mut reader = open(&mut pool, "reopen", true).await;
    read(&mut reader, 2, true).await;
    drop((pool, reader));
    second.stop();
    let trace = trace.lock().unwrap();
    assert!(trace.contains(&b'w'), "no direct record write: {trace:?}");
    assert!(
        trace.iter().filter(|&&op| op == b'r').count() >= 3,
        "expected direct metadata, suffix and final row reads: {trace:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recovery_keeps_the_packed_directory_after_row_truncation() {
    let base = support::base("packed-cut");
    let root = base.join("records");
    let mut config = selene_manager::store::Config::new(
        root.clone(),
        base.join("run/m.sock"),
        1 << 20,
        64 << 20,
    );
    config.compress_at_rest = true;
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
    put_fixture(&mut writer, 192, Some(64), &fixture(), 4096).await;
    at_rest::force_eviction(&mut writer).await;
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
    let layout = h.packed.as_ref().unwrap();
    assert_eq!(layout.offer().first_page, 1);
    assert_eq!(h.length(true), bytes.len() as u64);
    assert_eq!(layout.row_bytes(), row().len() as u64);
    assert_eq!(
        &bytes[h.page_segment as usize..h.row_segment as usize],
        [page0(), page1()].concat()
    );
    assert_eq!(
        &bytes[..h.page_segment as usize],
        h.packed_metadata().unwrap()
    );
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(h.row_segment).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        bytes[..h.row_segment as usize]
    );
    // A partial row also recovers as page-only while a child keeps the record reachable.
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&row()[..37])
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
        [
            (key(64), true),
            (key(192), false),
            (key(256), true),
            (key(8192), true)
        ]
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
    let (p0, p1, r) = (page0().len() as u64, page1().len() as u64, row().len() as u64);
    let cap = p0.max(p1);
    assert!(cap < p0 + p1);
    assert_eq!(
        layout
            .windows(0, true, cap)
            .iter()
            .map(|w| w.length)
            .collect::<Vec<_>>(),
        [p0, p1, r]
    );
    assert_eq!(layout.windows(1, true, p0 + p1 + r)[0].offset, p0);
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
    let mut damaged = page1().to_vec();
    let mid = damaged.len() / 2;
    damaged[mid] ^= 1;
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
async fn a_row_larger_than_staging_uses_only_its_stored_length() {
    let raw = vec![0u8; 9 << 20];
    let mut back = vec![0u8; raw.len()];
    let bytes = Plan::new(raw.len(), &[])
        .unwrap()
        .encode_verified(&raw, &mut back, None)
        .unwrap();
    let mut host = Host::with_storage(512 << 20, 128 << 20);
    host.config.compress_at_rest = true;
    host.config.staging = 8 << 20;
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let id = ident("large-row", 9 << 20);
    let mut writer = pool.open(&id).await;
    writer.snapshot(vec![]).await;
    put_fixture(&mut writer, 17, None, &vec![0; 9 << 20], 9 << 20).await;
    drop((writer, pool));
    host.restart();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut writer = pool.open(&id).await;
    writer.snapshot(vec![]).await;
    let candidate = writer.manager_candidate(4, 17).await;
    writer.claim(writer.claim_for(1, &candidate, 0, true)).await;
    let granted = writer.claim_granted(1).await;
    assert_eq!(granted.total_pages, 0);
    let w = writer.window_ready(1).await.window.unwrap();
    let data = writer.shm.as_ref().unwrap().read(&w);
    let expected = &granted.records[0].packed.as_ref().unwrap().units;
    packed::verify_window(expected, w.packed.as_ref().unwrap(), &data).unwrap();
    assert_eq!(data, bytes);
    writer.released(1, w.index).await;
    writer.copy_result(1, v2::CopyOutcome::CopyOk).await;
    writer.claim_terminal(1).await;
    writer.fence(1, v2::FenceOutcome::Drained).await;
    drop((writer, pool));
    host.stop();
}
