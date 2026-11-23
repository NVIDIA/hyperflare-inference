/*
 * SPDX-License-Identifier: Apache-2.0
 */

#[path = "packed/at_rest.rs"]
mod at_rest;
mod support;

use flare_selene::shared::{frame::ToInstance, packed, record::Header};
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
use tokio::io::AsyncReadExt;

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

async fn raw_put(
    writer: &mut Session,
    tokens: u64,
    parent: Option<u64>,
    payload: &[u8],
    row_bytes: usize,
) -> v2::WindowSpec {
    use flare_selene::record::checksum;
    let pages = (payload.len() - row_bytes) / 4096;
    let mut descriptor = support::descriptor();
    descriptor.ranks[0].row_bytes = row_bytes as u64;
    let mut header = selene_manager::record::new_header(
        &descriptor,
        key(tokens),
        parent.map(key),
        [4; 16],
        pages as u32,
    )
    .unwrap();
    writer
        .send(C::PutBegin(v2::PutBegin {
            put_id: 1,
            key: Some(key_to(key(tokens))),
            row: true,
            chain_pages: tokens / 64,
            published_tail_pages: pages as u64,
            published_parent: parent.map(|p| key_to(key(p))),
            has_published_parent: parent.is_some(),
            bucket: vec![4; 16],
            capture_role: v2::CaptureRole::Endpoint as i32,
            mode: v2::CopyMode::SharedWindows as i32,
            ..Default::default()
        }))
        .await;
    let accept = writer
        .take(|m| match m {
            M::PutAccept(a) => Some(a.clone()),
            M::PutRefused(r) => panic!("{r:?}"),
            _ => None,
        })
        .await;
    header =
        header.with_capture_role(flare_selene::CaptureRole::try_from(accept.capture_role).unwrap());
    assert_eq!(accept.record_bytes, header.length(true));
    let sums: Vec<_> = payload[..pages * 4096]
        .chunks_exact(4096)
        .map(checksum)
        .collect();
    header.row_checksum = checksum(&payload[pages * 4096..]);
    let mut metadata = vec![0; header.page_segment as usize];
    metadata[..4096].copy_from_slice(&header.encode().unwrap());
    for (entry, hash) in metadata[4096..].chunks_exact_mut(8).zip(&sums) {
        entry.copy_from_slice(&hash.to_le_bytes());
    }
    let grant = writer
        .take(|m| match m {
            M::PutGrant(g) => g.window.clone(),
            _ => None,
        })
        .await;
    assert_eq!(grant.length as usize, payload.len());
    writer.shm.as_ref().unwrap().write(&grant, payload);
    writer
        .send(C::PutFilled(v2::PutFilled {
            put_id: 1,
            index: grant.index,
            page_xxh3: sums,
            row_xxh3: header.row_checksum,
            has_row: true,
            ..Default::default()
        }))
        .await;
    writer
        .send(C::PutCommit(v2::PutCommit {
            put_id: 1,
            metadata,
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
    grant
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
            let ToInstance::PackedWindow(s, w) = ToInstance::read(c, 64).await.unwrap() else {
                panic!("missing packed window")
            };
            let mut data = vec![0; s.length as usize];
            c.read_exact(&mut data).await.unwrap();
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
    raw_put(&mut writer, 128, None, &fixture(), 4096).await;
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
        [PAGE0, PAGE1, ROW].concat()
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
    raw_put(&mut writer, 192, Some(64), &fixture(), 4096).await;
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
    assert_eq!(layout.row_bytes(), ROW.len() as u64);
    assert_eq!(
        &bytes[h.page_segment as usize..h.row_segment as usize],
        [PAGE0, PAGE1].concat()
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
async fn a_row_larger_than_staging_uses_only_its_stored_length() {
    let bytes = include_bytes!("../../selene/tests/fixtures/codec/large-row.sgc");
    let mut host = Host::with_storage(512 << 20, 128 << 20);
    host.config.compress_at_rest = true;
    host.config.staging = 8 << 20;
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let id = ident("large-row", 9 << 20);
    let mut writer = pool.open(&id).await;
    writer.snapshot(vec![]).await;
    raw_put(&mut writer, 17, None, &vec![0; 9 << 20], 9 << 20).await;
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
