// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::{fs, os::unix::fs::FileExt};

async fn writer() -> (Host, Pool, Session, Ident) {
    let mut host = Host::with_storage(1 << 20, 64 << 20);
    host.config.compress_at_rest = true;
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut ident = Ident::new("cpu-writer");
    let mut geometry = flare_selene::testing::geometry(64, &[("kv", 4000)], 1000);
    for plane in geometry.planes.iter().chain([&geometry.row]) {
        geometry
            .codec_fields
            .push(flare_selene::geometry::CodecField {
                buffer: plane.name.clone(),
                offset: 0,
                length: plane.unit_bytes,
                encoding: 0,
            });
    }
    ident.codec = Some(flare_selene::codec_copy::reader_caps(
        geometry.codec_profile(),
        false,
    ));
    ident.codec_layout = Some((&geometry).into());
    let mut writer = pool.open(&ident).await;
    writer.snapshot(vec![]).await;
    (host, pool, writer, ident)
}

// Keep the exact put bytes and the page grant so the integrity test can damage
// the committed arena record without changing its saved checksums.
async fn raw_put(writer: &mut Session) -> (Vec<u8>, v2::WindowSpec) {
    let payload = [vec![23; 8192], vec![87; 4096]].concat();
    let grant = super::put_fixture(writer, 128, None, &payload, 4096).await;
    (payload, grant)
}

pub(super) async fn force_eviction(writer: &mut Session) {
    // A 128-page root rounds to the whole 1 MiB arena.
    assert!(matches!(
        writer
            .put_record(2, key(8192), None, Default::default())
            .await,
        pool::PutResult::Stored(_)
    ));
}

fn file(host: &Host) -> Vec<u8> {
    fs::read(
        host.config
            .root
            .as_ref()
            .unwrap()
            .join(support::descriptor().name())
            .join(flare_selene::record::file_name(&key(128))),
    )
    .unwrap()
}

async fn copied(session: &mut Session, id: u64) {
    session.copy_result(id, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        session.claim_terminal(id).await.state,
        v2::ClaimState::Copied as i32
    );
    session.fence(id, v2::FenceOutcome::Drained).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_put_recovers_and_decodes_through_a_profile_claim() {
    let (mut host, pool, mut session, mut ident) = writer().await;
    let (original, _) = raw_put(&mut session).await;
    force_eviction(&mut session).await;
    drop((session, pool));
    host.restart();
    let bytes = file(&host);
    let header = Header::decode(&bytes).unwrap();
    assert!(header.packed.is_some());
    selene_manager::record::verify_record(&bytes, &header, true).unwrap();
    let plans = ident.codec_layout.take().unwrap().plans().unwrap();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut reader = pool.open(&ident).await;
    reader.snapshot(vec![]).await;
    let candidate = reader.manager_candidate(1, 128).await;
    reader.claim(reader.claim_for(1, &candidate, 0, true)).await;
    let granted = reader.claim_granted(1).await;
    assert!(granted.records[0].packed.is_some());
    let mut restored = Vec::new();
    while restored.len() < original.len() {
        let window = reader.window_ready(1).await.window.unwrap();
        let data = reader.shm.as_ref().unwrap().read(&window);
        let mut remaining = data.as_slice();
        for unit in window.packed.unwrap().units {
            let (packet, tail) = remaining.split_at(unit.stored_length as usize);
            let mut decoded = vec![0; unit.original_length as usize];
            plans[unit.kind as usize]
                .decode_checked(
                    &[packet],
                    &mut decoded,
                    unit.stored_crc,
                    unit.original_crc,
                    None,
                )
                .unwrap();
            restored.extend(decoded);
            remaining = tail;
        }
        assert!(remaining.is_empty());
        reader.released(1, window.index).await;
    }
    copied(&mut reader, 1).await;
    assert_eq!(restored, original);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn writeback_uses_the_profile_after_the_writer_closes() {
    let (host, mut pool, mut writer, mut ident) = writer().await;
    let profile = ident.codec_layout.take().unwrap().profile().unwrap();
    raw_put(&mut writer).await;
    writer.goodbye(true).await;

    // Opening the same instance closes its old session before accepting the new one.
    ident.codec = None;
    let mut raw = pool.open(&ident).await;
    writer.closed().await;
    raw.snapshot(vec![]).await;
    force_eviction(&mut raw).await;

    let bytes = file(&host);
    let header = Header::decode(&bytes).unwrap();
    selene_manager::record::verify_record(&bytes, &header, true).unwrap();
    let profiles = header.packed.map(|layout| layout.required_profiles(true));
    assert_eq!(
        profiles,
        Some(vec![profile.to_vec()]),
        "{}",
        host.metrics()
            .lines()
            .find(|l| l.starts_with("selene_codec_writeback_raw_total{cause=\"no_plan\"}"))
            .unwrap()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn corrupt_arena_payload_keeps_its_raw_checksum() {
    let (host, _pool, mut session, _) = writer().await;
    let (_, window) = raw_put(&mut session).await;
    session
        .shm
        .as_ref()
        .unwrap()
        .arena
        .write_all_at(&[22], window.offset)
        .unwrap();
    force_eviction(&mut session).await;
    let bytes = file(&host);
    let header = Header::decode(&bytes).unwrap();
    assert!(header.packed.is_none());
    assert_eq!(
        selene_manager::record::verify_record(&bytes, &header, true)
            .unwrap_err()
            .to_string(),
        "disk page checksum"
    );
    assert!(
        host.metrics()
            .contains("selene_codec_writeback_raw_total{cause=\"raw_checksum\"} 1\n")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eviction_changes_raw_claim_visibility_and_releases_the_arena_charge() {
    let (host, mut pool, mut session, _) = writer().await;
    let (original, _) = raw_put(&mut session).await;
    let mut legacy = open(&mut pool, "legacy", false).await;
    let candidate = legacy.manager_candidate(1, 128).await;
    legacy.claim(legacy.claim_for(1, &candidate, 0, true)).await;
    assert!(legacy.claim_granted(1).await.records[0].packed.is_none());
    let mut raw = Vec::new();
    while raw.len() < original.len() {
        let window = legacy.window_ready(1).await.window.unwrap();
        raw.extend(legacy.shm.as_ref().unwrap().read(&window));
        legacy.released(1, window.index).await;
    }
    copied(&mut legacy, 1).await;
    assert_eq!(raw, original);
    assert_eq!(host.gauge("nvme_reserved_bytes"), 32768);
    force_eviction(&mut session).await;
    assert_eq!(
        host.gauge("nvme_reserved_bytes"),
        (1 << 20) + file(&host).len() as u64
    );
    let disk = session.manager_candidate(2, 128).await;
    assert_eq!(disk.tier, v2::Tier::Nvme as i32);
    legacy.claim(legacy.claim_for(2, &candidate, 0, true)).await;
    assert_eq!(
        legacy.claim_terminal(2).await.reason,
        v2::ClaimReason::ChainGap as i32
    );
    let reply = legacy
        .lookup_request(v2::LookupRequest {
            lookup_id: 3000,
            boundaries: vec![v2::PrefixKey {
                consumed_tokens: 0,
                digest: vec![4; 16],
            }],
            before_tokens: 129,
            ..Default::default()
        })
        .await;
    assert!(reply.candidates.is_empty());
    host.wait(|h| {
        h.metrics()
            .contains("selene_pool_codec_incompatible_total{action=\"claim\"} 1\n")
    });
}
