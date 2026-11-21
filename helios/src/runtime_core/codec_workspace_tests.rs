// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::offload::KvCompression;

#[test]
#[ignore = "requires one GPU for memory-limited cache allocation"]
fn memory_limited_cache_holds_codec_before_kv() {
    memory_limited_gpu_cache(KvCompression::Gpu, false, true);
}

#[test]
#[ignore = "requires one GPU for memory-limited cache allocation"]
fn memory_limited_cache_starts_raw_when_mirror_workspace_fails() {
    memory_limited_gpu_cache(KvCompression::Gpu, true, true);
}

#[test]
#[ignore = "requires one GPU for memory-limited cache allocation"]
fn compression_off_preserves_raw_capacity_with_a_codec_layout() {
    let raw = memory_limited_gpu_cache(KvCompression::Off, false, false);
    let with_layout = memory_limited_gpu_cache(KvCompression::Off, false, true);
    assert_eq!(with_layout, raw);
}

fn memory_limited_gpu_cache(placement: KvCompression, fail: bool, has_layout: bool) -> (u64, u64) {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _lock = LOCK.lock().unwrap();
    use super::super::{MemoryLimit, driver::Driver, protobuf::wire};
    use crate::model::contract::{CodecField, CodecLayout};
    use crate::offload::{KvCompression, PrefixCacheOffload};
    use flare_selene::{
        CaptureRole, CheckpointId, InboundTicket, PageId, PublishedCheckpoint, StateSlotId,
        StoreEvent, StoreOutcome,
    };
    let driver = Driver::load().unwrap();
    let mut arch = [0; 2];
    unsafe {
        driver.check("cuInit", (driver.init)(0)).unwrap();
        for (value, attribute) in arch.iter_mut().zip([75, 76]) {
            driver
                .check(
                    "cuDeviceGetAttribute",
                    (driver.device_attribute)(value, attribute, 0),
                )
                .unwrap();
        }
    }
    let mut core = protobuf::decode(wire::BundleIndex {
        architecture: arch.to_vec(),
        context: Some(Default::default()),
        execution: Some(Default::default()),
        ..Default::default()
    })
    .unwrap();
    let mut rank = unsafe { NativeResources::initialize(&core, 0) }.unwrap();
    let unit = 2 << 20;
    let maximum = 1024;
    let mut contract = crate::model::testing::metadata();
    contract.cache.page_tokens = 1024;
    contract.cache.codec_layout = Some(CodecLayout {
        revision: 1,
        fields: vec![],
    });
    for (name, capacity) in [
        ("state_pool", Some(0)),
        ("kv", Some(1)),
        ("memory_capacity", None),
    ] {
        let nbytes = if capacity.is_some() {
            maximum * unit
        } else {
            unit
        };
        let mut address = 0;
        unsafe {
            driver
                .check("cuCtxSetCurrent", (driver.context_set)(rank.context))
                .unwrap();
            driver
                .check(
                    "cuMemAddressReserve",
                    (driver.reserve)(&mut address, nbytes, 0, 0, 0),
                )
                .unwrap();
        }
        rank.reservations.push((address, nbytes));
        core.allocations.push(super::super::format::Allocation {
            name: name.into(),
            nbytes: if capacity.is_none() { unit } else { 0 },
            alignment: unit,
            address: Some(address),
            memory: Memory::Device,
            sizing: capacity.map(|capacity| wire::Sizing {
                capacity,
                maximum: maximum as u64,
                unit_bytes: unit as u64,
            }),
            data_spans: vec![],
            sources: vec![],
            relocations: vec![],
            file: None,
        });
        if capacity.is_some() {
            contract
                .cache
                .codec_layout
                .as_mut()
                .unwrap()
                .fields
                .push(CodecField {
                    buffer: name.into(),
                    offset: 0,
                    length: unit as u64,
                    encoding: 0,
                });
        }
        let descriptor = if capacity.is_some() {
            (unit as u64, "uint8", vec![1, unit as u64])
        } else {
            (16, "int64", vec![2])
        };
        core.execution.as_mut().unwrap().buffers.push(wire::Buffer {
            name: name.into(),
            nbytes: descriptor.0,
            dtype: descriptor.1.into(),
            shape: descriptor.2,
            reference: Some(wire::Reference {
                allocation: name.into(),
                offset: 0,
            }),
            ..Default::default()
        });
    }
    if !has_layout {
        contract.cache.codec_layout = None;
    }
    let active_codec = placement != KvCompression::Off && !fail;
    let progress = StartupProgress::silent();
    unsafe { rank.allocate(&core, &progress.phase("test buffers", None)) }.unwrap();
    let initial = unsafe { rank.memory_snapshot() }.unwrap();
    let cap = (768 << 20) + if rank.integrated { 2u64 << 30 } else { 0 };
    let mut model = NativeModel {
        _core_files: vec![],
        ranks: vec![rank],
        engram: vec![],
        offload: Default::default(),
        codec_copiers: None,
        codec_workspace_error: None,
        cluster: None,
        roce: vec![],
        server_config: serde_json::json!({"page_tokens":1024, "max_sequence_tokens":8192,
                "model_contract":contract}),
    };
    let options = LoadOptions {
        memory_limit: Some(MemoryLimit::Bytes(cap)),
        prefix_cache: PrefixCacheOffload {
            memory_bytes: Some(16 << 20),
            compression: placement,
            ..Default::default()
        },
        ..Default::default()
    };
    let original = crate::offload::transfer::live_scratch();
    let preflight = super::super::memory::cache_geometry(
        &core,
        1024,
        1,
        1,
        contract.cache.codec_layout.as_ref(),
    )
    .unwrap();
    let layout = flare_selene::Layout::new(std::slice::from_ref(&preflight)).unwrap();
    let device = if active_codec {
        2 * crate::offload::transfer::workspace_parts(&preflight, &layout.ranks[0])
            .unwrap()
            .into_iter()
            .sum::<u64>()
    } else {
        0
    };
    let pinned = 3 * unit as u64;
    let backstop = unit as u64;
    let expected = if !active_codec {
        original
    } else {
        [original[0] + pinned, original[1] + device]
    };
    let integrated = model.ranks[0].integrated;
    let (other_device, other_host) = options
        .prefix_cache
        .headroom_with_inputs(1, &layout, 1, false)
        .unwrap();
    crate::offload::transfer::FAIL_MIRROR_RESERVATION.set(fail);
    unsafe {
        model.allocate_caches(
            std::slice::from_mut(&mut core),
            &progress,
            1,
            &mut |capacity: (u64, u64)| {
                assert_eq!(
                    crate::offload::transfer::live_scratch(),
                    expected,
                    "codec bytes held before mapping KV"
                );
                eprintln!(
                    "workspace_refused={fail} held={expected:?} rows={} KV pages={}",
                    capacity.0, capacity.1
                );
                let cache = (capacity.1 + 2) * unit as u64;
                let future = other_device + if integrated { other_host } else { 0 };
                let codec = if !active_codec {
                    0
                } else {
                    device + if integrated { pinned + backstop } else { 0 }
                };
                assert!(
                    cache
                        + codec
                        + future
                        + super::super::memory::HEADROOM_BYTES
                        + if integrated {
                            flare_selene::offload::HOST_SERVICE_HEADROOM_BYTES
                        } else {
                            0
                        }
                        <= cap
                );
                Ok(capacity)
            },
            &options,
            &[initial],
            0,
        )
    }
    .unwrap();
    let g = model.ranks[0].cache_geometry().unwrap().clone();
    let capacity = (g.row_capacity, g.page_capacity);
    if placement == KvCompression::Off {
        assert!(g.codec_fields.is_empty());
        assert!(model.codec_copiers.is_none());
    }
    assert!(g.page_capacity >= 8 && g.page_capacity < maximum as u64);
    assert_eq!(model.codec_workspace_error.is_some(), fail);
    let prepared = crate::offload::prepare_with_inputs(
        &options.prefix_cache,
        Path::new("."),
        [0; 32],
        vec![g],
        vec![model.ranks[0].context_handle()],
        1,
        0,
        None,
    )
    .unwrap();
    model.attach_offload(prepared).unwrap();
    let prepared = model.offload.shared().unwrap().take().unwrap();
    assert_eq!(prepared.codec_workspace_error.is_some(), fail);
    if fail {
        assert!(
            prepared
                .codec_workspace_error
                .as_ref()
                .unwrap()
                .contains("injected GPU mirror workspace refusal")
        );
    }
    crate::offload::transfer::FAIL_MIRROR_RESERVATION.set(true);
    let mut store = crate::offload::open(prepared).unwrap();
    assert!(
        crate::offload::transfer::FAIL_MIRROR_RESERVATION.replace(false),
        "late mirror allocation"
    );
    assert_eq!(crate::offload::transfer::live_scratch(), expected);
    let mut history = crate::cache::PrefixHistory::new([0; 32]);
    history.append(&[0; 1024], &[]).unwrap();
    let key = history.key_at(1024).unwrap();
    let ckpt = PublishedCheckpoint {
        capture_role: CaptureRole::Message,
        id: CheckpointId::new(1),
        key,
        ancestors: vec![],
        bucket: history.bucket_at(1024).unwrap(),
        chain_pages: vec![PageId(0)],
        tail_pages: 1,
        slot: StateSlotId(0),
    };
    assert_eq!(store.store(&ckpt).unwrap(), StoreOutcome::Queued);
    let wait = |store: &mut dyn flare_selene::HostStore| {
        let until = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let mut events = Vec::new();
            store.poll(&mut events).unwrap();
            for event in events {
                match event {
                    StoreEvent::Stored { result, .. } | StoreEvent::Fetched { result, .. } => {
                        result.unwrap();
                        return;
                    }
                    StoreEvent::SourceReleased { .. } => (),
                    other => panic!("unexpected event {other:?}"),
                }
            }
            assert!(std::time::Instant::now() < until);
            std::thread::yield_now();
        }
    };
    wait(store.as_mut());
    let stats = store.stats();
    assert_eq!(stats.codec_mirrored_original_bytes, 2 * unit as u64);
    assert_eq!(
        stats.backstop.by_encoder,
        if !active_codec {
            [0, 0, 0, 2 * unit as u64]
        } else {
            [2 * unit as u64, 0, 0, 0]
        }
    );
    assert_eq!(
        stats.codec_fallback_bytes.last(),
        Some(&(if fail { 2 * unit as u64 } else { 0 }))
    );
    let hit = store.lookup(&history, 0, u64::MAX).unwrap();
    store
        .fetch(
            hit,
            InboundTicket {
                key,
                pages: vec![PageId(1)],
                slot: StateSlotId(1),
            },
        )
        .unwrap();
    wait(store.as_mut());
    store.close().unwrap();
    drop(store);
    assert_eq!(crate::offload::transfer::live_scratch(), original);
    capacity
}
