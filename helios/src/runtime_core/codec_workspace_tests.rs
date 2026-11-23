// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::offload::{KvCompression, PrefixCacheOffload};

#[test]
#[ignore = "requires one GPU for memory-limited cache allocation"]
fn memory_limited_cache_holds_codec_before_kv() {
    pool_disk_cache(false);
}

#[test]
#[ignore = "requires one GPU for memory-limited cache allocation"]
fn gpu_workspace_refusal_restores_a_packed_file_on_the_cpu() {
    pool_disk_cache(true);
}

#[test]
#[ignore = "requires one GPU for memory-limited cache allocation"]
fn compression_off_preserves_raw_capacity_with_a_codec_layout() {
    let options = PrefixCacheOffload {
        memory_bytes: Some(16 << 20),
        ..Default::default()
    };
    let raw = memory_limited_gpu_cache(options.clone(), false, false, None);
    let with_layout = memory_limited_gpu_cache(options, false, true, None);
    assert_eq!(with_layout, raw);
}

#[test]
#[ignore = "requires one GPU for a raw disk round trip"]
fn core_without_layout_captures_and_restores_raw_with_decode_requested() {
    memory_limited_gpu_cache(disk_options(), false, false, None);
    memory_limited_gpu_cache(disk_options(), false, true, None);
}

#[test]
#[ignore = "requires one GPU for memory-limited cache allocation"]
fn dram_reserves_codec_memory_only_for_the_pool() {
    let mut options = PrefixCacheOffload {
        memory_bytes: Some(16 << 20),
        compression: KvCompression::Gpu,
        ..Default::default()
    };
    memory_limited_gpu_cache(options.clone(), false, true, None);
    let manager = selene_manager::testing::Manager::start();
    options.pool = Some(crate::offload::PoolOptions {
        address: manager.address(),
        ca: None,
    });
    memory_limited_gpu_cache(options, false, true, None);
}

fn pool_disk_cache(fail: bool) {
    let mut manager = selene_manager::testing::Manager::new();
    manager.config.root = Some(manager.base.join("records"));
    manager.config.nvme = 128 << 20;
    let mut options = disk_options();
    options.pool = Some(crate::offload::PoolOptions {
        address: manager.address(),
        ca: None,
    });
    memory_limited_gpu_cache(options, fail, true, Some(manager));
}

fn disk_options() -> PrefixCacheOffload {
    PrefixCacheOffload {
        disk: Some(std::env::temp_dir().join(format!("flare-packed-restore-{}", std::process::id()))),
        disk_limit: Some(64 << 20),
        compression: KvCompression::Gpu,
        ..Default::default()
    }
}

fn memory_limited_gpu_cache(
    prefix_cache: PrefixCacheOffload,
    fail: bool,
    has_layout: bool,
    mut manager: Option<selene_manager::testing::Manager>,
) -> (u64, u64) {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _lock = LOCK.lock().unwrap();
    use super::super::{MemoryLimit, driver::Driver, protobuf::wire};
    use crate::model::contract::{CodecField, CodecLayout};
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
    let placement = prefix_cache.compression;
    let directory = prefix_cache.disk.clone();
    let active_codec = has_layout && placement != KvCompression::Off && prefix_cache.pool.is_some();
    let progress = StartupProgress::silent();
    unsafe { rank.allocate(&core, &progress.phase("test buffers", None)) }.unwrap();
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
                "identity": "00".repeat(32), "speculative_widths": [], "model_contract":contract}),
    };
    let options = LoadOptions {
        memory_limit: Some(MemoryLimit::Bytes(cap)),
        prefix_cache,
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
    let identity = super::offload_identity(&model.server_config).unwrap();
    let mut packed_history = crate::cache::PrefixHistory::new(identity);
    packed_history.append(&vec![5; 3072], &[]).unwrap();
    let packed_key = packed_history.key_at(3072).unwrap();
    let original_page = vec![0x53u8; unit];
    let mut state = 37u32;
    let literal_page: Vec<u8> = (0..unit)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect();
    let original_row = vec![0x2du8; unit];
    if let Some(manager) = &mut manager {
        let directory = manager.config.root.as_ref().unwrap();
        use flare_selene::{
            record,
            shared::wire::{CodecLayout, PackedOffer, PackedUnit},
        };
        let ranks = std::slice::from_ref(&preflight);
        let fingerprint = flare_selene::geometry::fingerprint(ranks);
        let root = directory.join(record::directory_name(&identity, &fingerprint));
        std::fs::create_dir_all(&root).unwrap();
        let raw = record::Header::new(
            ranks,
            identity,
            fingerprint,
            packed_key,
            None,
            packed_history.bucket_at(3072).unwrap(),
            3,
            std::time::SystemTime::now(),
        )
        .unwrap();
        // The writer uses the source layout independently of the reader's reservation.
        let plans = CodecLayout::from(&preflight).plans().unwrap();
        let profile = preflight.checked_codec_profile().unwrap();
        let packets = [
            (0, 0, &original_page),
            (0, 1, &literal_page),
            (0, 2, &original_page),
            (1, 0, &original_row),
        ]
        .map(|(kind, logical_index, bytes)| {
            let unit = PackedUnit {
                kind: kind as i32,
                logical_index,
                ..Default::default()
            };
            flare_selene::testing::checkpoint_packet(&plans[kind], &profile, unit, bytes)
        });
        assert_eq!(
            packets.each_ref().map(|packet| packet.unit.representation),
            [1, 0, 1, 1]
        );
        let header = raw
            .with_packed(PackedOffer {
                first_page: 0,
                units: packets.iter().map(|p| p.unit.clone()).collect(),
            })
            .unwrap();
        let mut file = header.packed_metadata().unwrap();
        for p in packets {
            file.extend_from_slice(&p.data);
        }
        std::fs::write(root.join(record::file_name(&packed_key)), file).unwrap();
        manager.run();
    }
    let layout = flare_selene::Layout::new(std::slice::from_ref(&preflight)).unwrap();
    let device = if active_codec && !fail {
        crate::offload::transfer::workspace_parts(&preflight)
            .unwrap()
            .into_iter()
            .sum::<u64>()
    } else {
        0
    };
    let pinned = unit as u64;
    let expected = if !active_codec {
        original
    } else {
        [original[0] + pinned, original[1] + device]
    };
    let integrated = model.ranks[0].integrated;
    let (other_device, other_host) = options
        .prefix_cache
        .headroom_with_inputs(1, &layout, 1, false, None)
        .unwrap();
    crate::offload::transfer::FAIL_GPU_RESERVATION.set(fail);
    let initial = unsafe { model.ranks[0].memory_snapshot() }.unwrap();
    let mut planned_bytes = 0;
    unsafe {
        model.allocate_caches(
            std::slice::from_mut(&mut core),
            &progress,
            1,
            &mut |capacity: (u64, u64)| {
                if !fail {
                    assert_eq!(crate::offload::transfer::live_scratch(), expected);
                }
                eprintln!(
                    "workspace_refused={fail} held={expected:?} rows={} KV pages={}",
                    capacity.0, capacity.1
                );
                let cache = (capacity.1 + 2) * unit as u64;
                let future = other_device + if integrated { other_host } else { 0 };
                let codec = if !active_codec {
                    0
                } else {
                    device + if integrated { pinned } else { 0 }
                };
                planned_bytes = cache + codec + future + super::super::memory::HEADROOM_BYTES;
                if integrated {
                    planned_bytes += flare_selene::offload::HOST_SERVICE_HEADROOM_BYTES;
                }
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
    if !active_codec {
        assert!(g.codec_fields.is_empty());
        assert!(model.codec_copiers.is_none());
    }
    assert!(g.page_capacity >= 8 && g.page_capacity < maximum as u64);
    assert_eq!(model.codec_workspace_error.is_some(), fail);
    let prepared = crate::offload::prepare_with_inputs(
        &options.prefix_cache,
        Path::new("."),
        identity,
        vec![g.clone()],
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
                .contains("injected GPU decode workspace refusal")
        );
    }
    crate::offload::transfer::FAIL_GPU_RESERVATION.set(true);
    let mut store = crate::offload::open(prepared).unwrap();
    assert!(
        crate::offload::transfer::FAIL_GPU_RESERVATION.replace(false),
        "late GPU decoder allocation"
    );
    let mut history = crate::cache::PrefixHistory::new(identity);
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
    let raw_disk = directory.is_some();
    if raw_disk {
        for (plane, bytes) in [(&g.planes[0], &original_page), (&g.row, &original_row)] {
            unsafe {
                driver.check("cuCtxSetCurrent", (driver.context_set)(model.ranks[0].context)).unwrap();
                driver.check("cuMemcpyHtoD_v2",
                    (driver.upload)(plane.unit_address(0), bytes.as_ptr().cast(), bytes.len())).unwrap();
            }
        }
    }
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
                    other => panic!("unexpected event {other:?}"),
                }
            }
            assert!(std::time::Instant::now() < until);
            std::thread::yield_now();
        }
    };
    wait(store.as_mut());
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
    if raw_disk {
        for (plane, expected) in [(&g.planes[0], &original_page), (&g.row, &original_row)] {
            let mut restored = vec![0; unit];
            unsafe {
                driver.check("cuCtxSetCurrent", (driver.context_set)(model.ranks[0].context)).unwrap();
                driver.check("cuMemcpyDtoH_v2",
                    (driver.download)(restored.as_mut_ptr().cast(), plane.unit_address(1), unit)).unwrap();
            }
            assert_eq!(restored.as_slice(), expected.as_slice());
        }
    }
    if manager.is_some() {
        let lookup = store.begin_lookup(&packed_history).unwrap();
        let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let hit = loop {
            if let Some(hit) = store.lookup_for(Some(&lookup), &packed_history, 1024, u64::MAX) {
                break hit;
            }
            assert!(
                std::time::Instant::now() < until,
                "packed pool suffix missed: {:?}",
                lookup.state()
            );
            std::thread::yield_now();
        };
        assert_eq!(hit.chain.skip_pages, 1);
        store.fetch(hit, InboundTicket {
            key: packed_key,
            pages: vec![PageId(1), PageId(2)],
            slot: StateSlotId(1),
        })
        .unwrap();
        wait(store.as_mut());
        for (plane, id, expected) in [
            (&g.planes[0], 1, &literal_page),
            (&g.planes[0], 2, &original_page),
            (&g.row, 1, &original_row),
        ] {
            let mut restored = vec![0u8; unit];
            unsafe {
                driver.check("cuCtxSetCurrent", (driver.context_set)(model.ranks[0].context)).unwrap();
                driver.check(
                    "cuMemcpyDtoH_v2",
                    (driver.download)(restored.as_mut_ptr().cast(), plane.unit_address(id), unit),
                )
                .unwrap();
            }
            assert!(&restored == expected, "restored unit {id} differs");
        }
    }
    assert!(planned_bytes <= cap);
    assert_eq!(crate::offload::transfer::live_scratch(), expected);
    store.close().unwrap();
    drop(store);
    assert_eq!(crate::offload::transfer::live_scratch(), original);
    if let Some(directory) = directory {
        std::fs::remove_dir_all(directory).unwrap();
    }
    capacity
}
