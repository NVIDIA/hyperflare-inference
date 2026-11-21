// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    Job, PageCopier, Reply,
    codec_copy::{Action, Packet},
    memory::MemoryStore,
    shared::wire::PackedUnit,
    slab::SlabPool,
    testing::{Ids, SimulatedGpu, checkpoint},
};

const PAGE: &[u8] = include_bytes!("../../../../tests/fixtures/codec/page0.sgc");
const ROW: &[u8] = include_bytes!("../../../../tests/fixtures/codec/row.sgc");

struct MirrorCopier {
    copies: SimulatedGpu,
    packed: bool,
}

impl PageCopier for MirrorCopier {
    fn layout(&self) -> &Layout {
        self.copies.layout()
    }

    fn encode_mirrors(&self) -> bool {
        self.packed
    }

    fn submit(&mut self, job: Job) -> Result<()> {
        if let Some(window) = &job.codec {
            let Action::Mirror { ranks, .. } = window.0.as_ref() else {
                panic!("expected a mirror");
            };
            let packets = [(0, PAGE, 0x476cc8d3), (1, ROW, 0xb2b29b40)]
                .into_iter()
                .map(|(kind, data, original_crc)| Packet {
                    unit: PackedUnit {
                        kind,
                        original_crc,
                        stored_crc: crate::shared::packed::crc32(data),
                        representation: 1,
                        profile: vec![3; 32],
                        stored_length: data.len() as u64,
                        original_length: 4096,
                        ..Default::default()
                    },
                    data: data.into(),
                })
                .collect();
            ranks[0].set(packets).unwrap();
        }
        self.copies.submit(job)
    }

    fn poll(&mut self, replies: &mut Vec<Reply>) {
        self.copies.poll(replies);
    }

    fn in_flight(&self) -> usize {
        self.copies.in_flight()
    }

    fn close(&mut self) -> Result<()> {
        self.copies.close()
    }
}

fn retirement_schedule(packed: bool) -> Vec<u64> {
    let mut rank = geometry(64, &[("k", 4096)], 4096);
    rank.placement = if packed {
        crate::offload::KvCompression::Cpu
    } else {
        crate::offload::KvCompression::Off
    };
    let geometry = [rank];
    let layout = Layout::new(&geometry).unwrap();
    let (copies, gpu) = SimulatedGpu::new(&geometry).unwrap();
    let page: Vec<u8> = (0..4096).map(|i| (i / 8 % 2) as u8).collect();
    let row: Vec<u8> = (0..4096).map(|i| ((i / 12 + 2) % 4) as u8).collect();
    gpu.pages.lock().unwrap().insert(0, page.clone());
    gpu.rows.lock().unwrap().insert(0, row.clone());
    let copier = Arc::new(Mutex::new(MirrorCopier { copies, packed }));
    let slabs = SlabPool::new(4096, 12, 5).unwrap();
    let memory = MemoryStore::new([7; 32], slabs.clone(), copier).unwrap();
    let state = Arc::new(Mutex::new(PoolState {
        ready: true,
        ..Default::default()
    }));
    let port = FakePool(
        state.clone(),
        Announcer::new(Arc::new(tokio::sync::Notify::new())),
    );
    let mut store = PooledStore::new(memory, port, identity(), &layout);
    let mut ids = Ids::new();
    let mut events = Vec::new();
    let mut schedule = Vec::new();
    store.poll(&mut events).unwrap();
    for seed in 0..16 {
        let mut history = TestHistory::new([7; 32]);
        history.append(&[seed; 64], &[]).unwrap();
        let record = checkpoint(
            &mut ids,
            &history,
            64,
            None,
            vec![PageId(0)],
            1,
            StateSlotId(0),
        );
        let outcome = store.store(&record).unwrap();
        assert_eq!(
            outcome,
            StoreOutcome::Queued,
            "packed={packed}, capture={seed}, puts={}, slots in use={}",
            store.stats().retirement_paths,
            store.stats().slots_used
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while store.inner.announced_sums(&record.key).is_none() {
            store.poll(&mut events).unwrap();
            assert!(
                Instant::now() < deadline,
                "capture {seed} was never announced"
            );
            std::thread::yield_now();
        }
        let puts = std::mem::take(&mut state.lock().unwrap().puts);
        if !puts.is_empty() {
            store.poll(&mut events).unwrap();
            assert!(store.stats().dram_retirement_wait_ns > 0);
        }
        for (id, request) in puts {
            let PutSource::Host(mut source) = request.source;
            assert_eq!(source.packed().is_some(), packed);
            let expected = if packed {
                [PAGE, ROW].concat()
            } else {
                [page.as_slice(), row.as_slice()].concat()
            };
            let mut bytes = vec![0; expected.len()];
            if packed {
                source.fill_packed(0, 1, true, &mut bytes).unwrap();
            } else {
                source.fill(0, 1, true, &mut bytes).unwrap();
            }
            assert_eq!(bytes, expected);
            drop(source);
            state.lock().unwrap().events.push(PoolEvent::Put {
                id,
                outcome: PutOutcome::Stored {
                    bytes: bytes.len() as u64,
                    pages: 1,
                    linked: None,
                },
            });
        }
        store.poll(&mut events).unwrap();
        assert_eq!(store.stats().dram_retirement_wait_ns, 0);
        schedule.push(store.stats().retirement_paths);
        events.clear();
    }
    assert_eq!(store.stats().store_skipped, 0);
    assert_eq!(store.stats().memory_records, 1);
    assert_eq!(slabs.free_slots(), 10);
    store.close().unwrap();
    schedule
}

#[test]
fn packed_and_raw_mirrors_retire_at_the_same_dram_watermark() {
    let raw = retirement_schedule(false);
    assert_eq!(raw, (0..16).collect::<Vec<_>>());
    assert_eq!(retirement_schedule(true), raw);
}
