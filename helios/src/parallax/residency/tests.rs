use super::*;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct State {
    ready: bool,
    stage_error: Option<Error>,
    fail_mapping: Option<usize>,
    fail_rollback: bool,
    mapping_calls: usize,
    copies: usize,
    published: Vec<Option<Location>>,
}
struct Memory {
    regions: Vec<Region>,
    slots: usize,
    initial: Vec<Location>,
    backing: Vec<bool>,
    region_groups: Vec<usize>,
    slot_groups: Vec<usize>,
    state: Arc<Mutex<State>>,
    dropped: Arc<AtomicUsize>,
}
impl Drop for Memory {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

// The fake backend exercises transaction ownership and fault recovery, not CUDA.
unsafe impl Backend for Memory {
    fn regions(&self) -> &[Region] {
        &self.regions
    }
    fn initial_location(&self, region: usize) -> Location { self.initial[region] }
    fn has_backing(&self, region: usize) -> bool { self.backing[region] }
    fn region_group(&self, region: usize) -> usize { self.region_groups[region] }
    fn slot_group(&self, slot: usize) -> usize { self.slot_groups[slot] }
    fn slots(&self) -> usize {
        self.slots
    }
    fn slot_bytes(&self) -> usize {
        64
    }
    fn backing_bytes(&self) -> usize {
        self.regions.len() * 64
    }
    fn transfer_bytes(&self, _: usize) -> usize {
        64
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            stable_addresses: true,
            asynchronous_staging: true,
            publication_requires_quiescence: true,
            cold_moe_execution_qualified: false,
        }
    }
    fn address(&self, region: usize) -> u64 {
        4096 + self.regions[region].offset as u64
    }
    fn stage(&mut self, copies: &[(usize, usize)]) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if let Some(e) = state.stage_error.take() {
            return Err(e);
        }
        state.copies += copies.len();
        Ok(())
    }
    fn ready(&mut self) -> Result<bool> {
        Ok(self.state.lock().unwrap().ready)
    }
    fn copy_elapsed_ns(&self) -> Result<Option<u64>> { Ok(Some(123)) }
    fn discard(&mut self) -> Result<()> {
        if self.ready()? {
            Ok(())
        } else {
            Err(Error::Busy)
        }
    }
    unsafe fn publish(&mut self, bindings: &[Binding]) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        transaction::publish(bindings, |region, location| {
            state.mapping_calls += 1;
            // Reproduce a failure after the old mapping was already removed.
            state.published[region] = None;
            if state.fail_mapping == Some(state.mapping_calls)
                || (state.fail_rollback
                    && state
                        .fail_mapping
                        .is_some_and(|call| state.mapping_calls > call))
            {
                return Err(Error::Backend("injected map failure".into()));
            }
            state.published[region] = Some(location);
            Ok(())
        })
    }
}
fn expert(expert: usize) -> ExpertId {
    ExpertId { layer: 3, expert }
}
fn setup(slots: usize) -> (Controller<Memory>, Arc<Mutex<State>>, Arc<AtomicUsize>) {
    let (memory, state, dropped) = memory(slots, vec![Location::Backing; 4], vec![true; 4]);
    (Controller::new("synthetic-native".into(), memory).unwrap(), state, dropped)
}
fn memory(slots: usize, initial: Vec<Location>, backing: Vec<bool>) -> (Memory, Arc<Mutex<State>>, Arc<AtomicUsize>) {
    let state = Arc::new(Mutex::new(State {
        ready: true,
        published: initial.iter().copied().map(Some).collect(),
        ..State::default()
    }));
    let dropped = Arc::new(AtomicUsize::new(0));
    let regions = (0..4)
        .map(|i| Region {
            expert: expert(i / 2),
            part: i % 2,
            offset: i * 64,
            bytes: 37,
        })
        .collect();
    let memory = Memory {
            regions,
            slots,
            initial,
            backing,
            region_groups: vec![0; 4],
            slot_groups: vec![0; slots],
            state: state.clone(),
            dropped: dropped.clone(),
        };
    (memory, state, dropped)
}

#[test]
fn initial_hbm_occupancy_preserves_spare_slots_and_reader_fences() {
    let (memory, state, _) = memory(4,
        vec![Location::Hbm(0), Location::Hbm(1), Location::Backing, Location::Backing], vec![true; 4]);
    let mut c = Controller::new("synthetic-native".into(), memory).unwrap();
    assert_eq!(c.snapshot().free_slots, 2);
    assert!(c.snapshot().experts[&expert(0)]);
    let reader = c.acquire_reader().unwrap();
    c.begin(plan(0, &[1], &[0]), 0).unwrap();
    assert_eq!(c.snapshot().free_slots, 0);
    assert_eq!(unsafe { c.progress(1) }.unwrap(), Progress::WaitingForReaders);
    unsafe { reader.complete(); }
    assert_eq!(unsafe { c.progress(2) }.unwrap(), Progress::Committed(1));
    assert_eq!(state.lock().unwrap().published,
        vec![Some(Location::Backing), Some(Location::Backing), Some(Location::Hbm(2)), Some(Location::Hbm(3))]);
    assert_eq!(c.snapshot().free_slots, 2);
}

#[test]
fn large_catalog_exchange_preserves_pool_and_reader_constraints() {
    let layers = 40;
    let experts = 384;
    let rows = 288;
    let parts = 4;
    let mut regions = Vec::new();
    let mut initial = Vec::new();
    let mut region_groups = Vec::new();
    for layer in 0..layers {
        for expert in 0..experts {
            for part in 0..parts {
                regions.push(Region { expert: ExpertId { layer, expert }, part,
                    offset: regions.len() * 64, bytes: 64 });
                initial.push(if expert < rows - 1 {
                    Location::Hbm((layer * rows + expert) * parts + part)
                } else { Location::Backing });
                region_groups.push(layer);
            }
        }
    }
    let state = Arc::new(Mutex::new(State { ready: true,
        published: initial.iter().copied().map(Some).collect(), ..State::default() }));
    let memory = Memory { backing: vec![true; regions.len()], regions, initial, region_groups,
        slots: layers * rows * parts,
        slot_groups: (0..layers * rows * parts).map(|slot| slot / (rows * parts)).collect(),
        state: state.clone(), dropped: Arc::new(AtomicUsize::new(0)) };
    let mut controller = Controller::new("large-catalog".into(), memory).unwrap();
    let plan = Plan { version: 1, catalog_id: "large-catalog".into(), generation: 0,
        valid_until_ns: u64::MAX,
        promote: (0..layers).map(|layer| ExpertId { layer, expert: rows - 1 }).collect(),
        evict: (0..layers).map(|layer| ExpertId { layer, expert: 0 }).collect() };
    let mut samples = Vec::new();
    for _ in 0..3 {
        let started = std::time::Instant::now();
        assert_eq!(controller.begin(plan.clone(), 0).unwrap(), Progress::Staged);
        samples.push(started.elapsed().as_nanos());
        assert_eq!(controller.cancel().unwrap(), Progress::Cancelled);
    }
    println!("CPU reservation samples (ns), fake storage without CUDA: {samples:?}");
    let reader = controller.acquire_reader().unwrap();
    controller.begin(plan, 0).unwrap();
    assert_eq!(controller.snapshot().free_slots, 0);
    assert_eq!(unsafe { controller.progress(1) }.unwrap(), Progress::WaitingForReaders);
    unsafe { reader.complete(); }
    assert_eq!(unsafe { controller.progress(2) }.unwrap(), Progress::Committed(1));
    let published = &state.lock().unwrap().published;
    for layer in 0..layers {
        for part in 0..parts {
            assert_eq!(published[(layer * experts + rows - 1) * parts + part],
                Some(Location::Hbm((layer * rows + rows - 1) * parts + part)));
            assert_eq!(published[layer * experts * parts + part], Some(Location::Backing));
        }
    }
    assert_eq!(controller.snapshot().free_slots, layers * parts);
}

#[test]
fn pinned_initial_expert_cannot_be_evicted_without_backing() {
    let (memory, state, _) = memory(4,
        vec![Location::Hbm(0), Location::Hbm(1), Location::Backing, Location::Backing],
        vec![true, false, true, true]);
    let mut c = Controller::new("synthetic-native".into(), memory).unwrap();
    assert_eq!(c.begin(plan(0, &[1], &[0]), 0),
        Err(Error::Invalid("cannot evict an expert without immutable backing".into())));
    assert_eq!(state.lock().unwrap().copies, 0);
    assert_eq!(c.snapshot().generation, 0);
    assert_eq!(c.snapshot().free_slots, 2);
    c.begin(plan(0, &[1], &[]), 0).unwrap();
    assert_eq!(unsafe { c.progress(1) }.unwrap(), Progress::Committed(1));
    assert!(c.snapshot().experts.values().all(|resident| *resident));
}

#[test]
fn invalid_initial_placements_fail_before_copying() {
    for (initial, backing) in [
        (vec![Location::Hbm(0), Location::Hbm(0), Location::Backing, Location::Backing], vec![true; 4]),
        (vec![Location::Hbm(0), Location::Hbm(4), Location::Backing, Location::Backing], vec![true; 4]),
        (vec![Location::Hbm(0), Location::Backing, Location::Backing, Location::Backing], vec![true; 4]),
        (vec![Location::Backing; 4], vec![true, false, true, true]),
    ] {
        let (memory, state, _) = memory(4, initial, backing);
        assert!(matches!(Controller::new("synthetic-native".into(), memory), Err(Error::Invalid(_))));
        assert_eq!(state.lock().unwrap().copies, 0);
    }
}
fn plan(generation: u64, promote: &[usize], evict: &[usize]) -> Plan {
    Plan {
        version: 1,
        catalog_id: "synthetic-native".into(),
        generation,
        valid_until_ns: 100,
        promote: promote.iter().copied().map(expert).collect(),
        evict: evict.iter().copied().map(expert).collect(),
    }
}

#[test]
fn publication_waits_for_both_transfer_and_readers() {
    let (mut c, state, _) = setup(4);
    let addresses: Vec<_> = (0..2)
        .map(|part| c.address(expert(0), part).unwrap())
        .collect();
    let reader = c.acquire_reader().unwrap();
    state.lock().unwrap().ready = false;
    assert_eq!(c.begin(plan(0, &[0], &[]), 0).unwrap(), Progress::Staged);
    assert_eq!(
        unsafe { c.progress(1) }.unwrap(),
        Progress::WaitingForTransfer
    );
    state.lock().unwrap().ready = true;
    assert_eq!(
        unsafe { c.progress(2) }.unwrap(),
        Progress::WaitingForReaders
    );
    assert_eq!(c.snapshot().generation, 0);
    assert!(
        state
            .lock()
            .unwrap()
            .published
            .iter()
            .all(|v| *v == Some(Location::Backing))
    );
    unsafe {
        reader.complete();
    }
    assert_eq!(unsafe { c.progress(3) }.unwrap(), Progress::Committed(1));
    assert_eq!(c.statistics().ready_wait_ns, 1);
    assert!(c.snapshot().experts[&expert(0)]);
    assert!(!c.snapshot().experts[&expert(1)]);
    assert_eq!(c.snapshot().statistics.staged_bytes, 128); // physical padding is charged
    assert_eq!(
        (0..2)
            .map(|part| c.address(expert(0), part).unwrap())
            .collect::<Vec<_>>(),
        addresses
    );
}

#[test]
fn exchanges_need_spare_capacity_even_when_victims_will_be_evicted() {
    let (mut c, state, _) = setup(2);
    c.begin(plan(0, &[0], &[]), 0).unwrap();
    unsafe { c.progress(1) }.unwrap();
    assert_eq!(c.begin(plan(1, &[1], &[0]), 2), Err(Error::Capacity));
    assert_eq!(state.lock().unwrap().copies, 2);
    assert!(c.snapshot().experts[&expert(0)]);
    c.begin(plan(1, &[], &[0]), 3).unwrap();
    unsafe { c.progress(4) }.unwrap();
    c.begin(plan(2, &[1], &[]), 5).unwrap();
    unsafe { c.progress(6) }.unwrap();
    assert!(c.snapshot().experts[&expert(1)]);
}

#[test]
fn cancellation_and_expiration_retain_slots_until_copy_completion() {
    let (mut c, state, _) = setup(4);
    state.lock().unwrap().ready = false;
    c.begin(plan(0, &[0], &[]), 0).unwrap();
    assert_eq!(c.cancel().unwrap(), Progress::WaitingForTransfer);
    assert_eq!(c.snapshot().free_slots, 2);
    assert_eq!(c.begin(plan(0, &[1], &[]), 1), Err(Error::Busy));
    assert_eq!(
        unsafe { c.progress(101) }.unwrap(),
        Progress::WaitingForTransfer
    );
    state.lock().unwrap().ready = true;
    assert_eq!(unsafe { c.progress(102) }, Err(Error::Expired));
    assert_eq!(c.snapshot().free_slots, 4);
    assert_eq!(c.snapshot().generation, 0);
    c.begin(plan(0, &[1], &[]), 1).unwrap();
    assert_eq!(c.cancel().unwrap(), Progress::Cancelled);
    assert_eq!(c.snapshot().free_slots, 4);
}

#[test]
fn invalid_and_noop_plans_do_not_submit_copies() {
    let (mut c, state, _) = setup(4);
    assert_eq!(c.begin(plan(1, &[0], &[]), 0), Err(Error::Stale));
    assert_eq!(c.begin(plan(0, &[0], &[]), 101), Err(Error::Expired));
    for bad in [
        plan(0, &[0, 0], &[]),
        plan(0, &[0], &[0]),
        plan(0, &[99], &[]),
        plan(0, &[], &[0]),
    ] {
        assert!(matches!(c.begin(bad, 0), Err(Error::Invalid(_))));
    }
    let mut unsupported = plan(0, &[0], &[]);
    unsupported.version = 2;
    assert!(matches!(c.begin(unsupported, 0), Err(Error::Invalid(_))));
    let mut foreign = plan(0, &[0], &[]);
    foreign.catalog_id = "different-package".into();
    assert!(matches!(c.begin(foreign, 0), Err(Error::Invalid(_))));
    assert_eq!(c.begin(plan(0, &[], &[]), 0).unwrap(), Progress::Noop);
    assert_eq!(state.lock().unwrap().copies, 0);
    assert!(!c.snapshot().pending);
}

#[test]
fn recoverable_copy_failure_preserves_capacity_and_allows_retry() {
    let (mut c, state, _) = setup(4);
    state.lock().unwrap().stage_error = Some(Error::Backend("copy failed after drain".into()));
    assert!(matches!(
        c.begin(plan(0, &[0], &[]), 0),
        Err(Error::Backend(_))
    ));
    assert_eq!(c.snapshot().free_slots, 4);
    assert!(!c.snapshot().pending);
    c.begin(plan(0, &[0], &[]), 1).unwrap();
    assert_eq!(unsafe { c.progress(2) }.unwrap(), Progress::Committed(1));
}

#[test]
fn every_partial_exchange_failure_restores_all_previous_mappings() {
    for failure in 1..=4 {
        let (mut c, state, _) = setup(4);
        c.begin(plan(0, &[0], &[]), 0).unwrap();
        unsafe { c.progress(1) }.unwrap();
        let before = state.lock().unwrap().published.clone();
        {
            let mut s = state.lock().unwrap();
            s.mapping_calls = 0;
            s.fail_mapping = Some(failure);
        }
        c.begin(plan(1, &[1], &[0]), 2).unwrap();
        assert!(matches!(unsafe { c.progress(3) }, Err(Error::Backend(_))));
        assert_eq!(state.lock().unwrap().published, before);
        assert_eq!(c.snapshot().generation, 1);
        assert_eq!(c.snapshot().free_slots, 2);
        assert!(!c.snapshot().pending);
        state.lock().unwrap().fail_mapping = None;
        c.begin(plan(1, &[1], &[0]), 4).unwrap();
        assert_eq!(unsafe { c.progress(5) }.unwrap(), Progress::Committed(2));
        assert!(!c.snapshot().experts[&expert(0)]);
        assert!(c.snapshot().experts[&expert(1)]);
    }
}

#[test]
fn rollback_failure_poisoning_blocks_readers_and_retains_storage() {
    let (mut c, state, dropped) = setup(4);
    {
        let mut s = state.lock().unwrap();
        s.fail_mapping = Some(2);
        s.fail_rollback = true;
    }
    c.begin(plan(0, &[0], &[]), 0).unwrap();
    assert!(matches!(unsafe { c.progress(1) }, Err(Error::Poisoned(_))));
    assert!(c.snapshot().poisoned);
    assert!(matches!(c.acquire_reader(), Err(Error::Poisoned(_))));
    assert!(matches!(c.address(expert(0), 0), Err(Error::Poisoned(_))));
    assert!(matches!(
        c.begin(plan(0, &[], &[]), 2),
        Err(Error::Poisoned(_))
    ));
    drop(c);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
}

#[test]
fn abandoned_reader_never_authorizes_replacement_or_free() {
    let (mut c, _, dropped) = setup(4);
    drop(c.acquire_reader().unwrap());
    c.begin(plan(0, &[0], &[]), 0).unwrap();
    assert_eq!(
        unsafe { c.progress(1) }.unwrap(),
        Progress::WaitingForReaders
    );
    drop(c);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    let (c, _, dropped) = setup(4);
    drop(c);
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
}

#[test]
fn execution_gate_stages_during_readers_then_blocks_submission_for_publication() {
    use execution::ExecutionBinding;
    let (controller, state, dropped) = setup(4);
    let binding = ExecutionBinding::new(controller);
    assert!(binding.submit(plan(0, &[0], &[])).is_err());
    let mut gate = unsafe { binding.claim() }.unwrap();
    assert!(unsafe { binding.claim() }.is_err());
    let reader = gate.reader().unwrap();
    state.lock().unwrap().ready = false;
    binding.submit(plan(0, &[0], &[])).unwrap();
    assert_eq!(binding.submit(plan(0, &[1], &[])), Err(Error::Busy));
    unsafe { gate.tick(1) }.unwrap();
    assert!(!gate.paused());
    assert_eq!(state.lock().unwrap().copies, 2);
    assert_eq!(binding.snapshot().unwrap().generation, 0);
    state.lock().unwrap().ready = true;
    unsafe { gate.tick(2) }.unwrap();
    assert!(gate.paused());
    assert!(matches!(gate.reader(), Err(Error::Busy)));
    assert!(unsafe { binding.close() }.is_err());
    unsafe { reader.complete(); gate.tick(3).unwrap(); }
    assert!(!gate.paused());
    assert_eq!(binding.snapshot().unwrap().generation, 1);
    let report = binding.report().unwrap();
    assert_eq!(report.events.len(),1);
    assert_eq!(report.events[0].outcome,"committed:1");
    assert_eq!(report.events[0].promote,vec![expert(0)]);
    assert_eq!(report.snapshot.statistics.promoted_experts,1);
    assert_eq!(report.snapshot.statistics.committed_copy_bytes,128);
    assert_eq!(state.lock().unwrap().mapping_calls, 2);
    unsafe { binding.close() }.unwrap();
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert!(binding.snapshot().is_err());
    assert!(gate.reader().is_err());
}

#[test]
fn execution_gate_rejects_stale_plans_and_retains_static_storage_after_copy_failure() {
    let (controller, state, _) = setup(4);
    let binding = execution::ExecutionBinding::new(controller);
    let mut gate = unsafe { binding.claim() }.unwrap();
    binding.submit(plan(1, &[0], &[])).unwrap();
    unsafe { gate.tick(1) }.unwrap();
    assert_eq!(binding.snapshot().unwrap().generation, 0);
    state.lock().unwrap().stage_error = Some(Error::Backend("copy failure".into()));
    binding.submit(plan(0, &[0], &[])).unwrap();
    unsafe { gate.tick(2) }.unwrap();
    let reader = gate.reader().unwrap();
    unsafe { reader.complete() };
    assert!(binding.submit(plan(0, &[1], &[])).is_err());
    assert_eq!(binding.snapshot().unwrap().generation, 0);
    unsafe { binding.close() }.unwrap();
}

#[test]
fn movement_timing_is_once_per_copy_and_excludes_initial_placement() {
    let (controller, state, _) = setup(4);
    let binding=execution::ExecutionBinding::new(controller);
    unsafe { binding.initialize(plan(0,&[0],&[])) }.unwrap();
    let mut gate=unsafe {binding.claim()}.unwrap();
    let reader=gate.reader().unwrap();
    state.lock().unwrap().ready=false;
    binding.submit(plan(1,&[1],&[0])).unwrap();
    unsafe {gate.tick(10)}.unwrap();
    assert_eq!(binding.snapshot().unwrap().statistics.copy_gpu_ns,123);
    state.lock().unwrap().ready=true;
    unsafe {gate.tick(20)}.unwrap();
    unsafe {gate.tick(30)}.unwrap();
    assert!(gate.paused());
    assert_eq!(binding.snapshot().unwrap().statistics.copy_gpu_ns,246);
    unsafe {reader.complete();gate.tick(40)}.unwrap();
    let report=binding.report().unwrap();
    assert_eq!(report.events.len(),1);
    assert_eq!(report.events[0].copy_gpu_ns,Some(123));
    assert_eq!(report.events[0].ready_wait_ns,20);
    assert_eq!(report.events[0].staged_bytes,128); // Two 64-byte weight parts.
    assert!(report.events[0].submission_gate_closed_ns>=20);
    assert_eq!(report.snapshot.statistics.timed_copies,2);
    unsafe {binding.close()}.unwrap();
}

#[test]
fn execution_gate_stops_on_uncertain_publication_and_initial_placement_is_strict() {
    let (controller, state, _) = setup(4);
    let binding = execution::ExecutionBinding::new(controller);
    assert_eq!(unsafe { binding.initialize(plan(1, &[0], &[])) }, Err(Error::Stale));
    unsafe { binding.initialize(plan(0, &[0], &[])) }.unwrap();
    let mut gate = unsafe { binding.claim() }.unwrap();
    assert_eq!(unsafe { binding.initialize(plan(1, &[1], &[])) }, Err(Error::Busy));
    let mut state = state.lock().unwrap();
    state.fail_mapping = Some(state.mapping_calls + 1);
    state.fail_rollback = true;
    drop(state);
    binding.submit(plan(1, &[1], &[0])).unwrap();
    assert!(matches!(unsafe { gate.tick(2) }, Err(Error::Poisoned(_))));
    assert!(gate.reader().is_err());
    assert!(unsafe { binding.close() }.is_err());
}

#[test]
fn movement_history_is_bounded_and_expiry_retains_the_original_mapping() {
    let (controller,state,_) = setup(4);
    let binding=execution::ExecutionBinding::new(controller);
    let mut gate=unsafe {binding.claim()}.unwrap();
    for _ in 0..260 {
        binding.submit(plan(7,&[0],&[])).unwrap();
        unsafe {gate.tick(0)}.unwrap();
    }
    let history=binding.report().unwrap();
    assert_eq!(history.event_count,260);
    assert_eq!(history.events.len(),256);
    assert_eq!(history.events[0].sequence,5);
    state.lock().unwrap().ready=false;
    binding.submit(plan(0,&[0],&[])).unwrap();
    unsafe {gate.tick(1)}.unwrap();
    state.lock().unwrap().ready=true;
    unsafe {gate.tick(101)}.unwrap();
    let history=binding.report().unwrap();
    assert_eq!(history.snapshot.statistics.cancellations,1);
    assert_eq!(history.snapshot.statistics.committed_copy_bytes,0);
    assert_eq!(history.snapshot.statistics.staged_bytes,128);
    assert_eq!(history.snapshot.generation,0);
    assert!(history.events.last().unwrap().outcome.contains("Expired"));
    unsafe {binding.close()}.unwrap();
}

#[test]
fn observed_target_cancels_a_copy_even_before_its_estimated_time_deadline() {
    let (controller,state,_) = setup(4);
    let binding=execution::ExecutionBinding::new(controller);
    let mut gate=unsafe {binding.claim()}.unwrap();
    binding.record_demand(1);
    assert_eq!(binding.submit_for_demand(plan(0,&[0],&[]),1),Err(Error::Expired));
    state.lock().unwrap().ready=false;
    binding.submit_for_demand(plan(0,&[0],&[]),3).unwrap();
    unsafe {gate.tick(1)}.unwrap();
    binding.record_demand(3);
    unsafe {gate.tick(2)}.unwrap();
    assert_eq!(binding.snapshot().unwrap().generation,0);
    state.lock().unwrap().ready=true;
    unsafe {gate.tick(3)}.unwrap();
    let report=binding.report().unwrap();
    assert_eq!(report.snapshot.statistics.cancellations,1);
    assert_eq!(report.snapshot.generation,0);
    assert_eq!(report.events[0].outcome,"Cancelled");
    assert_eq!(report.events[0].target_demand,Some(3));
    binding.submit_for_demand(plan(0,&[0],&[]),4).unwrap();
    binding.record_demand(4);
    unsafe {gate.tick(4)}.unwrap();
    assert_eq!(binding.snapshot().unwrap().statistics.staged_bytes,128);
    assert!(binding.report().unwrap().events[1].outcome.contains("Expired"));
    unsafe {binding.close()}.unwrap();
}
