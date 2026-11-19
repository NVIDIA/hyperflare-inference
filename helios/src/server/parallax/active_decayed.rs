//! Reactive residency proposals with age and LRU updates tied to publication.
use super::{
    Record, decayed_miss,
    prediction::{DecayedMissLruConfig, Expert, Hardware, Profile},
};
use crate::parallax::residency::{
    Error, PlacementConstraint, Plan, Snapshot,
    execution::{ExecutionBinding, MovementEvent},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    DecayedMissLru,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub policy: Policy,
    pub interval_ns: u64,
    pub valid_for_ns: u64,
    pub publication_ns: u64,
}
impl Config {
    pub fn validate(&self, profile: &Profile) -> Result<(), String> {
        let algorithm = profile
            .settings
            .decayed_miss_lru
            .as_ref()
            .ok_or("active decayed-miss LRU requires decayed_miss_lru settings")?;
        if !(100_000_000..=60_000_000_000).contains(&self.interval_ns)
            || !(1..=5_000_000_000).contains(&self.valid_for_ns)
            || self.publication_ns == 0
            || self.publication_ns >= self.valid_for_ns
            || algorithm.max_promotions_per_observation > 64
        {
            return Err("active decayed-miss LRU requires 0.1..60 s decision spacing, a positive validity window up to 5 s, smaller positive publication cost and at most 64 promotions".into());
        }
        Ok(())
    }
}

struct Committed {
    policy: decayed_miss::State,
    installed: Vec<u64>,
    generation: u64,
    last_event: u64,
    commits: u64,
}
impl Committed {
    fn new(snapshot: &Snapshot, protected: &[Expert], last_event: u64) -> Self {
        let resident = snapshot
            .experts
            .iter()
            .filter_map(|(e, hot)| hot.then_some(*e))
            .collect();
        let policy = decayed_miss::State::new(&resident, protected);
        Self {
            installed: vec![snapshot.generation; policy.slots.len()],
            policy,
            generation: snapshot.generation,
            last_event,
            commits: 0,
        }
    }

    fn ingest(&mut self, events: &[MovementEvent]) -> Result<(), String> {
        for event in events {
            if event.sequence != self.last_event + 1 {
                return Err("active policy movement history has a gap".into());
            }
            if event.outcome.starts_with("committed:") {
                let next = self
                    .generation
                    .checked_add(1)
                    .ok_or("residency generation overflow")?;
                if event.generation != self.generation
                    || event.outcome != format!("committed:{next}")
                    || event.promote.len() != event.evict.len()
                {
                    return Err(
                        "active policy publication has inconsistent generation or exchanges".into(),
                    );
                }
                let mut indices = BTreeSet::new();
                let mut promotions = BTreeSet::new();
                let mut exchanges = vec![];
                for (&promote, victim) in event.promote.iter().zip(&event.evict) {
                    let index = *self
                        .policy
                        .slot_of
                        .get(victim)
                        .ok_or("publication evicted an unmanaged or protected expert")?;
                    if promote.layer != victim.layer
                        || self.policy.slot_of.contains_key(&promote)
                        || !indices.insert(index)
                        || !promotions.insert(promote)
                    {
                        return Err(
                            "publication violates the active policy's fixed layer pools".into()
                        );
                    }
                    exchanges.push((index, promote));
                }
                for (index, expert) in exchanges {
                    self.policy.replace(index, expert);
                    self.installed[index] = next;
                }
                self.generation = next;
                self.commits += 1;
            }
            // Rejection, cancellation and expiry cannot change policy residency.
            self.last_event = event.sequence;
        }
        Ok(())
    }

    fn observe(
        &mut self,
        record: &Record,
        executed: &BTreeSet<Expert>,
        config: &DecayedMissLruConfig,
    ) -> Result<Vec<(Expert, u64)>, String> {
        let generation = record
            .sample
            .residency_generation
            .ok_or("routing lacks residency generation")?;
        if generation > self.generation {
            return Err(
                "routing references a publication absent from active policy history".into(),
            );
        }
        let (mut hits, mut misses) = (vec![], vec![]);
        for layer in &record.sample.layers {
            if !self.policy.managed_layers.contains(&layer.layer) {
                continue;
            }
            for (expert, count) in layer
                .counts
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 0)
            {
                let expert = Expert {
                    layer: layer.layer,
                    expert,
                };
                if executed.contains(&expert) {
                    if let Some(&index) = self.policy.slot_of.get(&expert) {
                        if self.installed[index] <= generation {
                            hits.push((index, self.policy.slots[index].born));
                        }
                    }
                } else {
                    misses.push((expert, *count as u64));
                }
            }
        }
        self.policy.observe(&hits, &misses, config);
        Ok(misses)
    }
}

pub(super) struct Active {
    config: Config,
    algorithm: DecayedMissLruConfig,
    binding: ExecutionBinding,
    state: Committed,
    storage: BTreeMap<Expert, (u64, usize)>,
    constraints: BTreeMap<Expert, PlacementConstraint>,
    next_ns: u64,
    proposed: u64,
    no_movement: u64,
    stale: u64,
    failure: Option<String>,
}
impl Active {
    pub fn new(
        config: &Config,
        profile: &Profile,
        binding: ExecutionBinding,
    ) -> Result<Self, String> {
        config.validate(profile)?;
        let report = binding.report().map_err(|e| e.to_string())?;
        let snapshot = &report.snapshot;
        if snapshot.catalog_id != profile.model.identity
            || snapshot.pending
            || snapshot.poisoned
            || report.disabled.is_some()
        {
            return Err(
                "active decayed-miss LRU requires an available matching residency catalog".into(),
            );
        }
        let resident: Vec<_> = snapshot
            .experts
            .iter()
            .filter_map(|(e, hot)| hot.then_some(*e))
            .collect();
        let declared: BTreeSet<_> = profile
            .settings
            .initial_resident
            .as_ref()
            .ok_or("active decayed-miss LRU requires explicit initial residency")?
            .iter()
            .copied()
            .collect();
        if resident.iter().copied().collect::<BTreeSet<_>>() != declared {
            return Err(
                "active decayed-miss LRU initial residents differ from physical storage".into(),
            );
        }
        let algorithm = profile.settings.decayed_miss_lru.as_ref().unwrap().clone();
        let storage: BTreeMap<_, _> = binding
            .expert_storage()
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|(e, bytes, parts)| (e, (bytes, parts)))
            .collect();
        if storage.keys().copied().collect::<Vec<_>>() != profile.model.experts {
            return Err(
                "active decayed-miss LRU requires the complete physical expert catalog".into(),
            );
        }
        let constraints = binding.placement_constraints().map_err(|e| e.to_string())?;
        let protected: BTreeSet<_> = algorithm.protected_experts.iter().copied().collect();
        if resident
            .iter()
            .any(|e| !constraints[e].evictable && !protected.contains(e))
        {
            return Err("active decayed-miss LRU must protect every pinned resident expert".into());
        }
        let mut layer_layout = BTreeMap::new();
        for (e, layout) in &storage {
            let geometry = (*layout, constraints[e].group);
            if layer_layout
                .insert(e.layer, geometry)
                .is_some_and(|prior| prior != geometry)
            {
                return Err("active decayed-miss LRU requires one physical representation and allocation group per layer".into());
            }
        }
        Ok(Self {
            config: config.clone(),
            algorithm: algorithm.clone(),
            state: Committed::new(snapshot, &algorithm.protected_experts, report.event_count),
            binding,
            storage,
            constraints,
            next_ns: 0,
            proposed: 0,
            no_movement: 0,
            stale: 0,
            failure: None,
        })
    }

    pub fn disable(&mut self, reason: &str) {
        self.failure.get_or_insert(reason.into());
    }

    pub fn observe(
        &mut self,
        record: &Record,
        executed: &BTreeSet<Expert>,
        profile: &Profile,
        now: u64,
    ) {
        if self.failure.is_some() {
            return;
        }
        if let Err(error) = self.process(record, executed, profile, now) {
            self.disable(&error);
        }
    }

    fn process(
        &mut self,
        record: &Record,
        executed: &BTreeSet<Expert>,
        profile: &Profile,
        now: u64,
    ) -> Result<(), String> {
        let (events, _) = self
            .binding
            .events_after(self.state.last_event)
            .map_err(|e| e.to_string())?;
        self.state.ingest(&events)?;
        let misses = self.state.observe(record, executed, &self.algorithm)?;
        if now < self.next_ns {
            return Ok(());
        }
        let deadline = record
            .sample
            .available_ns
            .checked_add(self.config.valid_for_ns)
            .ok_or("active policy validity overflow")?;
        if now >= deadline {
            self.stale += 1;
            return Ok(());
        }
        let snapshot = self.binding.snapshot().map_err(|e| e.to_string())?;
        // A concurrent publication will be ingested before the following observation.
        if snapshot.generation != self.state.generation || snapshot.pending || snapshot.poisoned {
            return Ok(());
        }
        self.next_ns = now.saturating_add(self.config.interval_ns);
        let resident = snapshot
            .experts
            .iter()
            .filter_map(|(e, hot)| hot.then_some(*e))
            .collect();
        let candidates = self
            .state
            .policy
            .propose(&misses, &resident, &BTreeSet::new(), &self.algorithm)
            .into_iter()
            .map(|(expert, index)| (expert, self.state.policy.slots[index].expert))
            .collect::<Vec<_>>();
        let Some(plan) = admit(
            &snapshot,
            &self.storage,
            &self.constraints,
            &candidates,
            &profile.hardware,
            self.config.publication_ns,
            now,
            deadline,
        ) else {
            self.no_movement += 1;
            return Ok(());
        };
        match self.binding.submit(plan) {
            Ok(()) => self.proposed += 1,
            Err(Error::Busy) => {}
            Err(Error::Stale | Error::Expired) => self.stale += 1,
            Err(error) => return Err(error.to_string()),
        }
        Ok(())
    }

    pub fn report(&self) -> Value {
        json!({"policy":"decayed_miss_lru", "predictor":null, "failure":self.failure,
            "proposed_transactions":self.proposed, "accounted_commits":self.state.commits,
            "accounted_generation":self.state.generation, "policy_ticks":self.state.policy.tick,
            "no_movement_decisions":self.no_movement, "stale_observations":self.stale,
            "publication_cost_assumed_ns":self.config.publication_ns,
            "cost_scope":"Reactive placement with bounded validity, copy admission and commit-based replacement age; no estimated cold-execution gain."})
    }
}

/// Admit complete expert exchanges into currently spare storage before expiry.
fn admit(
    snapshot: &Snapshot,
    storage: &BTreeMap<Expert, (u64, usize)>,
    constraints: &BTreeMap<Expert, PlacementConstraint>,
    candidates: &[(Expert, Expert)],
    hardware: &Hardware,
    publication_ns: u64,
    now: u64,
    deadline: u64,
) -> Option<Plan> {
    if now >= deadline
        || snapshot.pending
        || snapshot.poisoned
        || hardware
            .transfer_calibration
            .as_ref()
            .is_none_or(|c| !c.usable())
    {
        return None;
    }
    let mut free = snapshot.free_slots_by_group.clone();
    let (mut promote, mut evict) = (vec![], vec![]);
    let mut cost = publication_ns as f64;
    for &(expert, victim) in candidates {
        let (bytes, parts) = storage[&expert];
        let group = constraints[&expert].group;
        if !constraints[&victim].evictable
            || constraints[&victim].group != group
            || storage[&victim] != (bytes, parts)
            || free.get(&group).copied().unwrap_or(0) < parts
        {
            continue;
        }
        let copy = bytes as f64 / hardware.bandwidth_bytes_per_ns
            + parts as f64 * hardware.transfer_latency_ns as f64;
        if !copy.is_finite() || cost + copy > (deadline - now) as f64 {
            continue;
        }
        cost += copy;
        *free.get_mut(&group).unwrap() -= parts;
        promote.push(expert);
        evict.push(victim);
    }
    if promote.is_empty() {
        return None;
    }
    Some(Plan {
        version: 1,
        catalog_id: snapshot.catalog_id.clone(),
        generation: snapshot.generation,
        valid_until_ns: deadline,
        promote,
        evict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallax::{LayerDemand, RoutingSample};

    fn e(expert: usize) -> Expert {
        Expert { layer: 0, expert }
    }
    fn configured() -> (Committed, DecayedMissLruConfig) {
        let snapshot = Snapshot {
            catalog_id: "synthetic-publication".into(),
            generation: 1,
            experts: (0..5).map(|i| (e(i), i < 3)).collect(),
            free_slots: 1,
            free_slots_by_group: BTreeMap::from([(0, 1)]),
            hbm_allocated_bytes: 400,
            backing_allocated_bytes: 500,
            readers: 0,
            pending: false,
            poisoned: false,
            statistics: Default::default(),
        };
        let config = DecayedMissLruConfig {
            version: 1,
            protected_experts: vec![e(0)],
            window: 2,
            min_misses: 1.,
            min_age_observations: 1,
            max_promotions_per_layer: 1,
            max_promotions_per_observation: 1,
        };
        (
            Committed::new(&snapshot, &config.protected_experts, 0),
            config,
        )
    }
    fn event(
        sequence: u64,
        generation: u64,
        promote: usize,
        evict: usize,
        outcome: &str,
    ) -> MovementEvent {
        MovementEvent {
            sequence,
            generation,
            started_ns: 10,
            finished_ns: 20,
            deadline_ns: 30,
            target_demand: None,
            promote: vec![e(promote)],
            evict: vec![e(evict)],
            outcome: outcome.into(),
            staged_bytes: 100,
            copy_gpu_ns: Some(5),
            publication_host_ns: 1,
            ready_wait_ns: 0,
            submission_gate_closed_ns: 0,
        }
    }
    fn record(sequence: u64, generation: u64, counts: Vec<u32>) -> Record {
        Record {
            sequence,
            graph: 0,
            operation: crate::execution::GraphKind::Decode,
            token_capacity: 1,
            phase: "decode",
            batch_size: 1,
            concurrency: 1,
            sample: RoutingSample {
                submission: sequence,
                residency_generation: Some(generation),
                gpu_ns: Some(10),
                demand_lower_bound_ns: sequence * 100,
                available_ns: sequence * 100 + 10,
                layers: vec![LayerDemand {
                    layer: 0,
                    tokens: counts.iter().sum(),
                    top_k: 1,
                    counts,
                }],
            },
        }
    }

    #[test]
    fn cancellations_preserve_age_and_delayed_routes_use_executed_residency() {
        let (mut state, config) = configured();
        let original = BTreeSet::from([e(0), e(1), e(2)]);
        state
            .observe(&record(1, 1, vec![0, 1, 0, 2, 0]), &original, &config)
            .unwrap();
        state.ingest(&[event(1, 1, 3, 2, "Cancelled")]).unwrap();
        assert_eq!(state.policy.slots[1].expert, e(2));
        assert_eq!(state.policy.slots[1].born, 0);
        assert_eq!(state.generation, 1);
        state.ingest(&[event(2, 1, 3, 2, "committed:2")]).unwrap();
        assert_eq!(state.policy.slots[1].born, 1);
        let misses = state
            .observe(&record(2, 1, vec![0, 0, 1, 4, 0]), &original, &config)
            .unwrap();
        assert_eq!(misses, vec![(e(3), 4)]); // Cold at execution, even though it is resident now.
        assert_eq!(state.policy.recent[&e(3)], 5.);
        assert_eq!(state.policy.slots[1].used, 1); // The earlier hit on expert 2 cannot refresh expert 3.
        state.ingest(&[event(3, 2, 2, 3, "committed:3")]).unwrap();
        state
            .observe(&record(3, 1, vec![0, 0, 1, 0, 0]), &original, &config)
            .unwrap();
        assert_eq!(state.policy.slots[1].used, 2); // Same expert identity, different incarnation.
        state
            .observe(&record(4, 3, vec![0, 0, 1, 0, 0]), &original, &config)
            .unwrap();
        assert_eq!(state.policy.slots[1].used, 4);
        assert_eq!(state.commits, 2);
    }

    #[test]
    fn missing_or_inconsistent_publications_are_rejected_before_replacement() {
        let (mut state, _) = configured();
        assert!(state.ingest(&[event(2, 1, 3, 2, "committed:2")]).is_err());
        assert!(state.ingest(&[event(1, 2, 3, 2, "committed:3")]).is_err());
        let mut invalid = event(1, 1, 3, 2, "committed:2");
        invalid.promote.push(e(4));
        invalid.evict.push(e(0)); // Protected experts are not managed cache slots.
        assert!(state.ingest(&[invalid]).is_err());
        assert_eq!(state.policy.slots[1].expert, e(2));
        assert_eq!(state.generation, 1);
        assert_eq!(state.last_event, 0);
    }

    #[test]
    fn copy_admission_charges_all_planes_and_preserves_spares_and_pinned_experts() {
        use crate::server::parallax::calibration::{Calibration, CopyCost};
        let mut snapshot = Snapshot {
            catalog_id: "synthetic-copy-budget".into(),
            generation: 8,
            experts: (0..5).map(|i| (e(i), i < 3)).collect(),
            free_slots: 4,
            free_slots_by_group: BTreeMap::from([(0, 4)]),
            hbm_allocated_bytes: 700,
            backing_allocated_bytes: 500,
            readers: 0,
            pending: false,
            poisoned: false,
            statistics: Default::default(),
        };
        let storage = (0..5).map(|i| (e(i), (100, 2))).collect();
        let mut constraints = (0..5)
            .map(|i| {
                (
                    e(i),
                    PlacementConstraint {
                        group: 0,
                        evictable: i != 0,
                    },
                )
            })
            .collect();
        let mut hardware = Hardware {
            identity: "synthetic".into(),
            hbm_bytes: 700,
            backing_bytes: 500,
            bandwidth_bytes_per_ns: 10.,
            transfer_latency_ns: 5,
            synchronization_ns: 0,
            cold_cost_ns: 0,
            transfer_calibration: Some(Calibration {
                version: 1,
                memory_path: "synthetic".into(),
                gpu_uuid: "synthetic".into(),
                elapsed_ns: 0,
                host_to_device: Some(CopyCost {
                    bandwidth_bytes_per_ns: 10.,
                    latency_ns: 5,
                }),
                device_to_host: Some(CopyCost {
                    bandwidth_bytes_per_ns: 10.,
                    latency_ns: 5,
                }),
                samples: vec![],
                failure: None,
            }),
        };
        let candidates = [(e(3), e(1)), (e(4), e(2))];
        // Each exchange costs 100/10 + 2*5 = 20 ns; publication costs 10 ns once.
        let plan = admit(
            &snapshot,
            &storage,
            &constraints,
            &candidates,
            &hardware,
            10,
            100,
            150,
        )
        .unwrap();
        assert_eq!(plan.promote, [e(3), e(4)]);
        assert_eq!(plan.evict, [e(1), e(2)]);
        assert_eq!((plan.generation, plan.valid_until_ns), (8, 150));
        let plan = admit(
            &snapshot,
            &storage,
            &constraints,
            &candidates,
            &hardware,
            10,
            100,
            149,
        )
        .unwrap();
        assert_eq!(plan.promote, [e(3)]);
        assert!(
            admit(
                &snapshot,
                &storage,
                &constraints,
                &candidates,
                &hardware,
                10,
                100,
                129
            )
            .is_none()
        );
        assert!(
            admit(
                &snapshot,
                &storage,
                &constraints,
                &candidates,
                &hardware,
                10,
                150,
                150
            )
            .is_none()
        );
        snapshot.free_slots_by_group.insert(0, 2);
        snapshot.free_slots_by_group.insert(1, 100); // Another layer's capacity is not available.
        assert_eq!(
            admit(
                &snapshot,
                &storage,
                &constraints,
                &candidates,
                &hardware,
                10,
                100,
                1000
            )
            .unwrap()
            .promote,
            [e(3)]
        );
        assert!(
            admit(
                &snapshot,
                &storage,
                &constraints,
                &[(e(3), e(0))],
                &hardware,
                10,
                100,
                1000
            )
            .is_none()
        );
        constraints.get_mut(&e(1)).unwrap().group = 1;
        assert!(
            admit(
                &snapshot,
                &storage,
                &constraints,
                &[(e(3), e(1))],
                &hardware,
                10,
                100,
                1000
            )
            .is_none()
        );
        snapshot.pending = true;
        assert!(
            admit(
                &snapshot,
                &storage,
                &constraints,
                &candidates,
                &hardware,
                10,
                100,
                1000
            )
            .is_none()
        );
        snapshot.pending = false;
        hardware.transfer_calibration.as_mut().unwrap().failure =
            Some("unusable calibration".into());
        assert!(
            admit(
                &snapshot,
                &storage,
                &constraints,
                &candidates,
                &hardware,
                10,
                100,
                1000
            )
            .is_none()
        );
        assert_eq!(snapshot.experts[&e(3)], false); // Proposals never publish mappings.
    }
}
