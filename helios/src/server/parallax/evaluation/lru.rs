//! Reactive control: protected residents plus LRU, driven only by available observations.
use super::*;
use std::collections::VecDeque;

pub(super) const NAME: &str = "reactive/static_core_lru";

pub(super) struct Reactive {
    state: VirtualResidency,
    last_used: BTreeMap<Expert, u64>,
    protected: BTreeSet<Expert>,
    waiting: VecDeque<Observation>,
    cells: usize,
}

impl Reactive {
    pub(super) fn new(profile: &Profile) -> Self {
        Self {
            state: VirtualResidency::new(profile),
            last_used: BTreeMap::new(),
            protected: profile
                .settings
                .static_core_lru
                .as_ref()
                .unwrap()
                .protected_experts
                .iter()
                .copied()
                .collect(),
            waiting: VecDeque::new(),
            cells: 0,
        }
    }

    pub(super) fn observe(
        &mut self,
        observation: &Observation,
        profile: &Profile,
        limit: usize,
    ) -> Result<(), String> {
        // Routing can become available after a later frame has already begun.
        // A transfer may start at availability, never at the earlier demand time.
        while self
            .waiting
            .front()
            .is_some_and(|o| o.available_ns <= observation.demand_ns)
        {
            let eligible = self.waiting.pop_front().unwrap();
            self.cells -= eligible.demands.len().max(1);
            self.place(&eligible, profile);
        }
        self.state.demand(observation);
        if self.cells.saturating_add(observation.demands.len().max(1)) > limit {
            return Err("static-core LRU observation backlog exceeded its bounded capacity".into());
        }
        self.cells += observation.demands.len().max(1);
        self.waiting.push_back(observation.clone());
        Ok(())
    }

    fn place(&mut self, observation: &Observation, profile: &Profile) {
        self.state.advance(observation.available_ns);
        let config = profile.settings.static_core_lru.as_ref().unwrap();
        let touched: BTreeSet<_> = observation
            .demands
            .iter()
            .filter(|(_, count)| *count > 0)
            .map(|(expert, _)| *expert)
            .collect();
        for expert in &touched {
            self.last_used.insert(*expert, observation.seq);
        }
        if profile
            .hardware
            .transfer_calibration
            .as_ref()
            .is_some_and(|c| !c.usable())
        {
            return;
        }
        let mut candidates = observation.demands.clone();
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        // Demand and protection are fixed for this observation. Reuse the same
        // victim order instead of scanning every resident for each cold expert.
        let mut victims: Vec<_> = self
            .state
            .resident
            .iter()
            .filter(|e| !self.protected.contains(e) && !touched.contains(e))
            .copied()
            .collect();
        victims.sort_by_key(|e| (self.last_used.get(e).copied(), *e));
        let mut promoted = 0;
        for (expert, count) in candidates {
            if promoted == config.max_promotions_per_observation {
                break;
            }
            if count == 0
                || self.state.resident.contains(&expert)
                || self
                    .state
                    .pending
                    .iter()
                    .any(|(_, pending)| *pending == expert)
            {
                continue;
            }
            let size = self.state.sizes[&expert];
            let evict = if self.state.reserved + size <= profile.hardware.hbm_bytes {
                None
            } else {
                let Some(index) = victims.iter().position(|e| {
                    self.state.reserved - self.state.sizes[e] + size <= profile.hardware.hbm_bytes
                }) else {
                    continue;
                };
                Some(victims.remove(index))
            };
            self.state.apply(
                Movement {
                    promote: expert,
                    evict,
                },
                observation.available_ns,
                profile,
            );
            promoted += 1;
        }
    }

    pub(super) fn report(&self, profile: &Profile, baseline: &VirtualResidency) -> Value {
        let s = &self.state;
        let mut row = json!({
            "strategy_kind":"reactive_baseline", "prediction_records":0,
            "hits":s.hits, "misses":s.misses, "hbm_hit_rate":ratio(s.hits,s.hits+s.misses),
            "promotion_count":s.promotions, "eviction_count":s.evictions,
            "movement_bytes":s.bytes, "modeled_movement_cost_ns":s.cost,
            "resident_experts":s.resident.len(), "cold_experts":profile.model.experts.len()-s.resident.len(),
            "pending_transfers":s.pending.len(), "reserved_hbm_bytes":s.reserved,
            "protected_experts":profile.settings.static_core_lru.as_ref().unwrap().protected_experts.len(),
            "policy_scope":"Shadow only. LRU uses completed routing observations at their availability time. Transfers serialize and reserve HBM; one eviction per promotion. Policy overhead and serving contention are not calibrated."
        });
        apply_ledger(&mut row, s, baseline, 0, profile);
        if profile.settings.evaluation.is_none() {
            row["estimated_net_route_cost_saved_ns"] = json!(
                (baseline.misses as i128 - s.misses as i128) as f64
                    * profile.hardware.cold_cost_ns as f64
                    - s.cost as f64
            );
        }
        row
    }
}

pub(super) fn evaluate(capture: &Capture) -> Value {
    let mut reactive = Reactive::new(&capture.profile);
    let mut baseline = VirtualResidency::new(&capture.profile);
    for observation in &capture.observations {
        baseline.demand(observation);
        // Offline input is already bounded and validated by the capture reader.
        if let Err(error) = reactive.observe(observation, &capture.profile, usize::MAX) {
            return json!({"error":error});
        }
    }
    reactive.report(&capture.profile, &baseline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::parallax::{
        prediction::StaticCoreLruConfig,
        prediction_tests::{fixture, profile},
    };

    fn expert(id: usize) -> Expert {
        Expert {
            layer: 0,
            expert: id,
        }
    }
    fn configured() -> Profile {
        let mut p = profile("static-core-lru");
        p.model.experts = (0..5).map(expert).collect();
        p.model.expert_bytes = vec![100; 5];
        p.hardware.hbm_bytes = 300;
        p.hardware.backing_bytes = 500;
        p.settings.static_core_lru = Some(StaticCoreLruConfig {
            version: 1,
            protected_experts: vec![expert(0)],
            max_promotions_per_observation: 1,
        });
        p.validate().unwrap();
        p
    }
    fn observation(seq: u64, demand: u64, available: u64, ids: &[usize]) -> Observation {
        let mut o = fixture().observations[0].clone();
        o.seq = seq;
        o.demand_ns = demand;
        o.available_ns = available;
        o.demands = ids.iter().map(|id| (expert(*id), 1)).collect();
        o
    }

    #[test]
    fn delayed_observation_cannot_start_a_copy_early() {
        let p = configured();
        let mut r = Reactive::new(&p);
        r.observe(&observation(1, 10, 1000, &[3]), &p, 100).unwrap();
        r.observe(&observation(2, 100, 1100, &[3]), &p, 100)
            .unwrap();
        assert_eq!(r.state.promotions, 0);
        r.observe(&observation(3, 1001, 1200, &[3]), &p, 100)
            .unwrap();
        assert_eq!(r.state.misses, 3);
        assert_eq!(
            r.state.pending,
            vec![(1000 + transfer(&p, expert(3)), expert(3))]
        );
        r.observe(&observation(4, 2000, 2100, &[3]), &p, 100)
            .unwrap();
        assert_eq!(r.state.hits, 1);
        assert_eq!(r.state.promotions, 1);
    }

    #[test]
    fn protects_core_and_recent_hits_and_reserves_pending_bytes() {
        let p = configured();
        let mut r = Reactive::new(&p);
        r.place(&observation(0, 10, 20, &[1]), &p);
        r.place(&observation(2, 20, 30, &[3]), &p);
        assert!(r.state.resident.contains(&expert(0)));
        assert!(r.state.resident.contains(&expert(1)));
        assert!(!r.state.resident.contains(&expert(2)));
        assert_eq!(r.state.reserved, 300);
        r.place(&observation(3, 30, 40, &[3, 4]), &p);
        assert_eq!(r.state.pending.len(), 2);
        assert!(r.state.pending[1].0 > r.state.pending[0].0);
        assert_eq!(r.state.reserved, 300);
        assert!(r.state.resident.contains(&expert(0)));
    }

    #[test]
    fn validation_and_byte_budget_handle_heterogeneous_experts() {
        let mut p = configured();
        p.model.expert_bytes[3] = 250;
        p.hardware.backing_bytes = 650;
        p.validate().unwrap();
        let mut r = Reactive::new(&p);
        r.place(&observation(1, 10, 20, &[3]), &p);
        assert_eq!(r.state.promotions, 0);
        assert_eq!(r.state.evictions, 0);
        for ids in [
            vec![expert(9)],
            vec![expert(0), expert(0)],
            vec![expert(0), expert(3)],
        ] {
            p.settings
                .static_core_lru
                .as_mut()
                .unwrap()
                .protected_experts = ids;
            assert!(p.validate().is_err());
        }
    }

    #[test]
    fn batch_keeps_skipped_small_victims_and_uses_released_capacity() {
        let mut p = configured();
        p.model.experts = (0..7).map(expert).collect();
        p.model.expert_bytes = vec![100, 100, 200, 100, 150, 100, 50];
        p.hardware.hbm_bytes = 500;
        p.hardware.backing_bytes = 800;
        p.settings.version = 2;
        p.settings.initial_resident = Some((0..4).map(expert).collect());
        p.settings
            .static_core_lru
            .as_mut()
            .unwrap()
            .max_promotions_per_observation = 3;
        p.validate().unwrap();
        let mut r = Reactive::new(&p);
        r.place(&observation(0, 10, 20, &[4, 5, 6]), &p);
        // Expert 1 is too small for 4, but remains the oldest victim for 5.
        // Replacing 2 with 4 leaves enough space for 6 without another eviction.
        assert_eq!(r.state.resident, [expert(0), expert(3)].into());
        assert_eq!(
            r.state.pending.iter().map(|(_, e)| *e).collect::<Vec<_>>(),
            vec![expert(4), expert(5), expert(6)]
        );
        assert_eq!(r.state.reserved, 500);
        assert_eq!(r.state.promotions, 3);
        assert_eq!(r.state.evictions, 2);
    }

    #[test]
    fn touching_every_resident_leaves_no_eviction_candidate() {
        let p = configured();
        let mut r = Reactive::new(&p);
        r.place(&observation(0, 10, 20, &[0, 1, 2, 3, 4]), &p);
        assert_eq!(r.state.resident, [expert(0), expert(1), expert(2)].into());
        assert!(r.state.pending.is_empty());
        assert_eq!(r.state.promotions, 0);
        assert_eq!(r.state.evictions, 0);
    }

    #[test]
    fn explicit_initial_set_preserves_free_capacity_and_independent_states() {
        let mut p = configured();
        p.settings.version = 2;
        p.settings.initial_resident = Some(vec![expert(4), expert(0)]);
        p.validate().unwrap();
        let baseline = VirtualResidency::new(&p);
        let mut reactive = Reactive::new(&p);
        assert_eq!(baseline.resident, BTreeSet::from([expert(0), expert(4)]));
        assert_eq!(baseline.reserved, 200);
        assert_eq!(reactive.state.resident, baseline.resident);
        reactive.place(&observation(1, 10, 20, &[3]), &p);
        reactive.state.advance(1000);
        assert!(reactive.state.resident.contains(&expert(3)));
        assert_eq!(reactive.state.reserved, 300);
        assert_eq!(baseline.reserved, 200);
        assert!(!baseline.resident.contains(&expert(3)));
        assert_eq!(baseline.initial, reactive.state.initial);
        p.settings.static_core_lru = None;
        p.settings.initial_resident = Some(vec![]);
        p.validate().unwrap();
        let empty = VirtualResidency::new(&p);
        assert!(empty.resident.is_empty());
        assert_eq!(empty.reserved, 0);
    }

    #[test]
    fn calibration_core_seeds_every_contender_without_sharing_mutations() {
        let mut p = configured();
        p.settings
            .static_core_lru
            .as_mut()
            .unwrap()
            .protected_experts = vec![expert(4)];
        p.validate().unwrap();
        let baseline = VirtualResidency::new(&p);
        let mut reactive = Reactive::new(&p);
        assert_eq!(
            baseline.resident,
            BTreeSet::from([expert(0), expert(1), expert(4)])
        );
        assert_eq!(reactive.state.resident, baseline.resident);
        reactive.place(&observation(1, 10, 20, &[2, 3]), &p);
        assert_eq!(reactive.state.promotions, 1);
        assert!(reactive.state.resident.contains(&expert(4)));
        assert!(baseline.pending.is_empty());
        assert_eq!(
            baseline.resident,
            BTreeSet::from([expert(0), expert(1), expert(4)])
        );
        let initial = reactive.state.initial.clone();
        assert_eq!(initial, baseline.initial);
    }

    #[test]
    fn live_retention_and_offline_replay_preserve_independent_state() {
        let mut capture = fixture();
        capture.profile = configured();
        capture.profile.settings.max_frames = 1;
        capture.observations = vec![];
        for forecasts in capture.forecasts.values_mut() {
            forecasts.clear();
        }
        capture
            .workers
            .insert("ewma".into(), WorkerCounts::default());
        capture.forecasts.insert("ewma".into(), vec![]);
        let mut continuous = Continuous::new(&capture, 100);
        for o in [
            observation(0, 10, 20, &[3]),
            observation(1, 1000, 1010, &[3]),
            observation(2, 2000, 2010, &[3]),
        ] {
            continuous.observe(&o);
            capture.observations.push(o);
        }
        let (_, totals) = continuous.snapshot(&capture.workers);
        let offline = evaluate(&capture);
        assert_eq!(totals["residency"][NAME], offline);
        assert_eq!(offline["hits"], 2);
        assert_eq!(totals["residency"]["recent/static"]["hits"], 0);
        assert_eq!(totals["residency"]["ewma/economic"]["promotion_count"], 0);
        assert_eq!(continuous.window.observations.len(), 1);
        // Replay the complete trace through the public reader, including configuration validation.
        capture.profile.settings.max_frames = 64;
        let directory = &capture.profile.settings.output;
        std::fs::create_dir_all(directory).unwrap();
        let path = directory.join("capture.json");
        std::fs::write(&path, serde_json::to_vec(&capture).unwrap()).unwrap();
        let replay = crate::server::parallax::replay(&path).unwrap();
        assert_eq!(replay["contenders"][NAME], offline);
        std::fs::remove_dir_all(directory).unwrap();
        let mut bounded = Reactive::new(&capture.profile);
        bounded
            .observe(&observation(1, 10, 1000, &[3]), &capture.profile, 1)
            .unwrap();
        assert!(
            bounded
                .observe(&observation(2, 20, 1100, &[3]), &capture.profile, 1)
                .is_err()
        );
    }
}
