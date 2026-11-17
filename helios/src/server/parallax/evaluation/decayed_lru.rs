// Copyright (c) 2026 chronarion. Reference policy: MIT; see helios/licenses/DerpSeek.txt.
//! Shadow policy inspired by chronarion/derpseek's residency.py decayed-miss LRU.
//! Each available routing frame is one policy tick; copy execution uses Flare's simulator.
use super::*;
use std::collections::VecDeque;

pub(super) const NAME: &str = "reactive/decayed_miss_lru";

struct Slot {
    expert: Expert,
    born: u64,
    used: u64,
}
struct Available {
    at: u64,
    misses: Vec<(Expert, u64)>,
    hits: Vec<(usize, u64)>,
    cells: usize,
}

pub(super) struct Reactive {
    state: VirtualResidency,
    slots: Vec<Slot>,
    slot_of: BTreeMap<Expert, usize>,
    managed_layers: BTreeSet<usize>,
    recent: BTreeMap<Expert, f32>,
    waiting: VecDeque<Available>,
    cells: usize,
    tick: u64,
}
impl Reactive {
    pub(super) fn new(profile: &Profile) -> Self {
        let state = VirtualResidency::new(profile);
        let protected: BTreeSet<_> = profile
            .settings
            .decayed_miss_lru
            .as_ref()
            .unwrap()
            .protected_experts
            .iter()
            .copied()
            .collect();
        let slots: Vec<_> = state
            .resident
            .iter()
            .filter(|e| !protected.contains(e))
            .map(|e| Slot {
                expert: *e,
                born: 0,
                used: 0,
            })
            .collect();
        let managed_layers = slots.iter().map(|s| s.expert.layer).collect();
        let slot_of = slots
            .iter()
            .enumerate()
            .map(|(i, s)| (s.expert, i))
            .collect();
        Self {
            state,
            slots,
            slot_of,
            managed_layers,
            recent: BTreeMap::new(),
            waiting: VecDeque::new(),
            cells: 0,
            tick: 0,
        }
    }

    pub(super) fn observe(
        &mut self,
        o: &Observation,
        p: &Profile,
        limit: usize,
    ) -> Result<(), String> {
        while self.waiting.front().is_some_and(|f| f.at <= o.demand_ns) {
            let frame = self.waiting.pop_front().unwrap();
            self.cells -= frame.cells;
            self.place(frame, p);
        }
        self.state.demand(o);
        let cells = o.demands.len().max(1);
        if self.cells.saturating_add(cells) > limit {
            return Err(
                "decayed-miss LRU observation backlog exceeded its bounded capacity".into(),
            );
        }
        // Freeze hit/miss classification at demand, before any later placement.
        // Delayed hits carry the slot generation so they cannot refresh a replacement.
        let mut frame = Available {
            at: o.available_ns,
            misses: vec![],
            hits: vec![],
            cells,
        };
        for (e, n) in &o.demands {
            if *n == 0 || !self.managed_layers.contains(&e.layer) {
                continue;
            }
            if self.state.resident.contains(e) {
                if let Some(i) = self.slot_of.get(e) {
                    frame.hits.push((*i, self.slots[*i].born));
                }
            } else {
                frame.misses.push((*e, *n));
            }
        }
        self.cells += cells;
        self.waiting.push_back(frame);
        Ok(())
    }

    fn place(&mut self, frame: Available, p: &Profile) {
        self.tick += 1;
        self.state.advance(frame.at);
        let config = p.settings.decayed_miss_lru.as_ref().unwrap();
        for (i, born) in frame.hits {
            if self.slots[i].born == born {
                self.slots[i].used = self.tick;
            }
        }
        let decay = (1. - 1. / config.window as f64) as f32;
        for score in self.recent.values_mut() {
            *score *= decay;
        }
        for (e, n) in &frame.misses {
            *self.recent.entry(*e).or_default() += *n as f32;
        }
        if p.hardware
            .transfer_calibration
            .as_ref()
            .is_some_and(|c| !c.usable())
        {
            return;
        }
        let mut candidates: Vec<_> = frame
            .misses
            .iter()
            .map(|(e, _)| *e)
            .filter(|e| {
                self.managed_layers.contains(&e.layer)
                    && !self.state.resident.contains(e)
                    && !self.state.pending.iter().any(|(_, pending)| pending == e)
                    && self.recent[e] >= config.min_misses
            })
            .collect();
        candidates.sort_by(|a, b| self.recent[b].total_cmp(&self.recent[a]).then(a.cmp(b)));
        candidates.dedup();
        // Quotas precede victim admission, matching the reference's policy order.
        // A busy layer does not donate its selected quota to a lower-ranked layer.
        let mut layers: Vec<(usize, Vec<Expert>)> = vec![];
        let mut selected = 0;
        for e in candidates {
            let index = layers
                .iter()
                .position(|(layer, _)| *layer == e.layer)
                .unwrap_or_else(|| {
                    layers.push((e.layer, vec![]));
                    layers.len() - 1
                });
            let picks = &mut layers[index].1;
            if picks.len() == config.max_promotions_per_layer {
                continue;
            }
            picks.push(e);
            selected += 1;
            if selected == config.max_promotions_per_observation {
                break;
            }
        }
        for (layer, picks) in layers {
            let mut victims: Vec<_> = self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| {
                    s.expert.layer == layer
                        && self.state.resident.contains(&s.expert)
                        && self.tick - s.born >= config.min_age_observations
                })
                .map(|(i, s)| (s.used, i))
                .collect();
            victims.sort_unstable();
            for (e, (_, i)) in picks.into_iter().zip(victims) {
                let victim = self.slots[i].expert;
                let before = self.state.promotions;
                self.state.apply(
                    Movement {
                        promote: e,
                        evict: Some(victim),
                    },
                    frame.at,
                    p,
                );
                if self.state.promotions == before {
                    continue;
                }
                self.slot_of.remove(&victim);
                self.slot_of.insert(e, i);
                self.slots[i] = Slot {
                    expert: e,
                    born: self.tick,
                    used: self.tick,
                };
            }
        }
    }

    pub(super) fn report(&self, p: &Profile, baseline: &VirtualResidency) -> Value {
        let s = &self.state;
        let mut row = json!({
            "strategy_kind":"reactive_baseline", "prediction_records":0,
            "hits":s.hits, "misses":s.misses, "hbm_hit_rate":ratio(s.hits,s.hits+s.misses),
            "promotion_count":s.promotions, "eviction_count":s.evictions,
            "movement_bytes":s.bytes, "modeled_movement_cost_ns":s.cost,
            "resident_experts":s.resident.len(), "cold_experts":p.model.experts.len()-s.resident.len(),
            "pending_transfers":s.pending.len(), "reserved_hbm_bytes":s.reserved,
            "protected_experts":p.settings.decayed_miss_lru.as_ref().unwrap().protected_experts.len(),
            "policy_ticks":self.tick, "cache_slots":self.slots.len(),
            "policy_scope":"Shadow only. Fixed per-layer pools, decayed misses and LRU; one tick per available frame. Transfers serialize and evict before copy. No physical execution, policy overhead or serving contention calibration; not a DerpSeek throughput reproduction."
        });
        apply_ledger(&mut row, s, baseline, 0, p);
        if p.settings.evaluation.is_none() {
            row["estimated_net_route_cost_saved_ns"] = json!(
                (baseline.misses as i128 - s.misses as i128) as f64
                    * p.hardware.cold_cost_ns as f64
                    - s.cost as f64
            );
        }
        row
    }
}

pub(super) fn evaluate(c: &Capture) -> Value {
    let mut reactive = Reactive::new(&c.profile);
    let mut baseline = VirtualResidency::new(&c.profile);
    for o in &c.observations {
        baseline.demand(o);
        if let Err(error) = reactive.observe(o, &c.profile, usize::MAX) {
            return json!({"error":error});
        }
    }
    reactive.report(&c.profile, &baseline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::parallax::{
        prediction::{DecayedMissLruConfig, StaticCoreLruConfig},
        prediction_tests::{fixture, profile},
    };

    fn e(layer: usize, expert: usize) -> Expert {
        Expert { layer, expert }
    }
    fn configured() -> Profile {
        let mut p = profile("decayed-miss-lru");
        p.model.experts = (0..2).flat_map(|l| (0..6).map(move |i| e(l, i))).collect();
        p.model.expert_bytes = vec![100; 12];
        p.hardware.hbm_bytes = 600;
        p.hardware.backing_bytes = 1200;
        p.settings.version = 2;
        p.settings.initial_resident =
            Some(vec![e(0, 0), e(0, 1), e(0, 2), e(1, 0), e(1, 1), e(1, 2)]);
        p.settings.decayed_miss_lru = Some(DecayedMissLruConfig {
            version: 1,
            protected_experts: vec![e(0, 0), e(1, 0)],
            window: 2,
            min_misses: 1.,
            min_age_observations: 0,
            max_promotions_per_layer: 1,
            max_promotions_per_observation: 2,
        });
        p.validate().unwrap();
        p
    }
    fn observation(
        seq: u64,
        demand_ns: u64,
        available_ns: u64,
        demands: &[(Expert, u64)],
    ) -> Observation {
        Observation {
            seq,
            demand_ns,
            available_ns,
            phase: "decode".into(),
            batch_size: 1,
            concurrency: 1,
            demands: demands.to_vec(),
        }
    }
    fn available(at: u64, misses: &[(Expert, u64)]) -> Available {
        Available {
            at,
            misses: misses.to_vec(),
            hits: vec![],
            cells: misses.len().max(1),
        }
    }

    #[test]
    fn decay_ranks_misses_with_fixed_layer_capacity_and_static_protection() {
        let mut p = configured();
        let c = p.settings.decayed_miss_lru.as_mut().unwrap();
        c.min_age_observations = 2;
        let mut r = Reactive::new(&p);
        r.place(available(100, &[(e(0, 3), 10)]), &p);
        assert_eq!(r.state.promotions, 0); // All initial slots are too young.
        r.place(
            available(200, &[(e(0, 3), 1), (e(0, 4), 5), (e(1, 3), 3)]),
            &p,
        );
        assert_eq!(r.recent[&e(0, 3)], 6.);
        assert_eq!(
            r.slots.iter().map(|s| s.expert).collect::<Vec<_>>(),
            vec![e(0, 3), e(0, 2), e(1, 3), e(1, 2)]
        );
        assert!(r.state.resident.contains(&e(0, 0)));
        assert!(r.state.resident.contains(&e(1, 0)));
        assert_eq!(r.state.reserved, 600);
        assert_eq!(r.state.bytes, 200);
        assert_eq!(r.state.pending, vec![(315, e(0, 3)), (430, e(1, 3))]);
        // A high historical score without a current miss cannot cause promotion.
        r.place(available(500, &[(e(0, 5), 1)]), &p);
        assert_eq!(r.slots[1].expert, e(0, 5));
    }

    #[test]
    fn busy_slots_and_quotas_match_reference_order() {
        let mut p = configured();
        p.settings
            .decayed_miss_lru
            .as_mut()
            .unwrap()
            .max_promotions_per_observation = 1;
        p.settings
            .decayed_miss_lru
            .as_mut()
            .unwrap()
            .protected_experts
            .extend([e(0, 2), e(1, 2)]);
        p.validate().unwrap();
        let mut r = Reactive::new(&p);
        r.place(available(10, &[(e(0, 3), 5)]), &p);
        r.place(available(20, &[(e(0, 4), 20), (e(1, 3), 10)]), &p);
        // The top candidate consumes the quota even though its layer is busy.
        assert_eq!(r.state.promotions, 1);
        assert_eq!(r.slots[1].expert, e(1, 1));
        r.place(available(30, &[(e(0, 3), 100), (e(1, 3), 10)]), &p);
        // Already-held candidates do not consume the quota.
        assert_eq!(r.state.promotions, 2);
        assert_eq!(r.slots[1].expert, e(1, 3));
    }

    #[test]
    fn delayed_observations_keep_demand_time_misses_and_slot_generations() {
        let p = configured();
        let mut r = Reactive::new(&p);
        r.observe(&observation(0, 10, 100, &[(e(0, 3), 2)]), &p, 20)
            .unwrap();
        r.observe(
            &observation(1, 20, 500, &[(e(0, 1), 1), (e(0, 3), 4)]),
            &p,
            20,
        )
        .unwrap();
        assert_eq!(r.state.promotions, 0);
        r.observe(&observation(2, 300, 600, &[(e(0, 3), 1)]), &p, 20)
            .unwrap();
        assert_eq!(r.state.hits, 2); // Initial expert 1, then completed promotion 3.
        assert_eq!(r.state.misses, 6);
        let birth = r.slots[0].born;
        r.observe(&observation(3, 550, 700, &[]), &p, 20).unwrap();
        assert_eq!(r.recent[&e(0, 3)], 5.); // Includes the earlier cold demand of four.
        assert_eq!(r.slots[0].used, birth); // Old hit on evicted expert 1 cannot refresh 3.
        assert_eq!(r.state.promotions, 1); // Expert 3 is now held, not copied twice.
        r.observe(&observation(4, 650, 800, &[]), &p, 20).unwrap();
        assert_eq!(r.slots[0].used, 3); // The hit on expert 3 becomes available.
        assert_eq!(r.recent[&e(0, 3)], 2.5);
    }

    #[test]
    fn layers_without_cache_slots_cannot_consume_promotion_quota() {
        let mut p = configured();
        p.settings
            .decayed_miss_lru
            .as_mut()
            .unwrap()
            .protected_experts = vec![e(1, 0)];
        p.settings
            .decayed_miss_lru
            .as_mut()
            .unwrap()
            .max_promotions_per_observation = 1;
        p.settings.initial_resident = Some(vec![e(1, 0), e(1, 1), e(1, 2)]);
        p.validate().unwrap();
        let mut r = Reactive::new(&p);
        r.observe(
            &observation(0, 10, 20, &[(e(0, 3), 100), (e(1, 3), 1)]),
            &p,
            20,
        )
        .unwrap();
        r.observe(&observation(1, 30, 40, &[]), &p, 20).unwrap();
        assert_eq!(r.state.pending, vec![(135, e(1, 3))]);
        assert_eq!(r.state.reserved, 300); // Unassigned global capacity remains unused.
        assert_eq!(r.state.misses, 101); // All cold demand remains in the ledger.
        assert!(!r.recent.contains_key(&e(0, 3)));
    }

    #[test]
    fn hits_refresh_lru_and_ties_use_stable_slot_order() {
        let p = configured();
        let mut r = Reactive::new(&p);
        r.observe(&observation(0, 10, 20, &[(e(0, 1), 1)]), &p, 20)
            .unwrap();
        r.observe(
            &observation(1, 30, 40, &[(e(0, 3), 1), (e(0, 4), 1)]),
            &p,
            20,
        )
        .unwrap();
        r.observe(&observation(2, 50, 60, &[]), &p, 20).unwrap();
        assert_eq!(r.slots[0].expert, e(0, 1));
        assert_eq!(r.slots[1].expert, e(0, 3));
    }

    #[test]
    fn synthetic_decisions_match_derpseek_reference() {
        let fixtures: Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/parallax_decayed_lru.json"
        ))
        .unwrap();
        for (case, f) in fixtures["cases"].as_array().unwrap().iter().enumerate() {
            let mut p = configured();
            let config = p.settings.decayed_miss_lru.as_mut().unwrap();
            config.window = f["window"].as_u64().unwrap() as u32;
            config.min_misses = f["minimum"].as_f64().unwrap() as f32;
            config.min_age_observations = f["age"].as_u64().unwrap();
            config.max_promotions_per_layer = f["per_layer"].as_u64().unwrap() as usize;
            config.max_promotions_per_observation = f["total"].as_u64().unwrap() as usize;
            let mut r = Reactive::new(&p);
            r.tick = 4;
            let mut misses = vec![];
            for layer in 0..2 {
                for j in 0..5 {
                    let expert = e(layer, j + 1);
                    r.recent
                        .insert(expert, f["previous"][layer][j].as_f64().unwrap() as f32);
                    let count = f["misses"][layer][j].as_u64().unwrap();
                    if count > 0 {
                        misses.push((expert, count));
                    }
                }
                for slot in 0..2 {
                    let s = &mut r.slots[2 * layer + slot];
                    s.born = f["born"][layer][slot].as_u64().unwrap();
                    s.used = f["used"][layer][slot].as_u64().unwrap();
                    if f["busy"][layer][slot].as_bool().unwrap() {
                        r.state.resident.remove(&s.expert);
                        r.state.pending.push((1_000_000, s.expert));
                        r.state.copy_until = 1_000_000;
                    }
                }
            }
            let pending = r.state.pending.len();
            r.place(available(100, &misses), &p);
            let actual: Vec<_> = r.state.pending[pending..]
                .iter()
                .map(|(_, expert)| {
                    let slot = r.slot_of[expert] % 2;
                    json!([expert.layer, expert.expert, slot])
                })
                .collect();
            assert_eq!(json!(actual), f["pairs"], "reference case {case}");
        }
    }

    #[test]
    fn validation_rejects_ambiguous_pools_and_backlog_is_bounded() {
        let p = configured();
        let mut invalid = p.clone();
        invalid.model.expert_bytes[1] = 99;
        assert!(
            invalid
                .validate()
                .unwrap_err()
                .contains("equal expert sizes")
        );
        invalid = p.clone();
        invalid.settings.version = 1;
        invalid.settings.initial_resident = None;
        assert!(
            invalid
                .validate()
                .unwrap_err()
                .contains("explicit initial_resident")
        );
        for protected in [vec![e(0, 5)], vec![e(0, 0), e(0, 0)]] {
            invalid = p.clone();
            invalid
                .settings
                .decayed_miss_lru
                .as_mut()
                .unwrap()
                .protected_experts = protected;
            assert!(invalid.validate().is_err());
        }
        invalid = p.clone();
        invalid.settings.decayed_miss_lru.as_mut().unwrap().window = 0;
        assert!(invalid.validate().is_err());
        let mut r = Reactive::new(&p);
        r.observe(&observation(0, 10, 100, &[(e(0, 3), 1)]), &p, 1)
            .unwrap();
        assert!(
            r.observe(&observation(1, 20, 200, &[(e(0, 4), 1)]), &p, 1)
                .is_err()
        );
        assert_eq!(r.waiting.len(), 1);
    }

    #[test]
    fn continuous_and_public_replay_agree_with_independent_contenders() {
        let mut c = fixture();
        c.profile = configured();
        c.profile.settings.static_core_lru = Some(StaticCoreLruConfig {
            version: 1,
            protected_experts: vec![e(0, 0), e(1, 0)],
            max_promotions_per_observation: 2,
        });
        c.observations.clear();
        for forecasts in c.forecasts.values_mut() {
            forecasts.clear();
        }
        c.workers.insert("ewma".into(), WorkerCounts::default());
        c.forecasts.insert("ewma".into(), vec![]);
        c.profile.validate().unwrap();
        let mut live = Continuous::new(&c, 100);
        live.window.profile.settings.max_frames = 1;
        for o in [
            observation(0, 10, 20, &[(e(0, 3), 2)]),
            observation(1, 1000, 1010, &[(e(0, 3), 2)]),
            observation(2, 2000, 2010, &[(e(0, 3), 2)]),
        ] {
            live.observe(&o);
            c.observations.push(o);
        }
        let (window, totals) = live.snapshot(&c.workers);
        let offline = super::super::evaluate(&c);
        assert_eq!(totals["residency"][NAME], offline["contenders"][NAME]);
        assert_eq!(totals["residency"][NAME]["hits"], 4);
        assert_eq!(totals["residency"]["recent/static"]["hits"], 0);
        assert_eq!(totals["residency"]["ewma/economic"]["promotion_count"], 0);
        assert_eq!(window.observations.len(), 1);
        let report = evaluate_continuous(&window, totals);
        assert_eq!(report["contenders"][NAME], offline["contenders"][NAME]);
        let path = c.profile.settings.output.join("capture.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_vec(&c).unwrap()).unwrap();
        let replay = crate::server::parallax::replay(&path).unwrap();
        assert_eq!(replay["contenders"][NAME], offline["contenders"][NAME]);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
