// Copyright (c) 2026 chronarion. Reference policy: MIT; see helios/licenses/DerpSeek.txt.
//! Decayed-miss ranking and LRU decisions, independent of transfer execution.
use super::prediction::{DecayedMissLruConfig, Expert};
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Slot {
    pub expert: Expert,
    pub born: u64,
    pub used: u64,
}

pub(super) struct State {
    pub slots: Vec<Slot>,
    pub slot_of: BTreeMap<Expert, usize>,
    pub managed_layers: BTreeSet<usize>,
    pub recent: BTreeMap<Expert, f32>,
    pub tick: u64,
}

impl State {
    pub fn new(resident: &BTreeSet<Expert>, protected: &[Expert]) -> Self {
        let protected: BTreeSet<_> = protected.iter().copied().collect();
        let slots: Vec<_> = resident
            .iter()
            .filter(|e| !protected.contains(e))
            .map(|e| Slot {
                expert: *e,
                born: 0,
                used: 0,
            })
            .collect();
        Self {
            managed_layers: slots.iter().map(|s| s.expert.layer).collect(),
            slot_of: slots
                .iter()
                .enumerate()
                .map(|(i, s)| (s.expert, i))
                .collect(),
            slots,
            recent: BTreeMap::new(),
            tick: 0,
        }
    }

    /// Hits identify the slot incarnation that served the routing observation.
    pub fn observe(
        &mut self,
        hits: &[(usize, u64)],
        misses: &[(Expert, u64)],
        config: &DecayedMissLruConfig,
    ) {
        self.tick += 1;
        for &(i, born) in hits {
            if self.slots[i].born == born {
                self.slots[i].used = self.tick;
            }
        }
        let decay = (1. - 1. / config.window as f64) as f32;
        for score in self.recent.values_mut() {
            *score *= decay;
        }
        for &(expert, count) in misses {
            *self.recent.entry(expert).or_default() += count as f32;
        }
    }

    /// Select proposals without changing residency, replacement age or LRU order.
    pub fn propose(
        &self,
        misses: &[(Expert, u64)],
        resident: &BTreeSet<Expert>,
        pending: &BTreeSet<Expert>,
        config: &DecayedMissLruConfig,
    ) -> Vec<(Expert, usize)> {
        let mut candidates: Vec<_> = misses
            .iter()
            .map(|(e, _)| *e)
            .filter(|e| {
                self.managed_layers.contains(&e.layer)
                    && !resident.contains(e)
                    && !pending.contains(e)
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
        let mut proposals = vec![];
        for (layer, picks) in layers {
            let mut victims: Vec<_> = self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| {
                    s.expert.layer == layer
                        && resident.contains(&s.expert)
                        && self.tick - s.born >= config.min_age_observations
                })
                .map(|(i, s)| (s.used, i))
                .collect();
            victims.sort_unstable();
            proposals.extend(
                picks
                    .into_iter()
                    .zip(victims)
                    .map(|(expert, (_, index))| (expert, index)),
            );
        }
        proposals
    }

    /// The caller confirms a simulated admission or a physical publication first.
    pub fn replace(&mut self, index: usize, expert: Expert) {
        self.slot_of.remove(&self.slots[index].expert);
        self.slot_of.insert(expert, index);
        self.slots[index] = Slot {
            expert,
            born: self.tick,
            used: self.tick,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposals_do_not_age_slots_and_delayed_hits_cannot_refresh_replacements() {
        let e = |expert| Expert { layer: 0, expert };
        let resident = BTreeSet::from([e(0), e(1), e(2)]);
        let config = DecayedMissLruConfig {
            version: 1,
            protected_experts: vec![e(0)],
            window: 2,
            min_misses: 1.,
            min_age_observations: 1,
            max_promotions_per_layer: 1,
            max_promotions_per_observation: 1,
        };
        let mut state = State::new(&resident, &config.protected_experts);
        state.observe(&[], &[(e(3), 2)], &config);
        assert_eq!(
            state.propose(&[(e(3), 2)], &resident, &BTreeSet::new(), &config),
            vec![(e(3), 0)]
        );
        // An expired proposal never calls replace; the same victim remains eligible.
        state.observe(&[], &[(e(3), 2)], &config);
        assert_eq!(
            state.propose(&[(e(3), 2)], &resident, &BTreeSet::new(), &config),
            vec![(e(3), 0)]
        );
        assert_eq!(state.slots[0].expert, e(1));
        assert_eq!(state.slots[0].born, 0);
        state.replace(0, e(3));
        state.observe(&[(0, 0)], &[], &config);
        assert_eq!(state.slots[0].used, 2);
        assert_eq!(state.slots[0].born, 2);
        assert_eq!(state.slot_of.get(&e(3)), Some(&0));
        assert!(!state.slot_of.contains_key(&e(1)));
    }
}
