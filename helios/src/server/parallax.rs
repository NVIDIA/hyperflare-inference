//! Optional routing capture and an in-process shadow observatory.

pub mod calibration;
pub mod active;
mod active_decayed;
mod evaluation;
mod ledger;
mod decayed_miss;
mod canary;
mod observer;
pub mod prediction;
pub use observer::{replay, reevaluate};
pub use prediction::{Profile, Registry};

use crate::parallax::{RoutingReadout, RoutingSample};
use base64::Engine as _;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

const CAPACITY: usize = 32;

#[derive(Serialize)]
struct Record {
    sequence: u64,
    graph: usize,
    operation: crate::execution::GraphKind,
    token_capacity: usize,
    phase: &'static str,
    batch_size: usize,
    concurrency: usize,
    sample: RoutingSample,
}

pub(super) struct Telemetry {
    geometry: Option<(usize, usize)>,
    residency: Option<crate::parallax::residency::execution::ExecutionBinding>,
    session: String,
    sequence: AtomicU64,
    dropped: AtomicU64,
    invalid: AtomicU64,
    history: Mutex<VecDeque<Arc<Record>>>,
    observer: OnceLock<observer::Observer>,
    protected_bitset: OnceLock<Option<String>>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            geometry: None,
            residency: None,
            session: format!(
                "{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            sequence: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            invalid: AtomicU64::new(0),
            history: Mutex::new(VecDeque::with_capacity(CAPACITY)),
            observer: OnceLock::new(),
            protected_bitset: OnceLock::new(),
        }
    }
}

impl Telemetry {
    pub fn configure(&mut self, manifest: &crate::execution::ExecutionManifest) {
        self.residency = manifest.residency.clone();
        self.geometry = manifest
            .buffers
            .iter()
            .find(|b| b.name == crate::parallax::BUFFER)
            .map(|b| (b.shape[1], b.shape[2] - crate::parallax::HEADER));
    }

    pub fn publish(
        &self,
        readout: RoutingReadout,
        graph: usize,
        kind: crate::execution::GraphKind,
        token_capacity: usize,
        batch_size: usize,
        concurrency: usize,
    ) {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(binding) = &self.residency { binding.record_demand(sequence); }
        let RoutingReadout::Valid(sample) = readout else {
            self.invalid.fetch_add(1, Ordering::Relaxed);
            if let Some(observer) = self.observer.get() {
                observer.invalidate();
            }
            return;
        };
        let record = Arc::new(Record {
            sequence,
            graph,
            operation: kind,
            token_capacity,
            phase: if kind.is_prefill() {
                "prefill"
            } else if kind == crate::execution::GraphKind::Speculative {
                "verify"
            } else {
                "decode"
            },
            batch_size,
            concurrency,
            sample,
        });
        if let Some(observer) = self.observer.get() {
            observer.publish(record.clone());
        }
        let Ok(mut history) = self.history.try_lock() else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if history.len() == CAPACITY {
            history.pop_front();
        }
        history.push_back(record);
    }

    pub fn start_predictions(&self, profile: Profile) -> Result<(), String> {
        let (layers, experts) = self
            .geometry
            .ok_or("Parallax requires explicitly enabled routing telemetry")?;
        if profile
            .model
            .experts
            .iter()
            .any(|e| e.layer >= layers || e.expert >= experts)
        {
            return Err("catalog exceeds native routing geometry".into());
        }
        let protected = profile_protected_bitset(&profile, layers, experts);
        let observer = observer::Observer::start_with_residency(profile, Registry::builtins(), self.residency.clone())?;
        let _ = self.protected_bitset.set(protected);
        self.observer
            .set(observer)
            .map_err(|_| "prediction providers already started".into())
    }
    pub fn is_active(&self) -> bool {
        self.observer.get().is_some() || self.residency.is_some()
    }
    pub fn predictions(&self) -> Value {
        let mut report = self.observer.get().map(|o| o.snapshot()).unwrap_or_else(
            || json!({"enabled":false,"contenders":{},"routing_frames":0,"workers":{}}),
        );
        if let Some(binding) = &self.residency {
            let protected = self.protected_bitset.get().and_then(|v| v.as_deref());
            report["physical_residency"] = physical_report(binding, report["active_policy"].is_object(), protected);
        }
        report
    }
    pub fn close_predictions(&self) {
        if let Some(observer) = self.observer.get() {
            observer.close();
        }
    }

    pub fn snapshot(&self, after: u64, model: &str) -> Value {
        // Clone only bounded Arc handles under the lock; JSON encoding happens outside it.
        let records: Vec<_> = self
            .history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|row| row.sequence > after)
            .cloned()
            .collect();
        let rows: Vec<_> = records
            .iter()
            .map(|r| serde_json::to_value(r.as_ref()).expect("routing values serialize"))
            .collect();
        json!({"version":1,"session":self.session,"model_id":model,
            "enabled":self.geometry.is_some(),"geometry":self.geometry,
            "clock":"native_process_monotonic_ns","native_now_ns":crate::parallax::now_ns(),
            "demand_timing":"host_dispatch_lower_bound",
            "demand_scope":"target_expert_execution_including_rejected_candidates; draft experts excluded",
            "capacity":CAPACITY,"latest_sequence":self.sequence.load(Ordering::Relaxed),
            "dropped_total":self.dropped.load(Ordering::Relaxed),
            "invalid_total":self.invalid.load(Ordering::Relaxed),"frames":rows})
    }
}

pub(super) fn encode_expert_bitset(
    layers: usize,
    experts_per_layer: usize,
    active: impl IntoIterator<Item = (usize, usize)>,
) -> String {
    let total = layers.saturating_mul(experts_per_layer);
    let mut bytes = vec![0u8; total.div_ceil(8)];
    for (layer, expert) in active {
        if layer < layers && expert < experts_per_layer {
            let idx = layer * experts_per_layer + expert;
            bytes[idx / 8] |= 1u8 << (idx % 8);
        }
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub(super) fn profile_protected_bitset(
    profile: &Profile,
    layers: usize,
    experts_per_layer: usize,
) -> Option<String> {
    let protected = profile
        .settings
        .decayed_miss_lru
        .as_ref()
        .map(|c| &c.protected_experts)
        .or_else(|| {
            profile
                .settings
                .static_core_lru
                .as_ref()
                .map(|c| &c.protected_experts)
        })?;
    if protected.is_empty() || layers == 0 || experts_per_layer == 0 {
        return None;
    }
    Some(encode_expert_bitset(
        layers,
        experts_per_layer,
        protected.iter().map(|e| (e.layer, e.expert)),
    ))
}

fn physical_report(
    binding: &crate::parallax::residency::execution::ExecutionBinding,
    active: bool,
    protected_bitset: Option<&str>,
) -> Value {
    match binding.report() {
        Ok(report) => {
            let state=report.snapshot;
            let hbm=state.experts.values().filter(|resident| **resident).count();
            let (layers, experts_per_layer) = state.experts.keys().fold((0usize, 0usize), |(l, e), id| {
                (l.max(id.layer + 1), e.max(id.expert + 1))
            });
            let placement = if layers > 0 && experts_per_layer > 0 {
                let hbm_bitset = encode_expert_bitset(
                    layers,
                    experts_per_layer,
                    state.experts.iter().filter_map(|(id, resident)| (*resident).then_some((id.layer, id.expert))),
                );
                json!({
                    "layers": layers,
                    "experts_per_layer": experts_per_layer,
                    "hbm_bitset": hbm_bitset,
                    "protected_bitset": protected_bitset,
                })
            } else {
                Value::Null
            };
            json!({"mode":if active {"active_policy"} else {"configured_initial_placement"},"catalog_id":state.catalog_id,
                "generation":state.generation,"hbm_experts":hbm,"grace_experts":state.experts.len()-hbm,
                "hbm_pool_bytes":state.hbm_allocated_bytes,"grace_backing_bytes":state.backing_allocated_bytes,
                "free_slots":state.free_slots,"pending":state.pending,"readers":state.readers,"poisoned":state.poisoned,
                "commits_including_startup":state.statistics.commits,
                "staged_copy_bytes_including_startup":state.statistics.staged_bytes,
                "committed_copy_bytes_including_startup":state.statistics.committed_copy_bytes,
                "promoted_experts_including_startup":state.statistics.promoted_experts,
                "evicted_experts_including_startup":state.statistics.evicted_experts,
                "publication_host_ns_including_startup":state.statistics.publication_host_ns,
                "ready_wait_ns_including_startup":state.statistics.ready_wait_ns,
                "copy_gpu_ns_including_startup":state.statistics.copy_gpu_ns,
                "timed_copy_transactions_including_startup":state.statistics.timed_copies,
                "rollbacks":state.statistics.rollbacks,"rejected_plans":state.statistics.rejected,
                "cancelled_transactions":state.statistics.cancellations,"movement_error":report.disabled,
                "movement_history_capacity":256,"movement_event_total":report.event_count,"movement_events":report.events,
                "placement":placement})
        },
        Err(error) => json!({"error":error.to_string()}),
    }
}

#[cfg(test)]
#[path = "parallax/tests.rs"]
mod prediction_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::GraphKind;
    fn sample() -> RoutingReadout {
        RoutingReadout::Valid(RoutingSample {
            residency_generation: None,
            gpu_ns: None,
            submission: 1,
            demand_lower_bound_ns: 10,
            available_ns: 20,
            layers: Vec::new(),
        })
    }

    #[test]
    fn submitted_prefill_operations_remain_distinct_in_routing_records() {
        let telemetry = Telemetry::default();
        for kind in [GraphKind::Prefill, GraphKind::TerminalPrefill, GraphKind::ScoredPrefill] {
            telemetry.publish(sample(), 0, kind, 8192, 1, 1);
        }
        let value = telemetry.snapshot(0, "fixture");
        let frames = value["frames"].as_array().unwrap();
        for (frame, expected) in frames.iter().zip(["prefill", "terminal_prefill", "scored_prefill"]) {
            assert_eq!(frame["operation"], expected);
            assert_eq!(frame["phase"], "prefill");
            assert_eq!(frame["token_capacity"], 8192);
        }
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn retention_and_cursor_are_bounded_and_loss_is_visible() {
        let telemetry = Telemetry::default();
        for _ in 0..40 {
            telemetry.publish(sample(), 0, GraphKind::Decode, 1, 1, 2);
        }
        let value = telemetry.snapshot(0, "fixture");
        assert_eq!(value["frames"].as_array().unwrap().len(), CAPACITY);
        assert_eq!(value["frames"][0]["sequence"], 9);
        assert_eq!(
            telemetry.snapshot(39, "fixture")["frames"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn telemetry_content_failure_and_busy_consumer_never_block_producer() {
        let telemetry = Telemetry::default();
        telemetry.publish(RoutingReadout::Invalid, 0, GraphKind::Decode, 1, 1, 1);
        let lock = telemetry.history.lock().unwrap();
        telemetry.publish(sample(), 0, GraphKind::Decode, 1, 1, 1);
        drop(lock);
        let value = telemetry.snapshot(0, "fixture");
        assert_eq!(value["invalid_total"], 1);
        assert_eq!(value["dropped_total"], 1);
        assert_eq!(value["latest_sequence"], 2);
        assert!(value["frames"].as_array().unwrap().is_empty());
    }
}
