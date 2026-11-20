/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optional bounded routing-telemetry ring exposed through `GET /v1/parallax/routing`.

use crate::parallax::{RoutingReadout, RoutingSample};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

const CAPACITY: usize = 32;

#[derive(Serialize)]
struct Record {
    sequence: u64,
    graph: usize,
    phase: &'static str,
    batch_size: usize,
    concurrency: usize,
    sample: RoutingSample,
}

pub(super) struct Telemetry {
    geometry: Option<(usize, usize)>,
    session: String,
    sequence: AtomicU64,
    dropped: AtomicU64,
    invalid: AtomicU64,
    history: Mutex<VecDeque<Arc<Record>>>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            geometry: None,
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
        }
    }
}

impl Telemetry {
    pub fn configure(&mut self, manifest: &crate::execution::ExecutionManifest) {
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
        batch_size: usize,
        concurrency: usize,
    ) {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let RoutingReadout::Valid(sample) = readout else {
            self.invalid.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let record = Arc::new(Record {
            sequence,
            graph,
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
        let Ok(mut history) = self.history.try_lock() else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if history.len() == CAPACITY {
            history.pop_front();
        }
        history.push_back(record);
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
        json!({
            "version": 1,
            "session": self.session,
            "model_id": model,
            "enabled": self.geometry.is_some(),
            "geometry": self.geometry,
            "clock": "native_process_monotonic_ns",
            "native_now_ns": crate::parallax::now_ns(),
            "demand_timing": "host_dispatch_lower_bound",
            "demand_scope": "target_expert_execution_including_rejected_candidates; draft experts excluded",
            "capacity": CAPACITY,
            "latest_sequence": self.sequence.load(Ordering::Relaxed),
            "dropped_total": self.dropped.load(Ordering::Relaxed),
            "invalid_total": self.invalid.load(Ordering::Relaxed),
            "frames": rows
        })
    }
}

pub(super) fn parse_after(query: Option<&str>) -> Result<u64, &'static str> {
    match query {
        None | Some("") => Ok(0),
        Some(query) => query
            .strip_prefix("after=")
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or("expected ?after=<routing sequence>"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::GraphKind;

    fn sample() -> RoutingReadout {
        RoutingReadout::Valid(RoutingSample {
            gpu_ns: None,
            submission: 1,
            demand_lower_bound_ns: 10,
            available_ns: 20,
            layers: Vec::new(),
        })
    }

    #[test]
    fn retention_and_cursor_are_bounded_and_loss_is_visible() {
        let telemetry = Telemetry::default();
        for _ in 0..40 {
            telemetry.publish(sample(), 0, GraphKind::Decode, 1, 2);
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
        telemetry.publish(RoutingReadout::Invalid, 0, GraphKind::Decode, 1, 1);
        let lock = telemetry.history.lock().unwrap();
        telemetry.publish(sample(), 0, GraphKind::Decode, 1, 1);
        drop(lock);
        let value = telemetry.snapshot(0, "fixture");
        assert_eq!(value["invalid_total"], 1);
        assert_eq!(value["dropped_total"], 1);
        assert_eq!(value["latest_sequence"], 2);
        assert!(value["frames"].as_array().unwrap().is_empty());
    }

    #[test]
    fn routing_query_accepts_empty_or_nonnegative_sequence_and_rejects_malformed() {
        assert_eq!(parse_after(None), Ok(0));
        assert_eq!(parse_after(Some("")), Ok(0));
        assert_eq!(parse_after(Some("after=42")), Ok(42));
        for bad in ["after=-1", "after=abc", "before=1", "after=1&extra=2"] {
            assert!(parse_after(Some(bad)).is_err());
        }
    }
}
