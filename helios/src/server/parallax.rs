//! Authenticated HTTP status and plan submission endpoints for VMM expert residency.

use crate::parallax::{
    ExpertId, now_ns,
    residency::{Error as ResidencyError, Plan, execution::ExecutionBinding},
};
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResidencyPlanRequest {
    pub version: u32,
    pub catalog_id: String,
    pub generation: u64,
    /// Deadline in Helios's native monotonic clock (`native_now_ns`).
    pub valid_until_ns: u64,
    #[serde(default)]
    pub promote: Vec<ExpertId>,
    #[serde(default)]
    pub evict: Vec<ExpertId>,
}

pub(super) fn status_response(binding: Option<&ExecutionBinding>) -> Response {
    let now = now_ns();
    let Some(binding) = binding else {
        return Json(json!({
            "version": 1,
            "enabled": false,
            "clock": "native_process_monotonic_ns",
            "native_now_ns": now,
        }))
        .into_response();
    };
    let report = match binding.report() {
        Ok(report) => report,
        Err(error) => {
            return super::error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("expert residency status unavailable: {error}"),
            );
        }
    };
    let constraints = match binding.placement_constraints() {
        Ok(constraints) => constraints,
        Err(error) => {
            return super::error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("expert residency constraints unavailable: {error}"),
            );
        }
    };
    let state = report.snapshot;
    let hbm = state.experts.values().filter(|resident| **resident).count();
    let experts: Vec<Value> = state
        .experts
        .iter()
        .map(|(&expert, &resident)| {
            let constraint = constraints.get(&expert);
            json!({
                "layer": expert.layer,
                "expert": expert.expert,
                "resident": resident,
                "group": constraint.map(|c| c.group),
                "evictable": constraint.map(|c| c.evictable),
            })
        })
        .collect();
    Json(json!({
        "version": 1,
        "enabled": true,
        "clock": "native_process_monotonic_ns",
        "native_now_ns": now,
        "catalog_id": state.catalog_id,
        "generation": state.generation,
        "hbm_experts": hbm,
        "grace_experts": state.experts.len().saturating_sub(hbm),
        "experts": experts,
        "hbm_pool_bytes": state.hbm_allocated_bytes,
        "grace_backing_bytes": state.backing_allocated_bytes,
        "free_slots": state.free_slots,
        "free_slots_by_group": state.free_slots_by_group,
        "pending": state.pending,
        "readers": state.readers,
        "poisoned": state.poisoned,
        "commits_including_startup": state.statistics.commits,
        "staged_copy_bytes_including_startup": state.statistics.staged_bytes,
        "committed_copy_bytes_including_startup": state.statistics.committed_copy_bytes,
        "promoted_experts_including_startup": state.statistics.promoted_experts,
        "evicted_experts_including_startup": state.statistics.evicted_experts,
        "publication_host_ns_including_startup": state.statistics.publication_host_ns,
        "ready_wait_ns_including_startup": state.statistics.ready_wait_ns,
        "copy_gpu_ns_including_startup": state.statistics.copy_gpu_ns,
        "timed_copy_transactions_including_startup": state.statistics.timed_copies,
        "rollbacks": state.statistics.rollbacks,
        "rejected_plans": state.statistics.rejected,
        "cancelled_transactions": state.statistics.cancellations,
        "movement_error": report.disabled,
        "movement_history_capacity": 256,
        "movement_event_total": report.event_count,
        "movement_events": report.events,
    }))
    .into_response()
}

pub(super) fn submit_plan_response(binding: Option<&ExecutionBinding>, body: &[u8]) -> Response {
    let Some(binding) = binding else {
        return super::error(
            StatusCode::BAD_REQUEST,
            "physical expert residency is not enabled",
        );
    };
    let request: ResidencyPlanRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(error) => {
            return super::error(
                StatusCode::BAD_REQUEST,
                format!("invalid residency plan JSON: {error}"),
            );
        }
    };
    let report = match binding.report() {
        Ok(report) => report,
        Err(error) => {
            return super::error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("expert residency storage unavailable: {error}"),
            );
        }
    };
    if report.snapshot.poisoned {
        return super::error(
            StatusCode::SERVICE_UNAVAILABLE,
            "expert residency storage is poisoned",
        );
    }
    if let Some(disabled) = report.disabled {
        return super::error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("expert residency movement is disabled: {disabled}"),
        );
    }
    if request.version != 1 {
        return super::error(
            StatusCode::BAD_REQUEST,
            "unsupported residency plan version; expected 1",
        );
    }
    let snapshot = &report.snapshot;
    if request.catalog_id != snapshot.catalog_id {
        return super::error(
            StatusCode::BAD_REQUEST,
            "residency plan catalog_id does not match loaded storage",
        );
    }
    if request.generation != snapshot.generation {
        return super::error(
            StatusCode::CONFLICT,
            format!(
                "stale residency generation {}; current generation is {}",
                request.generation, snapshot.generation
            ),
        );
    }
    let now = now_ns();
    if request.valid_until_ns < now {
        return super::error(
            StatusCode::BAD_REQUEST,
            "residency plan valid_until_ns is already expired",
        );
    }
    let constraints = match binding.placement_constraints() {
        Ok(constraints) => constraints,
        Err(error) => {
            return super::error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("expert residency constraints unavailable: {error}"),
            );
        }
    };
    let storage = match binding.expert_storage() {
        Ok(storage) => storage,
        Err(error) => {
            return super::error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("expert residency storage unavailable: {error}"),
            );
        }
    };
    let mut regions_per_expert = BTreeMap::<ExpertId, usize>::new();
    for (expert, _transfer_bytes, parts) in storage {
        regions_per_expert.insert(expert, parts);
    }

    let mut seen_promote = BTreeSet::new();
    let mut needed_slots_by_group = BTreeMap::<usize, usize>::new();
    for &expert in &request.promote {
        if !seen_promote.insert(expert) {
            return super::error(
                StatusCode::BAD_REQUEST,
                format!(
                    "duplicate promote expert (layer {}, expert {})",
                    expert.layer, expert.expert
                ),
            );
        }
        let Some(&resident) = snapshot.experts.get(&expert) else {
            return super::error(
                StatusCode::BAD_REQUEST,
                format!(
                    "unknown expert (layer {}, expert {})",
                    expert.layer, expert.expert
                ),
            );
        };
        if resident {
            return super::error(
                StatusCode::BAD_REQUEST,
                format!(
                    "expert (layer {}, expert {}) is already resident in HBM",
                    expert.layer, expert.expert
                ),
            );
        }
        if let Some(constraint) = constraints.get(&expert) {
            let slots = regions_per_expert.get(&expert).copied().unwrap_or(0);
            let entry = needed_slots_by_group.entry(constraint.group).or_default();
            *entry = entry.saturating_add(slots);
        }
    }

    let mut seen_evict = BTreeSet::new();
    for &expert in &request.evict {
        if !seen_evict.insert(expert) || seen_promote.contains(&expert) {
            return super::error(
                StatusCode::BAD_REQUEST,
                format!(
                    "duplicate or conflicting evict expert (layer {}, expert {})",
                    expert.layer, expert.expert
                ),
            );
        }
        let Some(&resident) = snapshot.experts.get(&expert) else {
            return super::error(
                StatusCode::BAD_REQUEST,
                format!(
                    "unknown expert (layer {}, expert {})",
                    expert.layer, expert.expert
                ),
            );
        };
        if !resident {
            return super::error(
                StatusCode::BAD_REQUEST,
                format!(
                    "expert (layer {}, expert {}) is not resident in HBM",
                    expert.layer, expert.expert
                ),
            );
        }
        if !constraints.get(&expert).is_some_and(|c| c.evictable) {
            return super::error(
                StatusCode::BAD_REQUEST,
                format!(
                    "expert (layer {}, expert {}) has no backing copy and cannot be evicted",
                    expert.layer, expert.expert
                ),
            );
        }
    }

    for (group, needed) in needed_slots_by_group {
        let free = snapshot
            .free_slots_by_group
            .get(&group)
            .copied()
            .unwrap_or(0);
        if needed > free {
            return super::error(
                StatusCode::CONFLICT,
                format!(
                    "insufficient spare HBM slots in pool group {group}: need {needed}, have {free}"
                ),
            );
        }
    }

    if request.promote.is_empty() && request.evict.is_empty() {
        return (
            StatusCode::OK,
            Json(json!({
                "status": "noop",
                "catalog_id": request.catalog_id,
                "generation": request.generation,
            })),
        )
            .into_response();
    }

    let plan = Plan {
        version: request.version,
        catalog_id: request.catalog_id.clone(),
        generation: request.generation,
        valid_until_ns: request.valid_until_ns,
        promote: request.promote.clone(),
        evict: request.evict.clone(),
    };
    match binding.submit(plan) {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(json!({
                "status": "queued",
                "catalog_id": request.catalog_id,
                "generation": request.generation,
                "valid_until_ns": request.valid_until_ns,
                "promote": request.promote,
                "evict": request.evict,
            })),
        )
            .into_response(),
        Err(ResidencyError::Busy) => super::error(
            StatusCode::CONFLICT,
            "a residency transaction is already queued or in progress",
        ),
        Err(ResidencyError::Poisoned(error)) => super::error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("expert residency storage is poisoned: {error}"),
        ),
        Err(ResidencyError::Invalid(message)) if message.contains("disabled") || message.contains("closed") => {
            super::error(StatusCode::SERVICE_UNAVAILABLE, message)
        }
        Err(error) => super::error(StatusCode::BAD_REQUEST, error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallax::residency::{
        Backend, Binding, Capabilities, Controller, Location, Region, Result as ResidencyResult,
    };

    struct FakeBackend {
        regions: Vec<Region>,
        initial: Vec<Location>,
        backing: Vec<bool>,
        slots: usize,
        ready: bool,
        poison_on_publish: bool,
    }

    unsafe impl Backend for FakeBackend {
        fn regions(&self) -> &[Region] {
            &self.regions
        }
        fn initial_location(&self, region: usize) -> Location {
            self.initial[region]
        }
        fn has_backing(&self, region: usize) -> bool {
            self.backing[region]
        }
        fn slots(&self) -> usize {
            self.slots
        }
        fn slot_bytes(&self) -> usize {
            4096
        }
        fn backing_bytes(&self) -> usize {
            self.regions.len() * 4096
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
            0x10000 + (region * 4096) as u64
        }
        fn stage(&mut self, _copies: &[(usize, usize)]) -> ResidencyResult<()> {
            Ok(())
        }
        fn ready(&mut self) -> ResidencyResult<bool> {
            Ok(self.ready)
        }
        fn discard(&mut self) -> ResidencyResult<()> {
            Ok(())
        }
        unsafe fn publish(&mut self, _bindings: &[Binding]) -> ResidencyResult<()> {
            if self.poison_on_publish {
                return Err(ResidencyError::Poisoned("simulated poison".into()));
            }
            Ok(())
        }
    }

    fn fixture(slots: usize, poison_on_publish: bool) -> (ExecutionBinding, crate::parallax::residency::execution::ExecutionGate) {
        // Two parts per expert (e.g., w13 and w2) across 3 experts:
        // expert 0: resident in HBM, evictable
        // expert 1: resident in HBM, pinned (no backing)
        // expert 2: in Grace backing, promotable
        let mut regions = Vec::new();
        let mut initial = Vec::new();
        let mut backing = Vec::new();
        for expert in 0..3 {
            for part in 0..2 {
                regions.push(Region {
                    expert: ExpertId { layer: 0, expert },
                    part,
                    offset: (expert * 2 + part) * 4096,
                    bytes: 4096,
                });
                let loc = match expert {
                    0 => Location::Hbm(part),
                    1 => Location::Hbm(2 + part),
                    _ => Location::Backing,
                };
                initial.push(loc);
                backing.push(expert != 1);
            }
        }
        let backend = FakeBackend {
            regions,
            initial,
            backing,
            slots,
            ready: true,
            poison_on_publish,
        };
        let controller = Controller::new("cat-v1".into(), backend).unwrap();
        let binding = ExecutionBinding::new(controller);
        let gate = unsafe { binding.claim() }.unwrap();
        (binding, gate)
    }

    async fn read_json(response: Response) -> (StatusCode, Value) {
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn disabled_status_reports_false_and_plan_rejects_with_bad_request() {
        let (status, body) = read_json(status_response(None)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["version"], 1);
        assert_eq!(body["enabled"], false);
        assert_eq!(body["clock"], "native_process_monotonic_ns");
        assert!(body["native_now_ns"].as_u64().is_some());

        let (status, body) = read_json(submit_plan_response(None, b"{}")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not enabled")
        );
    }

    #[tokio::test]
    async fn valid_plan_queues_executes_and_updates_status_snapshot() {
        let (binding, mut gate) = fixture(6, false);

        let (status, body) = read_json(status_response(Some(&binding))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["enabled"], true);
        assert_eq!(body["catalog_id"], "cat-v1");
        assert_eq!(body["generation"], 0);
        assert_eq!(body["hbm_experts"], 2);
        assert_eq!(body["grace_experts"], 1);
        assert_eq!(body["free_slots"], 2);
        assert_eq!(body["experts"].as_array().unwrap().len(), 3);
        assert_eq!(body["experts"][1]["evictable"], false);

        let plan = serde_json::to_vec(&ResidencyPlanRequest {
            version: 1,
            catalog_id: "cat-v1".into(),
            generation: 0,
            valid_until_ns: u64::MAX,
            promote: vec![ExpertId {
                layer: 0,
                expert: 2,
            }],
            evict: vec![ExpertId {
                layer: 0,
                expert: 0,
            }],
        })
        .unwrap();

        let (status, body) = read_json(submit_plan_response(Some(&binding), &plan)).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(body["status"], "queued");

        // Second submission while first is queued returns 409 Conflict.
        let (status, _) = read_json(submit_plan_response(Some(&binding), &plan)).await;
        assert_eq!(status, StatusCode::CONFLICT);

        // Executor tick stages and commits immediately with no active readers.
        unsafe { gate.tick(now_ns()) }.unwrap();

        let (status, body) = read_json(status_response(Some(&binding))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["generation"], 1);
        assert_eq!(body["hbm_experts"], 2);
        assert_eq!(body["grace_experts"], 1);
        assert_eq!(body["experts"][0]["resident"], false);
        assert_eq!(body["experts"][2]["resident"], true);
        assert_eq!(body["movement_event_total"], 1);
        assert_eq!(body["movement_events"][0]["outcome"], "committed:1");
    }

    #[tokio::test]
    async fn invalid_stale_overcapacity_and_noop_plans_return_expected_statuses() {
        let (binding, _gate) = fixture(5, false); // Only 1 free slot; 2 needed for one 2-part expert.

        // Malformed JSON / unknown field -> 400.
        let (status, _) = read_json(submit_plan_response(Some(&binding), b"{\"unknown\":1}")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // Wrong version -> 400.
        let bad_version = serde_json::to_vec(&ResidencyPlanRequest {
            version: 2,
            catalog_id: "cat-v1".into(),
            generation: 0,
            valid_until_ns: u64::MAX,
            promote: vec![],
            evict: vec![],
        })
        .unwrap();
        assert_eq!(
            read_json(submit_plan_response(Some(&binding), &bad_version))
                .await
                .0,
            StatusCode::BAD_REQUEST
        );

        // Wrong catalog -> 400.
        let bad_catalog = serde_json::to_vec(&ResidencyPlanRequest {
            version: 1,
            catalog_id: "other".into(),
            generation: 0,
            valid_until_ns: u64::MAX,
            promote: vec![],
            evict: vec![],
        })
        .unwrap();
        assert_eq!(
            read_json(submit_plan_response(Some(&binding), &bad_catalog))
                .await
                .0,
            StatusCode::BAD_REQUEST
        );

        // Stale generation -> 409.
        let stale = serde_json::to_vec(&ResidencyPlanRequest {
            version: 1,
            catalog_id: "cat-v1".into(),
            generation: 7,
            valid_until_ns: u64::MAX,
            promote: vec![],
            evict: vec![],
        })
        .unwrap();
        assert_eq!(
            read_json(submit_plan_response(Some(&binding), &stale))
                .await
                .0,
            StatusCode::CONFLICT
        );

        // Expired deadline -> 400.
        let expired = serde_json::to_vec(&ResidencyPlanRequest {
            version: 1,
            catalog_id: "cat-v1".into(),
            generation: 0,
            valid_until_ns: 0,
            promote: vec![],
            evict: vec![],
        })
        .unwrap();
        assert_eq!(
            read_json(submit_plan_response(Some(&binding), &expired))
                .await
                .0,
            StatusCode::BAD_REQUEST
        );

        // Pinned (non-evictable) expert -> 400.
        let pinned = serde_json::to_vec(&ResidencyPlanRequest {
            version: 1,
            catalog_id: "cat-v1".into(),
            generation: 0,
            valid_until_ns: u64::MAX,
            promote: vec![],
            evict: vec![ExpertId {
                layer: 0,
                expert: 1,
            }],
        })
        .unwrap();
        assert_eq!(
            read_json(submit_plan_response(Some(&binding), &pinned))
                .await
                .0,
            StatusCode::BAD_REQUEST
        );

        // Insufficient multi-part slots (1 free, 2 required) -> 409.
        let no_space = serde_json::to_vec(&ResidencyPlanRequest {
            version: 1,
            catalog_id: "cat-v1".into(),
            generation: 0,
            valid_until_ns: u64::MAX,
            promote: vec![ExpertId {
                layer: 0,
                expert: 2,
            }],
            evict: vec![ExpertId {
                layer: 0,
                expert: 0,
            }],
        })
        .unwrap();
        assert_eq!(
            read_json(submit_plan_response(Some(&binding), &no_space))
                .await
                .0,
            StatusCode::CONFLICT
        );

        // Empty promote + evict -> 200 noop without occupying queue slot.
        let noop = serde_json::to_vec(&ResidencyPlanRequest {
            version: 1,
            catalog_id: "cat-v1".into(),
            generation: 0,
            valid_until_ns: u64::MAX,
            promote: vec![],
            evict: vec![],
        })
        .unwrap();
        let (status, body) = read_json(submit_plan_response(Some(&binding), &noop)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "noop");
    }

    #[tokio::test]
    async fn poisoned_storage_rejects_plans_with_service_unavailable() {
        let (binding, mut gate) = fixture(6, true);
        let plan = serde_json::to_vec(&ResidencyPlanRequest {
            version: 1,
            catalog_id: "cat-v1".into(),
            generation: 0,
            valid_until_ns: u64::MAX,
            promote: vec![ExpertId {
                layer: 0,
                expert: 2,
            }],
            evict: vec![],
        })
        .unwrap();
        assert_eq!(
            read_json(submit_plan_response(Some(&binding), &plan))
                .await
                .0,
            StatusCode::ACCEPTED
        );
        assert!(unsafe { gate.tick(now_ns()) }.is_err());

        let (status, body) = read_json(submit_plan_response(Some(&binding), &plan)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("poisoned")
        );
    }
}
