/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generic persistent-worker CUDA Green Context partition and lifecycle controller.

use std::ffi::c_uint;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub use crate::execution::WorkerState;
use crate::execution::{PersistentWorkerControl, PersistentWorkerSnapshot};

use super::driver::{
    CU_DEV_RESOURCE_TYPE_SM, CU_GREEN_CTX_DEFAULT_STREAM, CU_STREAM_NON_BLOCKING, DevResource,
    Driver, Handle,
};
use super::format::invalid;
use crate::{Error, ExecutionError, Result};

/// Partitioned CUDA Green Contexts and non-blocking streams for a core with a persistent worker.
pub(super) struct GreenContextPartition {
    pub(super) worker_green_ctx: Handle,
    pub(super) compute_green_ctx: Handle,
    pub(super) worker_context: Handle,
    pub(super) compute_context: Handle,
    pub(super) worker_stream: Handle,
    pub(super) compute_stream: Handle,
    pub(super) worker_sm_count: u32,
    pub(super) compute_sm_count: u32,
}

// SAFETY: CUDA green contexts and streams may be transferred across threads while owned by
// `NativeResources`, which serializes creation and teardown.
unsafe impl Send for GreenContextPartition {}

impl GreenContextPartition {
    /// Split the device's SMs into a worker partition of at least `requested_worker_sms` and a
    /// non-empty compute remainder, then create green contexts and non-blocking streams for both.
    pub(super) unsafe fn create(
        driver: &Driver,
        device: i32,
        requested_worker_sms: u32,
    ) -> Result<Self> {
        let gc = driver
            .green_ctx
            .as_ref()
            .ok_or_else(|| invalid("persistent worker requires CUDA driver Green Context APIs"))?;
        let mut partition = Self {
            worker_green_ctx: null_mut(),
            compute_green_ctx: null_mut(),
            worker_context: null_mut(),
            compute_context: null_mut(),
            worker_stream: null_mut(),
            compute_stream: null_mut(),
            worker_sm_count: 0,
            compute_sm_count: 0,
        };
        let setup = (|| -> Result<()> {
            // SAFETY: Output structs and pointers satisfy cuda.h contracts; any handles created
            // before an error are released by `cleanup` below.
            unsafe {
                let mut device_sm = DevResource::zeroed();
                driver.check(
                    "cuDeviceGetDevResource(SM)",
                    (gc.device_get_dev_resource)(device, &mut device_sm, CU_DEV_RESOURCE_TYPE_SM),
                )?;
                let total_sms = device_sm
                    .sm()
                    .map(|sm| sm.sm_count)
                    .filter(|&count| count > 0)
                    .ok_or_else(|| invalid("device reported zero streaming multiprocessors"))?;
                if requested_worker_sms == 0 || requested_worker_sms >= total_sms {
                    return Err(invalid(format!(
                        "persistent worker requested {requested_worker_sms} SMs out of {total_sms}, leaving no compute SMs"
                    )));
                }
                let mut worker_res = DevResource::zeroed();
                let mut compute_res = DevResource::zeroed();
                let mut nb_groups: c_uint = 1;
                driver.check(
                    "cuDevSmResourceSplitByCount",
                    (gc.dev_sm_resource_split_by_count)(
                        &mut worker_res,
                        &mut nb_groups,
                        &device_sm,
                        &mut compute_res,
                        0,
                        requested_worker_sms,
                    ),
                )?;
                let worker_split = worker_res.sm().map(|sm| sm.sm_count).unwrap_or(0);
                let compute_split = compute_res.sm().map(|sm| sm.sm_count).unwrap_or(0);
                if nb_groups != 1 || worker_split == 0 || compute_split == 0 {
                    return Err(invalid(format!(
                        "SM split for persistent worker ({requested_worker_sms} requested out of {total_sms}) left no compute SMs"
                    )));
                }
                let mut worker_desc = null_mut();
                driver.check(
                    "cuDevResourceGenerateDesc(worker)",
                    (gc.dev_resource_generate_desc)(&mut worker_desc, &mut worker_res, 1),
                )?;
                let mut compute_desc = null_mut();
                driver.check(
                    "cuDevResourceGenerateDesc(compute)",
                    (gc.dev_resource_generate_desc)(&mut compute_desc, &mut compute_res, 1),
                )?;
                driver.check(
                    "cuGreenCtxCreate(worker)",
                    (gc.green_ctx_create)(
                        &mut partition.worker_green_ctx,
                        worker_desc,
                        device,
                        CU_GREEN_CTX_DEFAULT_STREAM,
                    ),
                )?;
                driver.check(
                    "cuGreenCtxCreate(compute)",
                    (gc.green_ctx_create)(
                        &mut partition.compute_green_ctx,
                        compute_desc,
                        device,
                        CU_GREEN_CTX_DEFAULT_STREAM,
                    ),
                )?;
                let mut actual_worker = DevResource::zeroed();
                driver.check(
                    "cuGreenCtxGetDevResource(worker)",
                    (gc.green_ctx_get_dev_resource)(
                        partition.worker_green_ctx,
                        &mut actual_worker,
                        CU_DEV_RESOURCE_TYPE_SM,
                    ),
                )?;
                let mut actual_compute = DevResource::zeroed();
                driver.check(
                    "cuGreenCtxGetDevResource(compute)",
                    (gc.green_ctx_get_dev_resource)(
                        partition.compute_green_ctx,
                        &mut actual_compute,
                        CU_DEV_RESOURCE_TYPE_SM,
                    ),
                )?;
                partition.worker_sm_count = actual_worker.sm().map(|sm| sm.sm_count).unwrap_or(0);
                partition.compute_sm_count = actual_compute.sm().map(|sm| sm.sm_count).unwrap_or(0);
                if partition.worker_sm_count == 0 || partition.compute_sm_count == 0 {
                    return Err(invalid("green context partition has zero SMs"));
                }
                driver.check(
                    "cuCtxFromGreenCtx(worker)",
                    (gc.ctx_from_green_ctx)(
                        &mut partition.worker_context,
                        partition.worker_green_ctx,
                    ),
                )?;
                driver.check(
                    "cuCtxFromGreenCtx(compute)",
                    (gc.ctx_from_green_ctx)(
                        &mut partition.compute_context,
                        partition.compute_green_ctx,
                    ),
                )?;
                driver.check(
                    "cuGreenCtxStreamCreate(worker)",
                    (gc.green_ctx_stream_create)(
                        &mut partition.worker_stream,
                        partition.worker_green_ctx,
                        CU_STREAM_NON_BLOCKING,
                        0,
                    ),
                )?;
                driver.check(
                    "cuGreenCtxStreamCreate(compute)",
                    (gc.green_ctx_stream_create)(
                        &mut partition.compute_stream,
                        partition.compute_green_ctx,
                        CU_STREAM_NON_BLOCKING,
                        0,
                    ),
                )?;
                Ok(())
            }
        })();
        if let Err(error) = setup {
            // SAFETY: No work was launched on these newly created streams or contexts.
            unsafe {
                partition.destroy_streams(driver);
                partition.destroy_green_contexts(driver);
            }
            return Err(error);
        }
        Ok(partition)
    }

    /// Destroy the worker and compute streams after all work on them has drained.
    pub(super) unsafe fn destroy_streams(&mut self, driver: &Driver) {
        unsafe {
            if !self.worker_stream.is_null() {
                (driver.stream_destroy)(self.worker_stream);
                self.worker_stream = null_mut();
            }
            if !self.compute_stream.is_null() {
                (driver.stream_destroy)(self.compute_stream);
                self.compute_stream = null_mut();
            }
        }
    }

    /// Destroy both green contexts after streams, graphs, modules, and allocations are released.
    pub(super) unsafe fn destroy_green_contexts(&mut self, driver: &Driver) {
        self.worker_context = null_mut();
        self.compute_context = null_mut();
        let Some(gc) = driver.green_ctx.as_ref() else {
            self.worker_green_ctx = null_mut();
            self.compute_green_ctx = null_mut();
            return;
        };
        unsafe {
            if !self.worker_green_ctx.is_null() {
                (gc.green_ctx_destroy)(self.worker_green_ctx);
                self.worker_green_ctx = null_mut();
            }
            if !self.compute_green_ctx.is_null() {
                (gc.green_ctx_destroy)(self.compute_green_ctx);
                self.compute_green_ctx = null_mut();
            }
        }
    }
}

/// Lock-free atomic read of a mapped-host `u64` counter array without GPU synchronization.
///
/// Returns an empty vector when the range is absent, unaligned, or would overflow address bounds.
///
/// # Safety
/// When `metrics_count > 0` and `metrics_address != 0`, `metrics_address` must point to
/// `metrics_count` live, 8-byte-aligned host-readable `u64` words.
unsafe fn read_mapped_counters(metrics_address: usize, metrics_count: usize) -> Vec<u64> {
    if metrics_count == 0
        || metrics_address == 0
        || metrics_address % std::mem::align_of::<u64>() != 0
    {
        return Vec::new();
    }
    let Some(bytes) = metrics_count.checked_mul(std::mem::size_of::<u64>()) else {
        return Vec::new();
    };
    if metrics_address.checked_add(bytes).is_none() {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(metrics_count);
    for index in 0..metrics_count {
        let ptr = (metrics_address + index * std::mem::size_of::<u64>()) as *mut u64;
        // SAFETY: Validated mapped-host range aligned to 8 bytes and retained through teardown.
        values.push(unsafe { AtomicU64::from_ptr(ptr).load(Ordering::Acquire) });
    }
    values
}

/// Deterministic state machine governing start, stop, and atomic stop-flag updates.
struct PersistentWorkerMachine {
    state: WorkerState,
    stop_flag_address: usize,
}

impl PersistentWorkerMachine {
    fn new(stop_flag_address: usize) -> Self {
        Self {
            state: WorkerState::Idle,
            stop_flag_address,
        }
    }

    fn state(&self) -> WorkerState {
        self.state
    }

    /// Zero the mapped-host stop flag and invoke `launch` once.
    ///
    /// Calling `start` while already `Running` is an idempotent no-op.
    /// Calling `start` after `Stopped` or `FailedDrain` fails explicitly.
    ///
    /// # Safety
    /// `stop_flag_address` must point to live, 4-byte-aligned, host-writable memory.
    unsafe fn start(&mut self, launch: impl FnOnce() -> Result<()>) -> Result<()> {
        match self.state {
            WorkerState::Idle => {
                // SAFETY: Validated mapped-host address aligned to 4 bytes and retained through drain.
                unsafe {
                    AtomicU32::from_ptr(self.stop_flag_address as *mut u32)
                        .store(0, Ordering::Release);
                }
                launch()?;
                self.state = WorkerState::Running;
                Ok(())
            }
            WorkerState::Running => Ok(()),
            WorkerState::Draining => Err(invalid("persistent worker is draining")),
            WorkerState::Stopped => Err(invalid("persistent worker has already been stopped")),
            WorkerState::FailedDrain => Err(Error::execution(
                ExecutionError::Drain,
                "persistent worker failed to drain",
            )),
        }
    }

    /// Store `1u32` (Release) to the mapped-host stop flag and invoke `sync` to drain the worker.
    ///
    /// Idempotent when `Idle` or `Stopped`. Transitions to `FailedDrain` if `sync` fails.
    ///
    /// # Safety
    /// `stop_flag_address` must point to live, 4-byte-aligned, host-writable memory.
    #[cfg(test)]
    unsafe fn stop(&mut self, sync: impl FnOnce() -> Result<()>) -> Result<()> {
        unsafe { self.stop_with_hook(|| {}, sync) }
    }

    unsafe fn stop_with_hook(
        &mut self,
        on_draining: impl FnOnce(),
        sync: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        match self.state {
            WorkerState::Idle => {
                self.state = WorkerState::Stopped;
                Ok(())
            }
            WorkerState::Running => {
                on_draining();
                // SAFETY: Validated mapped-host address aligned to 4 bytes and retained through drain.
                unsafe {
                    AtomicU32::from_ptr(self.stop_flag_address as *mut u32)
                        .store(1, Ordering::Release);
                }
                match sync() {
                    Ok(()) => {
                        self.state = WorkerState::Stopped;
                        Ok(())
                    }
                    Err(error) => {
                        self.state = WorkerState::FailedDrain;
                        Err(error)
                    }
                }
            }
            WorkerState::Draining => Err(invalid("persistent worker is already draining")),
            WorkerState::Stopped => Ok(()),
            WorkerState::FailedDrain => Err(Error::execution(
                ExecutionError::Drain,
                "persistent worker failed to drain",
            )),
        }
    }
}

/// Thread-safe runtime controller shared between `NativeResources` and `ExecutionManifest`.
pub(super) struct PersistentWorkerRuntime {
    name: String,
    worker_sm_count: u32,
    driver: Arc<Driver>,
    context: Handle,
    worker_stream: Handle,
    worker_graph: Handle,
    metrics_address: usize,
    metrics_count: usize,
    metrics_live: RwLock<bool>,
    atomic_state: AtomicU8,
    machine: Mutex<PersistentWorkerMachine>,
}

// SAFETY: State transitions and CUDA driver calls are serialized by `self.machine`, every driver
// operation saves and restores the calling thread's CUDA context, and telemetry reads hold a
// shared `metrics_live` read guard while reading validated mapped-host memory so teardown waits
// for active readers before unmapping allocations.
unsafe impl Send for PersistentWorkerRuntime {}
unsafe impl Sync for PersistentWorkerRuntime {}

impl PersistentWorkerRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        name: String,
        worker_sm_count: u32,
        driver: Arc<Driver>,
        context: Handle,
        worker_stream: Handle,
        worker_graph: Handle,
        stop_flag_address: usize,
        metrics_address: usize,
        metrics_count: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            name,
            worker_sm_count,
            driver,
            context,
            worker_stream,
            worker_graph,
            metrics_address,
            metrics_count,
            metrics_live: RwLock::new(metrics_count > 0 && metrics_address != 0),
            atomic_state: AtomicU8::new(WorkerState::Idle.to_u8()),
            machine: Mutex::new(PersistentWorkerMachine::new(stop_flag_address)),
        })
    }

    fn with_context(&self, operation: impl FnOnce() -> Result<()>) -> Result<()> {
        // SAFETY: `self.context` remains live while the worker is active or draining,
        // and the previous context is restored before returning.
        unsafe {
            let mut previous = null_mut();
            let had_previous = (self.driver.context_get)(&mut previous) == 0;
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            let result = operation();
            if had_previous && previous != self.context {
                let _ = (self.driver.context_set)(previous);
            }
            result
        }
    }

    pub(super) fn start(&self) -> Result<()> {
        let mut machine = self
            .machine
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // SAFETY: Mapped-host stop flag remains owned by `NativeResources` through teardown.
        let result = unsafe {
            machine.start(|| {
                self.with_context(|| {
                    self.driver.check(
                        "cuGraphLaunch(persistent_worker)",
                        (self.driver.graph_launch)(self.worker_graph, self.worker_stream),
                    )
                })
            })
        };
        self.atomic_state
            .store(machine.state().to_u8(), Ordering::Release);
        result
    }

    pub(super) fn stop(&self) -> Result<()> {
        let mut machine = self
            .machine
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // SAFETY: Mapped-host stop flag remains owned by `NativeResources` through teardown.
        let result = unsafe {
            machine.stop_with_hook(
                || {
                    self.atomic_state
                        .store(WorkerState::Draining.to_u8(), Ordering::Release);
                },
                || {
                    self.with_context(|| {
                        self.driver.check(
                            "cuStreamSynchronize(persistent_worker)",
                            (self.driver.stream_sync)(self.worker_stream),
                        )
                    })
                },
            )
        };
        self.atomic_state
            .store(machine.state().to_u8(), Ordering::Release);
        result
    }

    /// Lock-free read of the worker lifecycle state.
    pub(super) fn state(&self) -> WorkerState {
        WorkerState::from_u8(self.atomic_state.load(Ordering::Acquire))
    }

    /// Atomic read of the mapped-host counter array (empty when absent or unmapped).
    pub(super) fn read_metrics(&self) -> Vec<u64> {
        let live = self
            .metrics_live
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        if !*live {
            return Vec::new();
        }
        // SAFETY: Holding the shared `metrics_live` read guard prevents teardown from unmapping
        // the validated mapped-host metrics range while counters are read.
        unsafe { read_mapped_counters(self.metrics_address, self.metrics_count) }
    }

    pub(super) fn metrics(&self) -> Option<Vec<u64>> {
        let live = self
            .metrics_live
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        if !*live {
            return None;
        }
        // SAFETY: Holding the shared `metrics_live` read guard prevents teardown from unmapping
        // the validated mapped-host metrics range while counters are read.
        Some(unsafe { read_mapped_counters(self.metrics_address, self.metrics_count) })
    }

    /// Telemetry snapshot without blocking on lifecycle mutexes or GPU streams.
    pub(super) fn snapshot(&self) -> PersistentWorkerSnapshot {
        PersistentWorkerSnapshot {
            name: self.name.clone(),
            worker_sm_count: self.worker_sm_count,
            state: self.state(),
            counters: self.read_metrics(),
        }
    }

    /// Drain active counter readers and prevent future reads before mapped-host allocations unmap.
    pub(super) fn invalidate_metrics(&self) {
        let mut live = self
            .metrics_live
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        *live = false;
    }

    pub(super) fn control(self: &Arc<Self>) -> PersistentWorkerControl {
        let start_handle = Arc::clone(self);
        let stop_handle = Arc::clone(self);
        let metrics_handle = Arc::clone(self);
        let snapshot_handle = Arc::clone(self);
        PersistentWorkerControl::new(
            Arc::new(move || start_handle.start()),
            Arc::new(move || stop_handle.stop()),
            Arc::new(move || metrics_handle.metrics()),
            Arc::new(move || snapshot_handle.snapshot()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_start_is_idempotent_and_stop_signals_flag_once() {
        let flag = AtomicU32::new(7);
        let metrics = [AtomicU64::new(11), AtomicU64::new(29)];
        let mut machine = PersistentWorkerMachine::new((&flag as *const AtomicU32) as usize);
        assert_eq!(machine.state(), WorkerState::Idle);
        let mut launches = 0usize;
        let mut syncs = 0usize;
        unsafe {
            machine
                .start(|| {
                    assert_eq!(flag.load(Ordering::Acquire), 0);
                    launches += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(machine.state(), WorkerState::Running);
            // Second start is idempotent and does not launch again.
            machine
                .start(|| {
                    launches += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(launches, 1);
            assert_eq!(
                read_mapped_counters(metrics.as_ptr() as usize, metrics.len()),
                vec![11, 29]
            );

            // Stop stores 1u32 before syncing and transitions to Stopped.
            machine
                .stop(|| {
                    assert_eq!(flag.load(Ordering::Acquire), 1);
                    syncs += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(machine.state(), WorkerState::Stopped);
            assert_eq!(syncs, 1);

            // Double stop is idempotent and does not sync again.
            machine
                .stop(|| {
                    syncs += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(syncs, 1);

            // Restart after stop is rejected.
            assert!(machine.start(|| Ok(())).is_err());
        }
    }

    #[test]
    fn stop_before_start_transitions_without_cuda_sync() {
        let flag = AtomicU32::new(0);
        let mut machine = PersistentWorkerMachine::new((&flag as *const AtomicU32) as usize);
        let mut syncs = 0usize;
        unsafe {
            machine
                .stop(|| {
                    syncs += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(syncs, 0);
            assert_eq!(machine.state(), WorkerState::Stopped);
            assert!(read_mapped_counters(0, 0).is_empty());
        }
    }

    #[test]
    fn launch_failure_leaves_worker_idle_and_drain_failure_is_sticky() {
        let flag = AtomicU32::new(9);
        let mut machine = PersistentWorkerMachine::new((&flag as *const AtomicU32) as usize);
        unsafe {
            assert!(
                machine
                    .start(|| Err(invalid("injected launch error")))
                    .is_err()
            );
            assert_eq!(machine.state(), WorkerState::Idle);
            assert_eq!(flag.load(Ordering::Acquire), 0);

            machine.start(|| Ok(())).unwrap();
            assert_eq!(machine.state(), WorkerState::Running);

            assert!(
                machine
                    .stop(|| Err(Error::execution(ExecutionError::Drain, "stuck stream")))
                    .is_err()
            );
            assert_eq!(machine.state(), WorkerState::FailedDrain);
            assert_eq!(flag.load(Ordering::Acquire), 1);

            // Subsequent stop and start both surface FailedDrain without re-invoking callbacks.
            assert!(machine.stop(|| Ok(())).is_err());
            assert!(machine.start(|| Ok(())).is_err());
        }
    }

    #[test]
    fn lock_free_read_mapped_counters_enforces_bounds_alignment_and_values() {
        let counters = [
            AtomicU64::new(10),
            AtomicU64::new(2048),
            AtomicU64::new(3),
            AtomicU64::new(7),
            AtomicU64::new((5u64 << 32) | 42),
            AtomicU64::new(950),
            AtomicU64::new(50),
            AtomicU64::new(18),
        ];
        let base = counters.as_ptr() as usize;
        unsafe {
            assert_eq!(
                read_mapped_counters(base, counters.len()),
                vec![10, 2048, 3, 7, (5u64 << 32) | 42, 950, 50, 18]
            );
            assert_eq!(read_mapped_counters(base, 3), vec![10, 2048, 3]);
            // Zero count, null address, unaligned address, and overflow all return empty safely.
            assert!(read_mapped_counters(base, 0).is_empty());
            assert!(read_mapped_counters(0, 4).is_empty());
            assert!(read_mapped_counters(base + 1, 1).is_empty());
            assert!(read_mapped_counters(usize::MAX - 7, 2).is_empty());
        }

        // Concurrent atomic store is observed without any lock or GPU sync.
        counters[0].store(99, Ordering::Release);
        counters[5].store(1024, Ordering::Release);
        unsafe {
            let read = read_mapped_counters(base, counters.len());
            assert_eq!(read[0], 99);
            assert_eq!(read[5], 1024);
        }
    }

    #[test]
    fn stop_hook_publishes_draining_before_stream_sync() {
        let flag = AtomicU32::new(0);
        let observed = AtomicU8::new(WorkerState::Idle.to_u8());
        let mut machine = PersistentWorkerMachine::new((&flag as *const AtomicU32) as usize);
        unsafe {
            machine.start(|| Ok(())).unwrap();
            observed.store(machine.state().to_u8(), Ordering::Release);
            assert_eq!(
                WorkerState::from_u8(observed.load(Ordering::Acquire)),
                WorkerState::Running
            );
            machine
                .stop_with_hook(
                    || observed.store(WorkerState::Draining.to_u8(), Ordering::Release),
                    || {
                        assert_eq!(
                            WorkerState::from_u8(observed.load(Ordering::Acquire)),
                            WorkerState::Draining
                        );
                        assert_eq!(flag.load(Ordering::Acquire), 1);
                        Ok(())
                    },
                )
                .unwrap();
            observed.store(machine.state().to_u8(), Ordering::Release);
            assert_eq!(
                WorkerState::from_u8(observed.load(Ordering::Acquire)),
                WorkerState::Stopped
            );
        }
    }

    #[test]
    #[ignore = "requires an assigned CUDA_VISIBLE_DEVICES GPU supporting Green Contexts"]
    fn cuda_green_context_persistent_worker_runs_concurrently_with_cooperative_compute_and_drains_cleanly()
     {
        use super::super::NativeResources;
        use super::super::archive::PayloadFile;
        use super::super::format::{
            Allocation, Context, Core, Function, Graph, Memory, Module, Reservation,
        };
        use super::super::payload::{Files, RecordReader};
        use super::super::protobuf::wire;
        use prost::Message;
        use std::collections::HashMap;
        use std::time::{Duration, Instant};

        struct TestFiles(HashMap<String, Vec<u8>>);

        impl Files for TestFiles {
            fn span<'a>(&self, _core: &'a Core, _key: &str) -> Result<&'a wire::Span> {
                Err(invalid("unused test span"))
            }
            fn source(&self, _span: &wire::Span) -> Result<(&PayloadFile, u64)> {
                Err(invalid("unused test source"))
            }
            fn read(&self, _core: &Core, key: &str) -> Result<Vec<u8>> {
                self.0
                    .get(key)
                    .cloned()
                    .ok_or_else(|| invalid(format!("missing test resource {key}")))
            }
            fn records<'a>(&'a self, _core: &Core, key: &str) -> Result<RecordReader<'a>> {
                let bytes = self
                    .0
                    .get(key)
                    .cloned()
                    .ok_or_else(|| invalid(format!("missing test record resource {key}")))?;
                Ok(RecordReader::Slice { bytes, position: 0 })
            }
            fn path(&self) -> &std::path::Path {
                std::path::Path::new("")
            }
            fn source_index(&self, _span: &wire::Span) -> Result<usize> {
                Err(invalid("unused test source_index"))
            }
            fn asset_path(&self, _span: &wire::Span) -> Result<std::path::PathBuf> {
                Err(invalid("unused test asset_path"))
            }
            fn release_page_cache(&self) {}
            fn retained_files(&self) -> Result<Vec<std::sync::Arc<std::fs::File>>> {
                Ok(Vec::new())
            }
            fn direct_files(
                &self,
            ) -> Box<dyn Iterator<Item = Result<std::sync::Arc<std::fs::File>>> + '_> {
                Box::new(std::iter::empty())
            }
            fn validate(&self, _core: &Core) -> Result<()> {
                Ok(())
            }
            fn resource_reader<'a>(
                &'a self,
                _core: &Core,
                _key: &str,
            ) -> Result<Box<dyn std::io::Read + 'a>> {
                Err(invalid("unused test resource_reader"))
            }
        }

        let driver = Driver::load().unwrap();
        let (major, minor, total_sms, max_smem_per_sm, max_smem_optin) = unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            let mut device = 0;
            driver
                .check("cuDeviceGet", (driver.device)(&mut device, 0))
                .unwrap();
            let attr = |code: i32| {
                let mut val = 0;
                driver
                    .check(
                        "cuDeviceGetAttribute",
                        (driver.device_attribute)(&mut val, code, device),
                    )
                    .unwrap();
                val
            };
            (attr(75), attr(76), attr(16), attr(81), attr(97))
        };
        let device_va = 0x800_0000_0000u64;
        let host_va = 0xc00_0000_0000u64;
        assert!(
            driver.green_ctx.is_some(),
            "GPU driver must support Green Contexts"
        );
        assert!(total_sms >= 16, "test requires >= 16 SMs, got {total_sms}");
        let one_block_smem = ((max_smem_per_sm as u32) / 2 + 4096).min(max_smem_optin as u32);

        let cu_source = r#"
extern "C" __global__ void worker_spin(const unsigned int* stop_flag, unsigned long long* metrics) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long iters = 0;
        while (true) {
            unsigned int stop = 0;
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(stop) : "l"(stop_flag) : "memory");
            if (stop != 0u) {
                break;
            }
            ++iters;
            asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(metrics), "l"(iters) : "memory");
            asm volatile("nanosleep.u32 500;");
        }
        asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(metrics + 1), "l"(1ULL) : "memory");
    }
}

extern "C" __global__ void coop_all_sms(int* state, int step) {
    extern __shared__ unsigned int dyn_smem[];
    if (threadIdx.x == 0) {
        const unsigned int n = gridDim.x;
        const unsigned int b = blockIdx.x;
        dyn_smem[0] = b;
        state[4 + b] = step * 1000 + static_cast<int>(dyn_smem[0]);
        __threadfence();
        unsigned int prev = 0;
        asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;" : "=r"(prev) : "l"(state) : "memory");
        (void)prev;
        while (true) {
            unsigned int count = 0;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(count) : "l"(state) : "memory");
            if (count >= n) {
                break;
            }
            asm volatile("nanosleep.u32 50;");
        }
        const int neighbor = state[4 + (n - 1u - b)];
        state[256 + b] = neighbor + 7;
        if (b == 0u) {
            state[2] = static_cast<int>(n);
        }
    }
}
"#;
        let unique = format!(
            "flare-worker-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        )
        .replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_");
        let cu_path = std::env::temp_dir().join(format!("{unique}.cu"));
        let cubin_path = std::env::temp_dir().join(format!("{unique}.cubin"));
        std::fs::write(&cu_path, cu_source).unwrap();

        let nvcc = if std::path::Path::new("/usr/local/cuda-13.2/bin/nvcc").exists() {
            "/usr/local/cuda-13.2/bin/nvcc"
        } else {
            "nvcc"
        };
        let arch = if (major, minor) == (10, 3) {
            format!("-arch=sm_{major}{minor}a")
        } else {
            format!("-arch=sm_{major}{minor}")
        };
        let compiled = std::process::Command::new(nvcc)
            .args([
                "-cubin",
                "-O3",
                &arch,
                cu_path.to_str().unwrap(),
                "-o",
                cubin_path.to_str().unwrap(),
            ])
            .output()
            .expect("failed to execute nvcc for CUDA test kernel");
        assert!(
            compiled.status.success(),
            "nvcc failed ({}): {}",
            compiled.status,
            String::from_utf8_lossy(&compiled.stderr)
        );
        let cubin_bytes = std::fs::read(&cubin_path).unwrap();
        let _ = std::fs::remove_file(&cu_path);
        let _ = std::fs::remove_file(&cubin_path);

        let encode_record = |msg_bytes: Vec<u8>, out: &mut Vec<u8>| {
            out.extend_from_slice(&(msg_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&msg_bytes);
        };
        let ptr_blob = |allocation: &str, offset: u64| wire::Blob {
            data: vec![0u8; 8],
            relocations: vec![wire::Relocation {
                offset: 0,
                target: Some(wire::Reference {
                    allocation: allocation.into(),
                    offset,
                }),
                kind: wire::RelocationKind::Pointer as i32,
            }],
        };

        let mut worker_records = Vec::new();
        encode_record(
            wire::Node {
                operation: Some(wire::node::Operation::Kernel(wire::Kernel {
                    function: 0,
                    grid: vec![1, 1, 1],
                    block: vec![32, 1, 1],
                    shared: 0,
                    arguments: vec![ptr_blob("worker_ctrl", 0), ptr_blob("worker_ctrl", 64)],
                    attributes: vec![],
                })),
            }
            .encode_to_vec(),
            &mut worker_records,
        );

        let step_val: i32 = 17;
        let mut coop_records = Vec::new();
        encode_record(
            wire::Node {
                operation: Some(wire::node::Operation::Memset(wire::Memset {
                    destination: Some(wire::Reference {
                        allocation: "compute_state".into(),
                        offset: 0,
                    }),
                    pitch: 4096,
                    value: 0,
                    element_size: 4,
                    width: 1024,
                    height: 1,
                })),
            }
            .encode_to_vec(),
            &mut coop_records,
        );
        encode_record(
            wire::Node {
                operation: Some(wire::node::Operation::Kernel(wire::Kernel {
                    function: 1,
                    // Intentionally request full-device SMs so prepare_graphs proves clamping
                    // down to the complementary compute green context's SM count.
                    grid: vec![total_sms as u32, 1, 1],
                    block: vec![256, 1, 1],
                    shared: one_block_smem,
                    arguments: vec![
                        ptr_blob("compute_state", 0),
                        wire::Blob {
                            data: step_val.to_le_bytes().to_vec(),
                            relocations: vec![],
                        },
                    ],
                    attributes: vec![wire::Attribute {
                        id: 2,
                        value: Some(wire::Blob {
                            data: {
                                let mut bytes = vec![0u8; 64];
                                bytes[0] = 1;
                                bytes
                            },
                            relocations: vec![],
                        }),
                    }],
                })),
            }
            .encode_to_vec(),
            &mut coop_records,
        );
        encode_record(
            wire::Edge {
                source: 0,
                destination: 1,
                data: vec![0; 8],
            }
            .encode_to_vec(),
            &mut coop_records,
        );

        let mut blobs = HashMap::new();
        blobs.insert("worker_test.cubin".to_string(), cubin_bytes);
        blobs.insert("graph_worker".to_string(), worker_records);
        blobs.insert("graph_coop".to_string(), coop_records);
        let files = TestFiles(blobs);

        let build_core = || Core {
            resources: HashMap::new(),
            architecture: [major, minor],
            allocations: vec![
                Allocation {
                    name: "worker_ctrl".into(),
                    nbytes: 2 << 20,
                    alignment: 2 << 20,
                    file: None,
                    data_spans: vec![],
                    sources: vec![],
                    relocations: vec![],
                    address: Some(host_va),
                    memory: Memory::MappedHost,
                    sizing: None,
                },
                Allocation {
                    name: "compute_state".into(),
                    nbytes: 2 << 20,
                    alignment: 2 << 20,
                    file: None,
                    data_spans: vec![],
                    sources: vec![],
                    relocations: vec![],
                    address: Some(device_va),
                    memory: Memory::Device,
                    sizing: None,
                },
            ],
            modules: vec![Module {
                name: "worker_mod".into(),
                file: "worker_test.cubin".into(),
                globals: vec![],
            }],
            functions: vec![
                Function {
                    id: 0,
                    module: "worker_mod".into(),
                    symbol: "worker_spin".into(),
                    parameter_sizes: vec![8, 8],
                    attributes: vec![],
                },
                Function {
                    id: 1,
                    module: "worker_mod".into(),
                    symbol: "coop_all_sms".into(),
                    parameter_sizes: vec![8, 4],
                    attributes: vec![],
                },
            ],
            graphs: vec![
                Graph {
                    name: "worker_spin".into(),
                    flags: 0,
                    resource: "graph_worker".into(),
                    node_count: 1,
                    edge_count: 0,
                    functions: vec![0],
                },
                Graph {
                    name: "coop_step".into(),
                    flags: 0,
                    resource: "graph_coop".into(),
                    node_count: 2,
                    edge_count: 1,
                    functions: vec![1],
                },
            ],
            reservations: vec![Reservation {
                address: device_va,
                nbytes: 64 << 20,
            }],
            execution: Some(wire::Execution {
                abi_version: 5,
                persistent_worker: Some(wire::PersistentWorker {
                    worker_sm_count: 8,
                    graph: "worker_spin".into(),
                    stop_flag_buffer: "worker_ctrl".into(),
                    stop_flag_offset: 0,
                    metrics_buffer: "worker_ctrl".into(),
                    metrics_offset: 64,
                    metrics_count: 2,
                }),
                ..Default::default()
            }),
            // Exercise PyTorch's cuCtxGetFlags() value (CU_CTX_MAP_HOST = 0x08), which
            // cuDevicePrimaryCtxSetFlags_v2 rejects unless masked.
            context: Context { flags: 0x08, ..Context::default() },
            events: vec![],
        };

        for round in 0..2 {
            let mut owner =
                unsafe { NativeResources::load_prepared(&files, build_core(), 0).unwrap() };
            let partition = owner.green_partition.as_ref().unwrap();
            let worker_sms = partition.worker_sm_count;
            let compute_sms = partition.compute_sm_count;
            assert!(worker_sms >= 8, "worker SM count {worker_sms}");
            assert_eq!(
                compute_sms,
                total_sms as u32 - worker_sms,
                "partition SMs must sum to total {total_sms}"
            );
            eprintln!(
                "round={round} total_sms={total_sms} worker_sms={worker_sms} compute_sms={compute_sms}"
            );

            assert_eq!(owner.persistent_worker_state(), Some(WorkerState::Idle));
            owner.start_persistent_worker().unwrap();
            owner.start_persistent_worker().unwrap();
            assert_eq!(owner.persistent_worker_state(), Some(WorkerState::Running));

            let deadline = Instant::now() + Duration::from_secs(2);
            let metrics_before = loop {
                let metrics = owner.persistent_worker_metrics().unwrap();
                if metrics[0] > 0 {
                    break metrics[0];
                }
                assert!(
                    Instant::now() < deadline,
                    "persistent worker did not increment mapped-host counter in time"
                );
                std::thread::yield_now();
            };

            assert!(
                owner.download("compute_state").is_err(),
                "primary context download must refuse while persistent worker is running"
            );

            let coop_idx = owner
                .names
                .iter()
                .position(|name| name == "coop_step")
                .unwrap();
            let coop_graph = owner.executables[coop_idx] as usize;
            let compute_addr = owner.allocation("compute_state").unwrap().address;
            // Retain Stream and PinnedHost until after stop_persistent_worker(): cuMemFreeHost
            // synchronizes device work and must run only after the persistent worker drains,
            // matching GraphExecutor::close().
            let stream = crate::cuda::Stream::open(owner.execution_context() as usize, 0).unwrap();
            let mut host_buf = stream.host_buffer(4096).unwrap();
            for iter in 0..20 {
                host_buf.bytes_mut().fill(0);
                stream.launch(coop_graph).unwrap();
                unsafe {
                    stream.download(&host_buf, compute_addr, 4096).unwrap();
                }
                stream.synchronize().unwrap();
                let words: Vec<i32> = host_buf
                    .bytes()
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                assert_eq!(
                    words[2], compute_sms as i32,
                    "round={round} iter={iter}: expected clamped cooperative grid {compute_sms}, got {}",
                    words[2]
                );
                for b in 0..compute_sms as usize {
                    let expected_mirror =
                        step_val * 1000 + (compute_sms as usize - 1 - b) as i32 + 7;
                    assert_eq!(
                        words[256 + b],
                        expected_mirror,
                        "round={round} iter={iter} cta={b}: barrier/mirror mismatch"
                    );
                }
            }

            let metrics_after_compute = owner.persistent_worker_metrics().unwrap();
            assert!(
                metrics_after_compute[0] > metrics_before,
                "worker counter did not advance during concurrent cooperative launches ({} <= {metrics_before})",
                metrics_after_compute[0]
            );

            owner.stop_persistent_worker().unwrap();
            owner.stop_persistent_worker().unwrap();
            assert_eq!(owner.persistent_worker_state(), Some(WorkerState::Stopped));
            let final_metrics = owner.persistent_worker_metrics().unwrap();
            assert_eq!(final_metrics[1], 1, "worker exit sentinel not set");

            drop(host_buf);
            drop(stream);

            let downloaded = owner.download("compute_state").unwrap();
            let observed_grid = i32::from_le_bytes(downloaded[8..12].try_into().unwrap());
            assert_eq!(observed_grid, compute_sms as i32);

            let _ = &mut owner;
            drop(owner);
        }
    }

    #[test]
    #[ignore = "requires an assigned CUDA_VISIBLE_DEVICES GPU supporting Green Contexts"]
    fn cuda_graph_executor_starts_worker_after_startup_tuning_and_drains_on_close() {
        if std::env::var_os("FLARE_RUN_CUDA_TESTS").is_none() {
            return;
        }
        use std::collections::HashMap;
        use std::process::Command;
        use std::time::{Duration, Instant};

        use prost::Message;

        use super::super::archive::PayloadFile;
        use super::super::driver::Driver;
        use super::super::format::{Allocation, Context, Core, Function, Graph, Memory, Module, Reservation};
        use super::super::payload::{Files, RecordReader};
        use super::super::protobuf::wire;
        use super::super::NativeResources;
        use crate::execution::GraphExecutor;

        struct TestFiles(HashMap<String, Vec<u8>>);
        impl Files for TestFiles {
            fn span<'a>(&self, _core: &'a Core, _key: &str) -> Result<&'a wire::Span> {
                Err(invalid("unused test span"))
            }
            fn source(&self, _span: &wire::Span) -> Result<(&PayloadFile, u64)> {
                Err(invalid("unused test source"))
            }
            fn read(&self, _core: &Core, key: &str) -> Result<Vec<u8>> {
                self.0
                    .get(key)
                    .cloned()
                    .ok_or_else(|| invalid(format!("missing test blob {key}")))
            }
            fn records<'a>(&'a self, _core: &Core, key: &str) -> Result<RecordReader<'a>> {
                let bytes = self
                    .0
                    .get(key)
                    .cloned()
                    .ok_or_else(|| invalid(format!("missing test record resource {key}")))?;
                Ok(RecordReader::Slice { bytes, position: 0 })
            }
            fn path(&self) -> &std::path::Path {
                std::path::Path::new("")
            }
            fn source_index(&self, _span: &wire::Span) -> Result<usize> {
                Err(invalid("unused test source_index"))
            }
            fn asset_path(&self, _span: &wire::Span) -> Result<std::path::PathBuf> {
                Err(invalid("unused test asset_path"))
            }
            fn release_page_cache(&self) {}
            fn retained_files(&self) -> Result<Vec<std::sync::Arc<std::fs::File>>> {
                Ok(Vec::new())
            }
            fn direct_files(
                &self,
            ) -> Box<dyn Iterator<Item = Result<std::sync::Arc<std::fs::File>>> + '_> {
                Box::new(std::iter::empty())
            }
            fn validate(&self, _core: &Core) -> Result<()> {
                Ok(())
            }
            fn resource_reader<'a>(
                &'a self,
                _core: &Core,
                _key: &str,
            ) -> Result<Box<dyn std::io::Read + 'a>> {
                Err(invalid("unused test resource_reader"))
            }
        }

        let driver = Driver::load().unwrap();
        let (major, minor, total_sms) = unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            let mut device = 0;
            driver.check("cuDeviceGet", (driver.device)(&mut device, 0)).unwrap();
            let attr = |code: i32| {
                let mut val = 0;
                driver
                    .check("cuDeviceGetAttribute", (driver.device_attribute)(&mut val, code, device))
                    .unwrap();
                val
            };
            (attr(75), attr(76), attr(16))
        };
        assert!(driver.green_ctx.is_some());
        assert!(total_sms >= 16);

        let cu_source = r#"
extern "C" __global__ void worker_spin(const unsigned int* stop_flag, unsigned long long* metrics) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long iters = 0;
        while (true) {
            unsigned int stop = 0;
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(stop) : "l"(stop_flag) : "memory");
            if (stop != 0u) break;
            ++iters;
            asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(metrics), "l"(iters) : "memory");
            asm volatile("nanosleep.u32 500;");
        }
        asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(metrics + 1), "l"(1ULL) : "memory");
    }
}

extern "C" __global__ void tuning_verify_worker_idle(const unsigned long long* metrics, int* out_val, int* status) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long m = 0;
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(m) : "l"(metrics) : "memory");
        *out_val = 42;
        *status = (m == 0ULL) ? 0 : 1;
    }
}

extern "C" __global__ void step_emit_completion(
    const int* frame_index,
    const long long* submission_seq,
    const int* tuned_val,
    long long* completion_records
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int frame = frame_index[0];
        const long long seq = submission_seq[0];
        const long long tok = static_cast<long long>(*tuned_val);
        // Completion record stride for completion_token_capacity=8 is 24 i64 words per row (8 base + 8 vector hdr + 8 tokens).
        long long* row = completion_records + frame * 24;
        row[0] = 1;   // generation
        row[1] = seq; // submission_sequence
        row[2] = 1;   // consumed_tokens
        row[3] = 1;   // generated_tokens
        row[4] = tok; // token_id
        row[5] = 1;   // status = active
        row[6] = 0;   // error_code
        row[7] = 0;   // reserved
        row[8] = 1;   // emitted_count
        row[9] = 0;   // offered_drafts
        row[10] = 0;  // accepted_drafts
        row[11] = 0;  // copy_offered_mask
        row[12] = 0;  // model_reserved
        row[13] = 0;  // draft_cache_consumed
        row[14] = 0;
        row[15] = 1;  // verification_width
        row[16] = tok;
        for (int i = 17; i < 24; ++i) {
            row[i] = -1;
        }
    }
}
"#;
        let stamp = format!(
            "flare_pw_exec_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(stamp);
        std::fs::create_dir_all(&dir).unwrap();
        let cu_path = dir.join("exec_test.cu");
        let cubin_path = dir.join("exec_test.cubin");
        std::fs::write(&cu_path, cu_source).unwrap();
        let nvcc = if std::path::Path::new("/usr/local/cuda-13.2/bin/nvcc").exists() {
            "/usr/local/cuda-13.2/bin/nvcc"
        } else if std::path::Path::new("/usr/local/cuda/bin/nvcc").exists() {
            "/usr/local/cuda/bin/nvcc"
        } else {
            "nvcc"
        };
        let arch = if (major, minor) == (10, 3) {
            format!("-arch=sm_{major}{minor}a")
        } else {
            format!("-arch=sm_{major}{minor}")
        };
        let output = Command::new(nvcc)
            .args([
                "-cubin",
                "-O3",
                &arch,
                cu_path.to_str().unwrap(),
                "-o",
                cubin_path.to_str().unwrap(),
            ])
            .output()
            .expect("nvcc must be installed to compile CUDA test kernel");
        assert!(
            output.status.success(),
            "nvcc failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cubin_bytes = std::fs::read(&cubin_path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        fn encode_record(bytes: Vec<u8>, out: &mut Vec<u8>) {
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        fn ptr_blob(allocation: &str, offset: u64) -> wire::Blob {
            wire::Blob {
                data: vec![0u8; 8],
                relocations: vec![wire::Relocation {
                    offset: 0,
                    target: Some(wire::Reference {
                        allocation: allocation.into(),
                        offset,
                    }),
                    kind: 0,
                }],
            }
        }

        let mut worker_records = Vec::new();
        encode_record(
            wire::Node {
                operation: Some(wire::node::Operation::Kernel(wire::Kernel {
                    function: 0,
                    grid: vec![1, 1, 1],
                    block: vec![32, 1, 1],
                    shared: 0,
                    arguments: vec![ptr_blob("worker_ctrl", 0), ptr_blob("worker_ctrl", 64)],
                    attributes: vec![],
                })),
            }
            .encode_to_vec(),
            &mut worker_records,
        );

        let mut tuning_records = Vec::new();
        encode_record(
            wire::Node {
                operation: Some(wire::node::Operation::Kernel(wire::Kernel {
                    function: 1,
                    grid: vec![1, 1, 1],
                    block: vec![32, 1, 1],
                    shared: 0,
                    arguments: vec![
                        ptr_blob("worker_ctrl", 64),
                        ptr_blob("io_buf", 512),
                        ptr_blob("io_buf", 768),
                    ],
                    attributes: vec![],
                })),
            }
            .encode_to_vec(),
            &mut tuning_records,
        );

        let mut decode_records = Vec::new();
        encode_record(
            wire::Node {
                operation: Some(wire::node::Operation::Kernel(wire::Kernel {
                    function: 2,
                    grid: vec![1, 1, 1],
                    block: vec![32, 1, 1],
                    shared: 0,
                    arguments: vec![
                        ptr_blob("io_buf", 0),
                        ptr_blob("io_buf", 64),
                        ptr_blob("io_buf", 512),
                        ptr_blob("io_buf", 1024),
                    ],
                    attributes: vec![],
                })),
            }
            .encode_to_vec(),
            &mut decode_records,
        );

        let mut blobs = HashMap::new();
        blobs.insert("exec_test.cubin".to_string(), cubin_bytes);
        blobs.insert("g_worker".to_string(), worker_records);
        blobs.insert("g_tune".to_string(), tuning_records);
        blobs.insert("g_decode".to_string(), decode_records);
        let files = TestFiles(blobs);

        let device_va = 0x840_0000_0000u64;
        let host_va = 0xc40_0000_0000u64;
        let buf = |name: &str, alloc: &str, offset: u64, nbytes: u64, dtype: &str, shape: Vec<u64>, strides: Vec<u64>| wire::Buffer {
            name: name.into(),
            reference: Some(wire::Reference { allocation: alloc.into(), offset }),
            nbytes,
            dtype: dtype.into(),
            shape,
            strides,
        };

        let core = Core {
            resources: HashMap::new(),
            architecture: [major, minor],
            allocations: vec![
                Allocation {
                    name: "worker_ctrl".into(),
                    nbytes: 2 << 20,
                    alignment: 2 << 20,
                    file: None,
                    data_spans: vec![],
                    sources: vec![],
                    relocations: vec![],
                    address: Some(host_va),
                    memory: Memory::MappedHost,
                    sizing: None,
                },
                Allocation {
                    name: "io_buf".into(),
                    nbytes: 2 << 20,
                    alignment: 2 << 20,
                    file: None,
                    data_spans: vec![],
                    sources: vec![],
                    relocations: vec![],
                    address: Some(device_va),
                    memory: Memory::Device,
                    sizing: None,
                },
            ],
            modules: vec![Module {
                name: "exec_mod".into(),
                file: "exec_test.cubin".into(),
                globals: vec![],
            }],
            functions: vec![
                Function { id: 0, module: "exec_mod".into(), symbol: "worker_spin".into(), parameter_sizes: vec![8, 8], attributes: vec![] },
                Function { id: 1, module: "exec_mod".into(), symbol: "tuning_verify_worker_idle".into(), parameter_sizes: vec![8, 8, 8], attributes: vec![] },
                Function { id: 2, module: "exec_mod".into(), symbol: "step_emit_completion".into(), parameter_sizes: vec![8, 8, 8, 8], attributes: vec![] },
            ],
            graphs: vec![
                Graph { name: "worker_spin".into(), flags: 0, resource: "g_worker".into(), node_count: 1, edge_count: 0, functions: vec![0] },
                Graph { name: "startup_tune".into(), flags: 0, resource: "g_tune".into(), node_count: 1, edge_count: 0, functions: vec![1] },
                Graph { name: "decode_b1".into(), flags: 0, resource: "g_decode".into(), node_count: 1, edge_count: 0, functions: vec![2] },
            ],
            reservations: vec![Reservation { address: device_va, nbytes: 64 << 20 }],
            execution: Some(wire::Execution {
                abi_version: 5,
                max_active: 1,
                input_frame_bytes: 64,
                completion_token_capacity: 8,
                buffers: vec![
                    buf("frame_index", "io_buf", 0, 4, "int32", vec![1], vec![1]),
                    buf("submission_sequence", "io_buf", 64, 8, "int64", vec![1], vec![1]),
                    buf("tuned_val", "io_buf", 512, 4, "int32", vec![1], vec![1]),
                    buf("tuning_status", "io_buf", 768, 4, "int32", vec![1], vec![1]),
                    buf("completion_records", "io_buf", 1024, 2 * 24 * 8, "int64", vec![2, 1, 24], vec![24, 24, 1]),
                    buf("worker_ctrl", "worker_ctrl", 0, 128, "uint8", vec![128], vec![1]),
                ],
                graphs: vec![wire::ExecutionGraph {
                    name: "decode_b1".into(),
                    kind: "decode".into(),
                    sampling_mode: "general".into(),
                    batch_capacity: 1,
                    token_capacity: 1,
                    prefill_capacity: 0,
                    estimated_duration_ns: 1000,
                    input_bindings: vec![
                        wire::InputBinding { buffer_name: "frame_index".into(), host_offset: 0, nbytes: 4 },
                        wire::InputBinding { buffer_name: "submission_sequence".into(), host_offset: 8, nbytes: 8 },
                    ],
                    tuned_bindings: vec![wire::TunedBinding {
                        buffer: "tuned_val".into(),
                        tuning_graph: "startup_tune".into(),
                    }],
                    ..Default::default()
                }],
                tuning_graphs: vec![wire::TuningGraph {
                    name: "startup_tune".into(),
                    outputs: vec!["tuned_val".into()],
                    status_buffer: "tuning_status".into(),
                }],
                persistent_worker: Some(wire::PersistentWorker {
                    worker_sm_count: 8,
                    graph: "worker_spin".into(),
                    stop_flag_buffer: "worker_ctrl".into(),
                    stop_flag_offset: 0,
                    metrics_buffer: "worker_ctrl".into(),
                    metrics_offset: 64,
                    metrics_count: 2,
                }),
                ..Default::default()
            }),
            // Exercise PyTorch's cuCtxGetFlags() value (CU_CTX_MAP_HOST = 0x08), which
            // cuDevicePrimaryCtxSetFlags_v2 rejects unless masked.
            context: Context { flags: 0x08, ..Context::default() },
            events: vec![],
        };

        let owner = unsafe { NativeResources::load_prepared(&files, core, 0).unwrap() };
        assert_eq!(owner.persistent_worker_state(), Some(WorkerState::Idle));

        let manifest = owner.execution_manifest().unwrap();
        let mut executor = unsafe { GraphExecutor::new(manifest).unwrap() };
        // Startup tuning graph verified metrics[0] == 0 before finish_startup() launched the worker.
        assert_eq!(owner.persistent_worker_state(), Some(WorkerState::Running));

        let deadline = Instant::now() + Duration::from_secs(2);
        let metrics_before = loop {
            let m = owner.persistent_worker_metrics().unwrap();
            if m[0] > 0 {
                break m[0];
            }
            assert!(Instant::now() < deadline, "worker counter did not advance after GraphExecutor::new");
            std::thread::yield_now();
        };

        for step in 0..5u64 {
            let fence = executor.submit(0, &[], false).unwrap().expect("expected submission fence");
            let poll_deadline = Instant::now() + Duration::from_secs(2);
            let done = loop {
                if let Some(c) = executor.poll().unwrap() {
                    break c;
                }
                assert!(Instant::now() < poll_deadline, "GraphExecutor step {step} timed out");
                std::thread::yield_now();
            };
            assert_eq!(done.fence, fence);
            assert_eq!(done.records[0].token_id, 42, "startup tuning output must be visible to serving graph");
        }

        let metrics_after = owner.persistent_worker_metrics().unwrap();
        assert!(
            metrics_after[0] > metrics_before,
            "worker did not continue running during GraphExecutor submits ({} <= {metrics_before})",
            metrics_after[0]
        );

        let shutdown = executor.close();
        assert!(shutdown.drained, "GraphExecutor::close must drain streams and persistent worker");
        assert!(shutdown.error.is_none(), "unexpected shutdown error: {:?}", shutdown.error);
        assert_eq!(owner.persistent_worker_state(), Some(WorkerState::Stopped));
        assert_eq!(owner.persistent_worker_metrics().unwrap()[1], 1, "worker exit sentinel not set");
        drop(executor);
        drop(owner);
    }
}
