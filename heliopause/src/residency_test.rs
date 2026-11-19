//! Qualification-only access to native storage for captured-kernel parity tests.
//! Python never controls residency in the serving runtime.
use crate::{
    lifetime::OwnerLease,
    parallax::{
        ExpertId,
        residency::{
            Controller, Plan, Progress, Region,
            gb300::{Gb300, Limits},
            rows::{Layer, Plane, Rows},
        },
    },
    runtime_core::driver::{Driver, Handle},
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::time::{Duration, Instant};

fn error(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Own native Grace/HBM mappings while a qualification test captures kernels.
/// The owner retains source tensors and the current context. Callers must exclude
/// concurrent graph submission during exchange/close and retain this object while
/// any tensor or graph can reference its addresses. Views must not write weights.
#[pyclass(module = "flare_heliopause", unsendable)]
pub(crate) struct ExpertResidencyTestStorage {
    controller: Option<Box<dyn TestController>>,
    driver: Driver,
    context: usize,
    owner: OwnerLease<Py<PyAny>>,
}

#[pymethods]
impl ExpertResidencyTestStorage {
    #[new]
    fn new(
        py: Python<'_>,
        owner: Py<PyAny>,
        regions: Vec<(usize, usize, usize, usize, u64, usize)>,
        hbm_slots: usize,
        hbm_bytes: usize,
        backing_bytes: usize,
    ) -> PyResult<Self> {
        if owner.bind(py).is_none() || regions.is_empty() {
            return Err(PyValueError::new_err(
                "retain the source tensors and CUDA context in owner",
            ));
        }
        let (context, device) = crate::current_context().map_err(error)?;
        let driver = Driver::load().map_err(error)?;
        driver
            .check("cuCtxSynchronize", unsafe { (driver.context_sync)() })
            .map_err(error)?;
        let mut sources = std::collections::BTreeMap::new();
        let mut geometry = Vec::with_capacity(regions.len());
        for (layer, expert, part, offset, source, bytes) in regions {
            let expert = ExpertId { layer, expert };
            if source == 0 || bytes == 0 || sources.insert((expert, part), source).is_some() {
                return Err(PyValueError::new_err("invalid or duplicate source region"));
            }
            geometry.push(Region {
                expert,
                part,
                offset,
                bytes,
            });
        }
        // SAFETY: The test owns the current context and immutable source tensors.
        // OwnerLease retains their Python owner until this storage drains.
        let backend = unsafe {
            Gb300::new(
                context,
                device,
                geometry,
                Limits {
                    hbm_slots,
                    hbm_bytes,
                    backing_bytes,
                    virtual_base: None,
                },
                |region, output| {
                    driver
                        .check(
                            "cuMemcpyDtoH",
                            (driver.download)(
                                output.as_mut_ptr().cast(),
                                sources[&(region.expert, region.part)],
                                output.len(),
                            ),
                        )
                        .map_err(|e| crate::parallax::residency::Error::Backend(e.to_string()))
                },
            )
        }
        .map_err(error)?;
        Self::attach(
            Controller::new("kernel-qualification".into(), backend).map_err(error)?,
            driver,
            context,
            owner,
        )
    }

    /// Borrow prepared canonical maps and weight planes for kernel qualification.
    /// Each layer supplies its identity, expert count, tier IDs, map address,
    /// and (host address, HBM address, row bytes) for every weight plane.
    #[staticmethod]
    fn from_rows(
        py: Python<'_>,
        owner: Py<PyAny>,
        layers: Vec<(
            usize,
            usize,
            Vec<usize>,
            Vec<usize>,
            u64,
            Vec<(u64, u64, usize)>,
        )>,
    ) -> PyResult<Self> {
        if owner.bind(py).is_none() || layers.is_empty() {
            return Err(PyValueError::new_err(
                "retain the weight planes, maps and context in owner",
            ));
        }
        let (context, device) = crate::current_context().map_err(error)?;
        let driver = Driver::load().map_err(error)?;
        driver
            .check("cuCtxSynchronize", unsafe { (driver.context_sync)() })
            .map_err(error)?;
        let layers = layers
            .into_iter()
            .map(
                |(layer, experts, host_experts, hbm_experts, mapping_address, planes)| Layer {
                    layer,
                    experts,
                    host_experts,
                    hbm_experts,
                    mapping_address,
                    planes: planes
                        .into_iter()
                        .map(|(host_address, hbm_address, row_bytes)| Plane {
                            host_address,
                            hbm_address,
                            row_bytes,
                        })
                        .collect(),
                },
            )
            .collect();
        // SAFETY: This qualification API requires exclusive test ownership of
        // these prepared arrays. OwnerLease retains them through confirmed drain.
        let backend = unsafe { Rows::new(context, device, layers) }.map_err(error)?;
        Self::attach(
            Controller::new("kernel-qualification".into(), backend).map_err(error)?,
            driver,
            context,
            owner,
        )
    }

    fn address(&self, layer: usize, expert: usize, part: usize) -> PyResult<u64> {
        self.controller
            .as_ref()
            .ok_or_else(|| error("storage is closed"))?
            .address(ExpertId { layer, expert }, part)
            .map_err(error)
    }

    /// Drain the test context, then stage and publish a complete transaction.
    /// Returns elapsed wall time for diagnostic accounting, not serving speedup.
    fn exchange(
        &mut self,
        promote: Vec<(usize, usize)>,
        evict: Vec<(usize, usize)>,
    ) -> PyResult<u64> {
        self.drain()?;
        self.controller
            .as_mut()
            .ok_or_else(|| error("storage is closed"))?
            .exchange(promote, evict)
    }

    fn close(&mut self) -> PyResult<()> {
        self.drain()?;
        if let Some(controller) = &mut self.controller {
            controller.closeable()?;
        }
        drop(self.controller.take());
        drop(self.owner.take_after_drain());
        Ok(())
    }
}
impl ExpertResidencyTestStorage {
    fn attach<B: crate::parallax::residency::Backend + Send + 'static>(
        controller: Controller<B>,
        driver: Driver,
        context: usize,
        owner: Py<PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            controller: Some(Box::new(controller)),
            driver,
            context,
            owner: OwnerLease(Some(owner)),
        })
    }

    fn drain(&self) -> PyResult<()> {
        self.driver
            .check("cuCtxSetCurrent", unsafe {
                (self.driver.context_set)(self.context as Handle)
            })
            .map_err(error)?;
        self.driver
            .check("cuCtxSynchronize", unsafe { (self.driver.context_sync)() })
            .map_err(error)
    }
}
impl Drop for ExpertResidencyTestStorage {
    fn drop(&mut self) {
        if self.controller.is_none() {
            return;
        }
        if self.close().is_err() {
            if let Some(controller) = self.controller.take() {
                std::mem::forget(controller);
            }
            // OwnerLease retains Python resources when drain or rollback is unconfirmed.
        }
    }
}

// Qualification retains one ownership/drain path for VMM and row controllers.
trait TestController {
    fn address(&self, expert: ExpertId, part: usize) -> crate::parallax::residency::Result<u64>;
    fn exchange(
        &mut self,
        promote: Vec<(usize, usize)>,
        evict: Vec<(usize, usize)>,
    ) -> PyResult<u64>;
    fn closeable(&mut self) -> PyResult<()>;
}
impl<B: crate::parallax::residency::Backend> TestController for Controller<B> {
    fn address(&self, expert: ExpertId, part: usize) -> crate::parallax::residency::Result<u64> {
        self.address(expert, part)
    }
    fn exchange(
        &mut self,
        promote: Vec<(usize, usize)>,
        evict: Vec<(usize, usize)>,
    ) -> PyResult<u64> {
        let start = Instant::now();
        self.begin(
            Plan {
                version: 1,
                catalog_id: "kernel-qualification".into(),
                generation: self.snapshot().generation,
                valid_until_ns: u64::MAX,
                promote: promote
                    .into_iter()
                    .map(|(layer, expert)| ExpertId { layer, expert })
                    .collect(),
                evict: evict
                    .into_iter()
                    .map(|(layer, expert)| ExpertId { layer, expert })
                    .collect(),
            },
            0,
        )
        .map_err(error)?;
        loop {
            // SAFETY: The qualification caller drains the sole submitting context
            // and excludes graph submissions throughout this transaction.
            match unsafe { self.progress(0) }.map_err(error)? {
                Progress::Noop | Progress::Committed(_) => break,
                Progress::WaitingForTransfer if start.elapsed() < Duration::from_secs(10) => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                state => {
                    return Err(error(format!(
                        "residency test transaction did not finish: {state:?}"
                    )));
                }
            }
        }
        Ok(start.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }
    fn closeable(&mut self) -> PyResult<()> {
        let state = self.snapshot();
        if state.poisoned || state.readers != 0 {
            return Err(error(
                "expert test storage has unconfirmed readers or mappings",
            ));
        }
        if self.cancel().map_err(error)? == Progress::WaitingForTransfer {
            return Err(error("expert test storage still has an active transfer"));
        }
        Ok(())
    }
}
