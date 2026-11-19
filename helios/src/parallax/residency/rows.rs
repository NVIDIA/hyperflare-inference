//! Copy complete expert rows into spare HBM storage, then publish canonical row IDs.
//! Weight-plane bases and the mapping buffer stay fixed across graph replay.

use super::*;
use crate::{
    cuda,
    runtime_core::driver::{Driver, Handle},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem::ManuallyDrop,
    sync::Arc,
};

mod layout;
pub mod package;
use layout::Layout;
pub use layout::{Layer, Plane};

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

// The writer returns a recoverable error only after draining every upload.
// Restore all changed maps, including a partially written failing map.
fn replace_maps(
    before: &[Vec<i32>],
    after: &[Vec<i32>],
    changed: &BTreeSet<usize>,
    mut write: impl FnMut(&[Vec<i32>], &BTreeSet<usize>) -> Result<()>,
) -> Result<()> {
    match write(after, changed) {
        Ok(()) => Ok(()),
        Err(error @ Error::Poisoned(_)) => Err(error),
        Err(error) => match write(before, changed) {
            Ok(()) => Err(Error::Backend(format!(
                "expert map publication rolled back: {error}"
            ))),
            Err(rollback) => Err(Error::Poisoned(format!(
                "expert map publication failed ({error}); rollback failed ({rollback})"
            ))),
        },
    }
}

pub struct Rows {
    layout: Layout,
    locations: Vec<Location>,
    staged: BTreeMap<usize, usize>,
    active: bool,
    poisoned: bool,
    context: usize,
    driver: Arc<Driver>,
    stream: ManuallyDrop<cuda::Stream>,
    started: ManuallyDrop<cuda::Event>,
    done: ManuallyDrop<cuda::Event>,
    map_upload: ManuallyDrop<cuda::PinnedHost>,
}
impl Rows {
    /// # Safety
    /// All addresses are validated device-accessible ranges in this context.
    /// The ranges do not physically alias. They retain their layout and owners
    /// until this backend and every reader finish. Host rows stay immutable.
    /// Every row contains the declared expert's original representation; duplicate
    /// tier copies agree bytewise. No other writer changes HBM rows or mappings.
    /// A poisoned backend requires retaining the storage and context.
    /// Construction occurs after weight loading and before graph submissions.
    pub unsafe fn new(context: usize, device: i32, layers: Vec<Layer>) -> Result<Self> {
        let layout = Layout::new(layers)?;
        let driver = Arc::new(Driver::load().map_err(|e| Error::Backend(e.to_string()))?);
        let stream = cuda::Stream::open_lowest_priority(context, device)
            .map_err(|e| Error::Backend(e.to_string()))?;
        for (layer, expected) in layout.layers.iter().zip(layout.maps(&layout.initial)?) {
            let mut actual = vec![0i32; layer.experts];
            driver
                .check("cuMemcpyDtoH", unsafe {
                    (driver.download)(
                        actual.as_mut_ptr().cast(),
                        layer.mapping_address,
                        actual.len() * 4,
                    )
                })
                .map_err(|e| Error::Backend(e.to_string()))?;
            if actual != expected {
                return Err(invalid(
                    "prepared expert mapping disagrees with initial physical rows",
                ));
            }
        }
        let started = stream
            .event_with_timing(true)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let done = stream
            .event_with_timing(true)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let map_upload = stream
            .host_buffer(layout.map_bytes)
            .map_err(|e| Error::Backend(e.to_string()))?;
        Ok(Self {
            locations: layout.initial.clone(),
            layout,
            staged: BTreeMap::new(),
            active: false,
            poisoned: false,
            context,
            driver,
            stream: ManuallyDrop::new(stream),
            started: ManuallyDrop::new(started),
            done: ManuallyDrop::new(done),
            map_upload: ManuallyDrop::new(map_upload),
        })
    }
    fn bind(&self) -> Result<()> {
        if self.poisoned {
            return Err(Error::Poisoned(
                "expert row storage has unconfirmed CUDA work".into(),
            ));
        }
        self.driver
            .check("cuCtxSetCurrent", unsafe {
                (self.driver.context_set)(self.context as Handle)
            })
            .map_err(|e| Error::Backend(e.to_string()))
    }
    fn failed_transfer(&mut self, error: impl std::fmt::Display) -> Error {
        if let Err(drain) = self.stream.synchronize() {
            self.poisoned = true;
            return Error::Poisoned(format!(
                "expert row transfer failed ({error}); drain failed ({drain})"
            ));
        }
        self.active = false;
        self.staged.clear();
        Error::Backend(error.to_string())
    }
    fn upload_maps(&mut self, maps: &[Vec<i32>], changed: &BTreeSet<usize>) -> Result<()> {
        for &index in changed {
            let offset = self.layout.map_offsets[index];
            for (i, value) in maps[index].iter().enumerate() {
                self.map_upload.bytes_mut()[offset + i * 4..offset + (i + 1) * 4]
                    .copy_from_slice(&value.to_ne_bytes());
            }
        }
        for &index in changed {
            if let Err(error) = unsafe {
                self.stream.upload(
                    self.layout.layers[index].mapping_address as usize,
                    &self.map_upload,
                    self.layout.map_offsets[index],
                    maps[index].len() * 4,
                )
            } {
                return Err(self.failed_transfer(error));
            }
        }
        self.stream
            .synchronize()
            .map_err(|e| Error::Poisoned(format!("expert map upload drain failed: {e}")))
    }
}

// The storage owner supplies the prepared ranges; publication runs behind the
// executor's reader/submission gate. Copies touch only previously unused HBM rows.
unsafe impl Backend for Rows {
    fn regions(&self) -> &[Region] {
        &self.layout.regions
    }
    fn initial_location(&self, region: usize) -> Location {
        self.layout.initial[region]
    }
    fn has_backing(&self, region: usize) -> bool {
        self.layout.backing[region].is_some()
    }
    fn slots(&self) -> usize {
        self.layout.slots.len()
    }
    fn slot_bytes(&self) -> usize {
        self.layout.slots.iter().map(|s| s.bytes).max().unwrap()
    }
    fn slot_capacity(&self, slot: usize) -> usize {
        self.layout.slots[slot].bytes
    }
    fn slot_accepts(&self, region: usize, slot: usize) -> bool {
        self.layout.accepts(region, slot)
    }
    fn backing_bytes(&self) -> usize {
        self.layout.backing_bytes
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            stable_addresses: false,
            asynchronous_staging: true,
            publication_requires_quiescence: true,
            cold_moe_execution_qualified: false,
        }
    }
    fn address(&self, region: usize) -> u64 {
        match self.locations[region] {
            Location::Backing => self.layout.backing[region].unwrap(),
            Location::Hbm(slot) => self.layout.slots[slot].address,
        }
    }
    fn stage(&mut self, copies: &[(usize, usize)]) -> Result<()> {
        self.bind()?;
        if self.active {
            return Err(Error::Busy);
        }
        let mut final_locations = self.locations.clone();
        let mut unique = BTreeSet::new();
        for &(region, slot) in copies {
            if self.locations.get(region) != Some(&Location::Backing)
                || !self.layout.accepts(region, slot)
                || !unique.insert(region)
                || self.locations.contains(&Location::Hbm(slot))
            {
                return Err(invalid(
                    "row staging would overwrite live or duplicate storage",
                ));
            }
            final_locations[region] = Location::Hbm(slot);
        }
        self.layout.maps(&final_locations)?;
        if let Err(e) = self.stream.record(&self.started) {
            return Err(self.failed_transfer(e));
        }
        for &(region, slot) in copies {
            let bytes = self.layout.regions[region].bytes;
            for offset in (0..bytes).step_by(cuda::BATCH_BYTES) {
                let n = (bytes - offset).min(cuda::BATCH_BYTES);
                if let Err(e) = unsafe {
                    self.stream.copy_batch(
                        &[self.layout.slots[slot].address + offset as u64],
                        &[self.layout.backing[region].unwrap() + offset as u64],
                        &[n],
                    )
                } {
                    return Err(self.failed_transfer(e));
                }
            }
        }
        if let Err(e) = self.stream.record(&self.done) {
            return Err(self.failed_transfer(e));
        }
        self.staged = copies.iter().copied().collect();
        self.active = true;
        Ok(())
    }
    fn ready(&mut self) -> Result<bool> {
        self.bind()?;
        if !self.active {
            return Ok(true);
        }
        self.done.completed().map_err(|e| self.failed_transfer(e))
    }
    fn copy_elapsed_ns(&self) -> Result<Option<u64>> {
        self.done
            .elapsed_ns(&self.started)
            .map(Some)
            .map_err(|e| Error::Backend(e.to_string()))
    }
    fn discard(&mut self) -> Result<()> {
        if !self.ready()? {
            return Err(Error::Busy);
        }
        self.staged.clear();
        self.active = false;
        Ok(())
    }
    unsafe fn publish(&mut self, bindings: &[Binding]) -> Result<()> {
        if !self.ready()? {
            return Err(Error::Busy);
        }
        let mut locations = self.locations.clone();
        let mut unique = BTreeSet::new();
        for b in bindings {
            if self.locations.get(b.region) != Some(&b.from) || !unique.insert(b.region) {
                return Err(invalid(
                    "row publication disagrees with authoritative locations",
                ));
            }
            if let Location::Hbm(slot) = b.to {
                if self.staged.get(&b.region) != Some(&slot) {
                    return Err(invalid("row publication has no completed staged copy"));
                }
            }
            locations[b.region] = b.to;
        }
        let before = self.layout.maps(&self.locations)?;
        let after = self.layout.maps(&locations)?;
        let changed = (0..before.len())
            .filter(|i| before[*i] != after[*i])
            .collect();
        if let Err(error) = replace_maps(&before, &after, &changed, |maps, changed| {
            self.upload_maps(maps, changed)
        }) {
            self.poisoned |= matches!(error, Error::Poisoned(_));
            return Err(error);
        }
        self.locations = locations;
        self.staged.clear();
        self.active = false;
        Ok(())
    }
}

impl Drop for Rows {
    fn drop(&mut self) {
        if self.bind().is_err() || self.stream.synchronize().is_err() {
            std::mem::forget(self.driver.clone());
            return;
        }
        unsafe {
            ManuallyDrop::drop(&mut self.map_upload);
            ManuallyDrop::drop(&mut self.done);
            ManuallyDrop::drop(&mut self.started);
            ManuallyDrop::drop(&mut self.stream);
        }
    }
}

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod tests;
