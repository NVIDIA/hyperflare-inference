//! Bind authenticated expert storage before native graph submission.
use super::{
    NativeResources, Storage,
    format::{Core, Memory, invalid},
};
use crate::{
    Result,
    parallax::{
        ExpertId,
        residency::{
            Controller, Plan, Region,
            execution::ExecutionBinding,
            gb300::{Gb300, Limits},
            package::{BoundRegion, StorageConfig},
        },
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type ResidencyOptions = VirtualMemoryOptions;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VirtualMemoryOptions {
    pub version: u32,
    pub storage: StorageConfig,
    pub catalog_id: String,
    pub canonical: Vec<ExpertId>,
    pub hbm_slots: usize,
    pub hbm_bytes: usize,
    pub backing_bytes: usize,
    pub initial_hbm: Vec<ExpertId>,
}

impl VirtualMemoryOptions {
    pub fn catalog_id(&self) -> &str {
        &self.catalog_id
    }

    pub fn canonical(&self) -> &[ExpertId] {
        &self.canonical
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.version != 1
            || self.catalog_id.trim().is_empty()
            || self.hbm_slots == 0
            || self.hbm_bytes == 0
            || self.backing_bytes == 0
        {
            return Err(invalid("invalid physical residency configuration"));
        }
        let canonical: BTreeSet<_> = self.canonical.iter().copied().collect();
        let initial: BTreeSet<_> = self.initial_hbm.iter().copied().collect();
        if canonical.is_empty()
            || canonical.len() != self.canonical.len()
            || initial.len() != self.initial_hbm.len()
            || !initial.is_subset(&canonical)
        {
            return Err(invalid(
                "initial residency requires unique canonical experts",
            ));
        }
        Ok(())
    }

    pub(super) fn resolve(
        &self,
        core: &crate::core::protection::VerifiedCore,
    ) -> Result<ResolvedResidency> {
        self.validate()?;
        let regions = self
            .storage
            .resolve(core, &self.canonical, 1)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(ResolvedResidency {
            options: self.clone(),
            regions,
        })
    }
}

pub(super) struct ResolvedResidency {
    options: VirtualMemoryOptions,
    regions: Vec<BoundRegion>,
}

impl ResolvedResidency {
    pub(super) unsafe fn plan(&self, owner: &mut NativeResources, core: &Core) -> Result<()> {
        unsafe { owner.plan_residency(core, &self.options, &self.regions) }
    }

    pub(super) unsafe fn prepare(&self, owner: &mut NativeResources, core: &Core) -> Result<()> {
        unsafe { owner.prepare_residency(core, &self.options, &self.regions) }
    }

    pub(super) unsafe fn finish(self, owner: &mut NativeResources) -> Result<()> {
        unsafe { owner.finish_residency(&self.options) }
    }
}

/// CUDA mappings are indivisible; this loader admits only entire immutable allocations.
fn allocations(package: &Core, regions: &[BoundRegion]) -> Result<Vec<usize>> {
    let mut groups = BTreeMap::<usize, Vec<&BoundRegion>>::new();
    for region in regions {
        groups.entry(region.allocation).or_default().push(region);
    }
    if groups.is_empty() {
        return Err(invalid("expert storage is empty"));
    }
    for (&index, parts) in &mut groups {
        let allocation = package
            .allocations
            .get(index)
            .ok_or_else(|| invalid("expert allocation index is unavailable"))?;
        if !matches!(allocation.memory, Memory::Device)
            || allocation.sizing.is_some()
            || allocation.address.is_none()
            || !allocation.relocations.is_empty()
        {
            return Err(invalid(
                "expert storage requires immutable fixed device allocations",
            ));
        }
        parts.sort_unstable_by_key(|r| r.offset);
        let mut end = 0usize;
        for region in parts {
            if region.offset != end
                || allocation.address.unwrap().checked_add(end as u64) != Some(region.address)
            {
                return Err(invalid(
                    "expert regions must cover whole allocations without gaps or overlaps",
                ));
            }
            end = end
                .checked_add(region.bytes)
                .ok_or_else(|| invalid("expert extent overflows"))?;
        }
        if end != allocation.nbytes {
            return Err(invalid("expert regions leave unowned allocation bytes"));
        }
    }
    Ok(groups.into_keys().collect())
}

fn geometry(
    regions: &[BoundRegion],
    options: &VirtualMemoryOptions,
) -> Result<(Vec<Region>, Limits)> {
    let base = regions
        .iter()
        .map(|r| r.address)
        .min()
        .ok_or_else(|| invalid("expert storage is empty"))?;
    let mapped = regions
        .iter()
        .map(|r| {
            Ok(Region {
                expert: r.expert,
                part: r.part,
                offset: usize::try_from(r.address - base)
                    .map_err(|_| invalid("expert address extent overflows"))?,
                bytes: r.bytes,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((
        mapped,
        Limits {
            hbm_slots: options.hbm_slots,
            hbm_bytes: options.hbm_bytes,
            backing_bytes: options.backing_bytes,
            virtual_base: Some(base),
        },
    ))
}

impl NativeResources {
    pub(super) unsafe fn plan_residency(
        &mut self,
        package: &Core,
        options: &VirtualMemoryOptions,
        regions: &[BoundRegion],
    ) -> Result<()> {
        options.validate()?;
        allocations(package, regions)?;
        let (regions, limits) = geometry(regions, options)?;
        let (pool, backing) = crate::parallax::residency::gb300::package_footprint(
            &regions,
            &limits,
            &options.initial_hbm,
        )
        .map_err(|e| invalid(e.to_string()))?;
        self.parallax_device_bytes_removed = backing;
        self.parallax_pool_bytes = pool;
        let fixed = self.resident_fixed_bytes(package)?;
        let free = unsafe { self.available_memory()? } as u64;
        if fixed > free {
            return Err(invalid(format!(
                "expert pool and fixed model need {fixed} HBM bytes; only {free} are free"
            )));
        }
        Ok(())
    }

    pub(super) unsafe fn prepare_residency(
        &mut self,
        package: &Core,
        options: &VirtualMemoryOptions,
        regions: &[BoundRegion],
    ) -> Result<()> {
        options.validate()?;
        let selected = allocations(package, regions)?;
        if selected
            .iter()
            .any(|i| self.allocations.contains_key(&package.allocations[*i].name))
        {
            return Err(invalid("expert storage must be prepared before allocation"));
        }
        let (regions, limits) = geometry(regions, options)?;
        // SAFETY: The package reserves these pages before this call. No allocation
        // or graph references them yet. This resource owner retains the context.
        let loading =
            unsafe { Gb300::prepare_package(self.context as usize, self.device, regions, limits) }
                .map_err(|e| invalid(e.to_string()))?;
        if loading.backing_bytes() as u64 != self.parallax_device_bytes_removed
            || loading.hbm_bytes() as u64 != self.parallax_pool_bytes
        {
            return Err(invalid(
                "physical expert allocation differs from its admitted footprint",
            ));
        }
        self.parallax_loading = Some(loading);
        for index in selected {
            let allocation = &package.allocations[index];
            self.allocations.insert(
                allocation.name.clone(),
                Storage {
                    base: 0,
                    address: allocation.address.unwrap(),
                    nbytes: allocation.nbytes,
                    memory: Memory::Device,
                    fixed: true,
                    // Parallax owns physical mappings; the package owns address reservations.
                    mappings: Vec::new(),
                },
            );
        }
        Ok(())
    }

    pub(super) unsafe fn finish_residency(&mut self, options: &VirtualMemoryOptions) -> Result<()> {
        let loading = self
            .parallax_loading
            .take()
            .ok_or_else(|| invalid("expert storage was not prepared"))?;
        let backend = unsafe { loading.finish() }.map_err(|e| invalid(e.to_string()))?;
        let controller = Controller::new(options.catalog_id.clone(), backend)
            .map_err(|e| invalid(e.to_string()))?;
        // Retain the owner before any asynchronous operation can fail.
        let binding = ExecutionBinding::new(controller);
        self.parallax = Some(binding.clone());
        // Initial placement uses the same transaction path as serving. Publication
        // precedes graph instantiation, so there are no readers or submissions.
        if !options.initial_hbm.is_empty() {
            unsafe {
                binding.initialize(Plan {
                    version: 1,
                    catalog_id: options.catalog_id.clone(),
                    generation: 0,
                    valid_until_ns: u64::MAX,
                    promote: options.initial_hbm.clone(),
                    evict: vec![],
                })
            }
            .map_err(|e| invalid(e.to_string()))?;
        }
        Ok(())
    }

    pub(super) fn resident_fixed_bytes(&self, package: &Core) -> Result<u64> {
        super::memory::fixed_bytes(package, self.integrated)?
            .checked_sub(self.parallax_device_bytes_removed)
            .and_then(|bytes| bytes.checked_add(self.parallax_pool_bytes))
            .ok_or_else(|| invalid("expert residency footprint overflows"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallax::residency::package::Plane;

    fn sample_options() -> VirtualMemoryOptions {
        VirtualMemoryOptions {
            version: 1,
            storage: StorageConfig {
                version: 1,
                package_id: "00".repeat(32),
                rank: 0,
                planes: vec![Plane {
                    layer: 0,
                    part: 0,
                    weight_view: "layer0.w13".into(),
                    expert_ids: vec![0, 1],
                }],
            },
            catalog_id: "catalog-1".into(),
            canonical: vec![
                ExpertId {
                    layer: 0,
                    expert: 0,
                },
                ExpertId {
                    layer: 0,
                    expert: 1,
                },
            ],
            hbm_slots: 2,
            hbm_bytes: 4 << 20,
            backing_bytes: 4 << 20,
            initial_hbm: vec![ExpertId {
                layer: 0,
                expert: 0,
            }],
        }
    }

    #[test]
    fn virtual_memory_options_reject_invalid_versions_capacities_and_initial_sets() {
        let valid = sample_options();
        valid.validate().unwrap();

        for mutation in 0..8 {
            let mut bad = valid.clone();
            match mutation {
                0 => bad.version = 2,
                1 => bad.catalog_id = "   ".into(),
                2 => bad.hbm_slots = 0,
                3 => bad.hbm_bytes = 0,
                4 => bad.backing_bytes = 0,
                5 => bad.canonical.clear(),
                6 => bad.canonical.push(ExpertId {
                    layer: 0,
                    expert: 0,
                }),
                _ => bad.initial_hbm.push(ExpertId {
                    layer: 0,
                    expert: 9,
                }),
            }
            assert!(bad.validate().is_err(), "mutation {mutation} should fail");
        }
    }

    #[test]
    #[ignore = "requires an explicitly selected signed package and storage configuration; CPU only"]
    fn parallax_loader_accepts_authenticated_whole_allocations() {
        let path = std::env::var("FLARE_PARALLAX_PACKAGE").unwrap();
        let profiles = std::env::var("FLARE_PUBLISHER_PROFILES").unwrap();
        let config = std::env::var("FLARE_PARALLAX_STORAGE_CONFIG").unwrap();
        let publishers: Vec<super::super::Publisher> =
            serde_json::from_slice(&std::fs::read(profiles).unwrap()).unwrap();
        let selected =
            super::super::VerifiedCore::open(std::path::Path::new(&path), &publishers, None)
                .unwrap();
        let package = selected.authorize("local-development", false).unwrap();
        let super::super::protection::AuthorizedKind::Payload(payload) = package.kind else {
            panic!("payload package required")
        };
        let config: StorageConfig =
            serde_json::from_slice(&std::fs::read(config).unwrap()).unwrap();
        let canonical: Vec<_> = config
            .planes
            .iter()
            .flat_map(|p| {
                p.expert_ids.iter().map(|e| ExpertId {
                    layer: p.layer,
                    expert: *e,
                })
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut regions = config
            .resolve(payload.verified_metadata(), &canonical, 2 << 20)
            .unwrap();
        let package = &payload.rank_cores()[0];
        let selected = allocations(package, &regions).unwrap();
        assert_eq!(selected.len(), 84);
        eprintln!(
            "{} experts, {} entire allocations, {} packed bytes eligible for direct Grace loading",
            canonical.len(),
            selected.len(),
            regions.iter().map(|r| r.bytes as u64).sum::<u64>()
        );
        // A partial tensor view cannot silently discard the allocation's remaining bytes.
        regions.remove(0);
        assert!(allocations(package, &regions).is_err());
    }
}
