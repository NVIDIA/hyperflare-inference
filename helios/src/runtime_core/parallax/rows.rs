//! Attach row storage after ordinary package loading and before graph submission.
use super::*;
use crate::{
    core::PayloadId,
    parallax::residency::rows::{Layer, Rows, package},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RowResidencyOptions {
    pub version: u32,
    pub package_id: String,
    pub catalog_id: String,
    pub canonical: Vec<ExpertId>,
    pub spare_rows_per_layer: usize,
}
impl RowResidencyOptions {
    pub(super) fn resolve(
        &self,
        core: &crate::core::protection::VerifiedCore,
    ) -> Result<ResolvedResidency> {
        let id = PayloadId::parse(&self.package_id)?;
        if self.version != 2
            || id.as_bytes() != &core.core_id()
            || self.catalog_id.trim().is_empty()
        {
            return Err(invalid(
                "row residency requires version 2 and the loaded core identity",
            ));
        }
        let layers = package::resolve(core, 0).map_err(|e| invalid(e.to_string()))?;
        let evict = spare_experts(&layers, &self.canonical, self.spare_rows_per_layer)?;
        Ok(ResolvedResidency::Rows {
            catalog_id: self.catalog_id.clone(),
            layers,
            evict,
        })
    }
}

fn spare_experts(layers: &[Layer], canonical: &[ExpertId], count: usize) -> Result<Vec<ExpertId>> {
    let declared: BTreeSet<_> = layers
        .iter()
        .flat_map(|l| {
            (0..l.experts).map(move |expert| ExpertId {
                layer: l.layer,
                expert,
            })
        })
        .collect();
    if count == 0
        || declared.is_empty()
        || declared.len() != canonical.len()
        || canonical.iter().copied().collect::<BTreeSet<_>>() != declared
    {
        return Err(invalid(
            "row residency requires the complete canonical catalog and positive spare capacity",
        ));
    }
    let mut evict = Vec::new();
    for layer in layers {
        let backing: BTreeSet<_> = layer.host_experts.iter().copied().collect();
        let cached: Vec<_> = layer
            .hbm_experts
            .iter()
            .rev()
            .copied()
            .filter(|e| backing.contains(e))
            .collect();
        if count >= cached.len() {
            return Err(invalid(
                "spare rows must leave at least one replaceable HBM expert per layer",
            ));
        }
        evict.extend(cached.into_iter().take(count).map(|expert| ExpertId {
            layer: layer.layer,
            expert,
        }));
    }
    Ok(evict)
}

impl NativeResources {
    pub(super) unsafe fn finish_row_residency(
        &mut self,
        catalog_id: String,
        layers: Vec<Layer>,
        evict: Vec<ExpertId>,
    ) -> Result<()> {
        // Resolve again against the actual retained allocations. The signed row
        // resolver already checked the byte extents and tier-specific layouts.
        for layer in &layers {
            let ranges = std::iter::once((layer.mapping_address, layer.experts * 4, false)).chain(
                layer.planes.iter().flat_map(|p| {
                    [
                        (p.host_address, p.row_bytes * layer.host_experts.len(), true),
                        (p.hbm_address, p.row_bytes * layer.hbm_experts.len(), false),
                    ]
                }),
            );
            for (address, bytes, host) in ranges {
                if !self.allocations.values().any(|a| {
                    a.address <= address
                        && address
                            .checked_add(bytes as u64)
                            .zip(a.address.checked_add(a.nbytes as u64))
                            .is_some_and(|(end, limit)| end <= limit)
                        && if host {
                            matches!(a.memory, Memory::MappedHost)
                        } else {
                            matches!(a.memory, Memory::Device)
                        }
                }) {
                    return Err(invalid(
                        "expert row storage is not owned by the loaded rank",
                    ));
                }
            }
        }
        // SAFETY: All ranges belong to this owner after authenticated payload
        // loading. No graphs have been submitted. NativeResources retains CUDA
        // storage/context until the existing binding drain succeeds.
        let backend = unsafe { Rows::new(self.context as usize, self.device, layers) }
            .map_err(|e| invalid(e.to_string()))?;
        let controller =
            Controller::new(catalog_id.clone(), backend).map_err(|e| invalid(e.to_string()))?;
        let binding = ExecutionBinding::new(controller);
        self.parallax = Some(binding.clone());
        // Evict only duplicate cached rows. Both static and active experiments
        // reserve the same capacity before any serving graph is instantiated.
        unsafe {
            binding.initialize(Plan {
                version: 1,
                catalog_id,
                generation: 0,
                valid_until_ns: u64::MAX,
                promote: vec![],
                evict,
            })
        }
        .map_err(|e| invalid(e.to_string()))
    }
}

#[cfg(test)]
mod tests;
