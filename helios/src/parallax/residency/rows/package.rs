//! Bind authenticated canonical row declarations to immutable weight payloads.
use super::{Layer, Layout, Plane, Result, invalid};
use crate::core::{prepared_weights, protection::VerifiedCore, schema::wire};
use std::collections::BTreeMap;

/// Resolve storage before CUDA work. Ordinary loading verifies payload contents
/// and retains their owners before these addresses may be passed to `Rows::new`.
pub fn resolve(core: &VerifiedCore, rank: u32) -> Result<Vec<Layer>> {
    if rank != 0 || core.deployment().is_none_or(|d| d.tensor_parallel != 1) {
        return Err(invalid(
            "expert row movement requires one local execution rank",
        ));
    }
    resolve_body(
        core.rank(rank)
            .ok_or_else(|| invalid("expert row rank is absent"))?,
    )
}

fn resolve_body(body: &wire::RankBody) -> Result<Vec<Layer>> {
    let execution = body
        .execution
        .as_ref()
        .ok_or_else(|| invalid("execution metadata is absent"))?;
    let rows = execution
        .expert_rows
        .as_ref()
        .ok_or_else(|| invalid("core has no expert row storage declaration"))?;
    if rows.version != 1 || rows.layers.is_empty() {
        return Err(invalid(
            "unsupported or empty expert row storage declaration",
        ));
    }
    let catalog = prepared_weights::catalog(body)
        .map_err(|e| invalid(&e.to_string()))?
        .filter(|c| c.version == 1)
        .ok_or_else(|| invalid("expert rows require a prepared-weight catalog"))?;
    let views: BTreeMap<_, _> = catalog.views.iter().map(|v| (v.name.as_str(), v)).collect();
    let buffers: BTreeMap<_, _> = execution
        .buffers
        .iter()
        .map(|v| (v.name.as_str(), v))
        .collect();
    if views.len() != catalog.views.len() || buffers.len() != execution.buffers.len() {
        return Err(invalid("duplicate expert weight or buffer names"));
    }
    let mut layers = Vec::new();
    for row in &rows.layers {
        let buffer = buffers
            .get(row.mapping_buffer.as_str())
            .ok_or_else(|| invalid("expert map buffer is absent"))?;
        let reference = buffer
            .reference
            .as_ref()
            .ok_or_else(|| invalid("expert map reference is absent"))?;
        let allocation = body
            .allocations
            .iter()
            .find(|a| a.name == reference.allocation)
            .ok_or_else(|| invalid("expert map allocation is absent"))?;
        let bytes = u64::from(row.experts) * 4;
        if buffer.dtype != "int32"
            || buffer.shape != [u64::from(row.experts)]
            || buffer.strides != [1]
            || buffer.nbytes != bytes
            || execution.graphs.iter().any(|g| {
                g.input_bindings
                    .iter()
                    .any(|b| b.buffer_name == buffer.name)
                    || g.tuned_bindings.iter().any(|b| b.buffer == buffer.name)
            })
            || execution
                .tuning_graphs
                .iter()
                .any(|g| g.outputs.contains(&buffer.name) || g.status_buffer == buffer.name)
        {
            return Err(invalid(
                "expert map must be a dedicated int32 buffer without host or tuning writers",
            ));
        }
        let mapping_address = range(allocation, reference.offset, bytes, 0)?;
        initialized(body, allocation, reference.offset, bytes, false)?;
        let mut planes = Vec::new();
        for plane in &row.planes {
            let host = views
                .get(plane.host_weight_view.as_str())
                .ok_or_else(|| invalid("host expert weight view is absent"))?;
            let hbm = views
                .get(plane.hbm_weight_view.as_str())
                .ok_or_else(|| invalid("HBM expert weight view is absent"))?;
            let (host_address, host_bytes) = weight(body, host, row.host_experts.len(), 1)?;
            let (hbm_address, row_bytes) = weight(body, hbm, row.hbm_experts.len(), 0)?;
            if host_bytes != row_bytes
                || host.dtype != hbm.dtype
                || host.element_bytes != hbm.element_bytes
                || host.shape[1..] != hbm.shape[1..]
                || host.stride != hbm.stride
            {
                return Err(invalid(
                    "host and HBM expert planes have different row layouts",
                ));
            }
            planes.push(Plane {
                host_address,
                hbm_address,
                row_bytes,
            });
        }
        layers.push(Layer {
            layer: row.layer as usize,
            experts: row.experts as usize,
            host_experts: row.host_experts.iter().map(|v| *v as usize).collect(),
            hbm_experts: row.hbm_experts.iter().map(|v| *v as usize).collect(),
            planes,
            mapping_address,
        });
    }
    // Reuse the backend's canonical coverage, physical overlap and row checks.
    Layout::new(layers.clone())?;
    Ok(layers)
}

fn weight(
    body: &wire::RankBody,
    view: &wire::PreparedWeightView,
    count: usize,
    memory: i32,
) -> Result<(u64, usize)> {
    if count == 0
        || view.shape.len() < 2
        || view.shape[0] != count as u64
        || view.shape.len() != view.stride.len()
        || view.element_bytes == 0
    {
        return Err(invalid("expert weight view has invalid row geometry"));
    }
    let mut axes: Vec<_> = view.stride[1..]
        .iter()
        .copied()
        .zip(view.shape[1..].iter().copied())
        .collect();
    axes.sort_unstable();
    let mut elements = 1u64;
    for (stride, size) in axes {
        if size == 0 || (size > 1 && stride != elements) {
            return Err(invalid(
                "expert weight rows must be dense and nonoverlapping",
            ));
        }
        elements = elements
            .checked_mul(size)
            .ok_or_else(|| invalid("expert row extent overflows"))?;
    }
    let bytes = elements
        .checked_mul(u64::from(view.element_bytes))
        .ok_or_else(|| invalid("expert row bytes overflow"))?;
    if view.stride[0] != elements || bytes.checked_mul(count as u64) != Some(view.nbytes) {
        return Err(invalid(
            "expert weight row extent differs from prepared bytes",
        ));
    }
    let allocation = body
        .allocations
        .get(view.allocation as usize)
        .ok_or_else(|| invalid("expert weight allocation is absent"))?;
    let address = range(allocation, view.offset, view.nbytes, memory)?;
    initialized(body, allocation, view.offset, view.nbytes, true)?;
    Ok((
        address,
        usize::try_from(bytes).map_err(|_| invalid("expert row exceeds host address space"))?,
    ))
}

fn range(allocation: &wire::Allocation, offset: u64, bytes: u64, memory: i32) -> Result<u64> {
    let end = offset
        .checked_add(bytes)
        .filter(|end| bytes != 0 && *end <= allocation.nbytes)
        .ok_or_else(|| invalid("expert storage exceeds its allocation"))?;
    if allocation.memory != memory
        || allocation.sizing.is_some()
        || allocation
            .relocations
            .iter()
            .any(|r| r.offset < end && r.offset.saturating_add(128) > offset)
    {
        return Err(invalid(
            "expert storage requires fixed, numeric, tier-matched allocations",
        ));
    }
    allocation
        .address
        .and_then(|a| a.checked_add(offset))
        .filter(|a| *a != 0 && a.checked_add(bytes).is_some())
        .ok_or_else(|| invalid("expert storage has an invalid fixed address"))
}

fn initialized(
    body: &wire::RankBody,
    allocation: &wire::Allocation,
    offset: u64,
    bytes: u64,
    weight: bool,
) -> Result<()> {
    let end = offset + bytes; // range() checked this sum.
    let mut spans = Vec::new();
    for i in body
        .initializers
        .iter()
        .filter(|i| i.allocation == allocation.name)
    {
        let source = i
            .source
            .as_ref()
            .ok_or_else(|| invalid("expert initializer has no source"))?;
        let slice = source
            .slice
            .as_ref()
            .ok_or_else(|| invalid("expert initializer has no payload slice"))?;
        if weight && source.role != wire::PayloadRole::Weights as i32 {
            continue;
        }
        let stop = i
            .destination_offset
            .checked_add(slice.nbytes)
            .ok_or_else(|| invalid("expert initializer overflows"))?;
        spans.push((i.destination_offset, stop));
    }
    spans.sort_unstable();
    let mut covered = offset;
    for (start, stop) in spans {
        if start <= covered && covered < stop {
            covered = stop;
        }
        if covered >= end {
            return Ok(());
        }
    }
    Err(invalid(
        "expert storage is not fully initialized from the required payloads",
    ))
}

#[cfg(test)]
mod tests;
