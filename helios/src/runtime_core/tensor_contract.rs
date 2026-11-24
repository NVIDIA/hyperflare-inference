/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed fixed tensor bindings carried by authenticated rank metadata.

use super::{format::invalid, wire};
use crate::Result;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct TensorView {
    pub name: String,
    pub allocation: String,
    pub offset: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl TensorView {
    pub fn nbytes(&self) -> Result<usize> {
        if self.name.is_empty()
            || !self
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
        {
            return Err(invalid(
                "tensor names require letters, digits, underscores or hyphens",
            ));
        }
        if self.allocation.is_empty() || self.shape.is_empty() || self.shape.contains(&0) {
            return Err(invalid(
                "tensor views require an allocation and positive dimensions",
            ));
        }
        let width: usize = match self.dtype.as_str() {
            "U8" => 1,
            "F16" | "BF16" => 2,
            "F32" | "I32" => 4,
            "I64" => 8,
            _ => {
                return Err(invalid(format!(
                    "unsupported tensor dtype {:?}",
                    self.dtype
                )));
            }
        };
        if self.offset % width != 0 {
            return Err(invalid("tensor offset is not aligned to its element size"));
        }
        self.shape
            .iter()
            .try_fold(width, |bytes, extent| bytes.checked_mul(*extent))
            .ok_or_else(|| invalid("tensor byte extent overflows"))
    }
}

#[derive(Clone, Debug)]
pub struct PreparedTensorContract {
    graphs: Vec<String>,
    inputs: Vec<TensorView>,
    outputs: Vec<TensorView>,
}

impl PreparedTensorContract {
    /// Validate declaration geometry; loading also checks the signed resources.
    pub fn parse(value: &wire::TensorContract) -> Result<Self> {
        if value.version != 1 {
            return Err(invalid("unsupported tensor contract version"));
        }
        let mut graphs = HashSet::new();
        if value.graphs.is_empty()
            || value
                .graphs
                .iter()
                .any(|name| name.is_empty() || !graphs.insert(name))
        {
            return Err(invalid(
                "tensor contracts require a nonempty sequence of unique graph names",
            ));
        }
        if value.inputs.is_empty() || value.outputs.is_empty() {
            return Err(invalid("tensor contracts require inputs and outputs"));
        }
        let inputs = value
            .inputs
            .iter()
            .map(binding)
            .collect::<Result<Vec<_>>>()?;
        let outputs = value
            .outputs
            .iter()
            .map(binding)
            .collect::<Result<Vec<_>>>()?;
        let mut names = HashSet::new();
        for view in inputs.iter().chain(&outputs) {
            view.offset
                .checked_add(view.nbytes()?)
                .ok_or_else(|| invalid("tensor range overflows"))?;
            if !names.insert(&view.name) {
                return Err(invalid(format!("duplicate tensor name {:?}", view.name)));
            }
        }
        for (index, view) in inputs.iter().enumerate() {
            let end = view
                .offset
                .checked_add(view.nbytes()?)
                .ok_or_else(|| invalid("input tensor range overflows"))?;
            for other in &inputs[..index] {
                let other_end = other
                    .offset
                    .checked_add(other.nbytes()?)
                    .ok_or_else(|| invalid("input tensor range overflows"))?;
                if view.allocation == other.allocation
                    && view.offset < other_end
                    && other.offset < end
                {
                    return Err(invalid("input tensor allocation ranges overlap"));
                }
            }
        }
        Ok(Self {
            graphs: value.graphs.clone(),
            inputs,
            outputs,
        })
    }

    pub fn graphs(&self) -> &[String] {
        &self.graphs
    }
    pub fn inputs(&self) -> &[TensorView] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[TensorView] {
        &self.outputs
    }

    fn resources<'a>(
        &self,
        allocations: impl Iterator<Item = (&'a str, u64, bool)>,
        graphs: impl Iterator<Item = &'a str>,
    ) -> Result<()> {
        let mut extents = HashMap::new();
        for (name, nbytes, fixed_device) in allocations {
            if extents.insert(name, (nbytes, fixed_device)).is_some() {
                return Err(invalid("duplicate tensor allocation name"));
            }
        }
        let mut catalog = HashSet::new();
        for name in graphs {
            if !catalog.insert(name) {
                return Err(invalid("duplicate tensor graph name"));
            }
        }
        for name in &self.graphs {
            if !catalog.contains(name.as_str()) {
                return Err(invalid(format!("unknown tensor graph {name:?}")));
            }
        }
        for view in self.inputs.iter().chain(&self.outputs) {
            let &(extent, fixed_device) =
                extents.get(view.allocation.as_str()).ok_or_else(|| {
                    invalid(format!("unknown tensor allocation {:?}", view.allocation))
                })?;
            if !fixed_device {
                return Err(invalid("tensor bindings require fixed device allocations"));
            }
            let end = view
                .offset
                .checked_add(view.nbytes()?)
                .ok_or_else(|| invalid("tensor allocation range overflows"))?;
            if end as u64 > extent {
                return Err(invalid("tensor range lies outside allocation"));
            }
        }
        Ok(())
    }
}

fn binding(value: &wire::TensorBinding) -> Result<TensorView> {
    let reference = value
        .reference
        .as_ref()
        .ok_or_else(|| invalid("tensor binding has no allocation reference"))?;
    let dtype = match wire::TensorDtype::try_from(value.dtype) {
        Ok(wire::TensorDtype::U8) => "U8",
        Ok(wire::TensorDtype::F16) => "F16",
        Ok(wire::TensorDtype::Bf16) => "BF16",
        Ok(wire::TensorDtype::F32) => "F32",
        Ok(wire::TensorDtype::I32) => "I32",
        Ok(wire::TensorDtype::I64) => "I64",
        _ => return Err(invalid("unknown or unspecified tensor dtype")),
    };
    let view = TensorView {
        name: value.name.clone(),
        allocation: reference.allocation.clone(),
        offset: usize::try_from(reference.offset)
            .map_err(|_| invalid("tensor offset overflows"))?,
        dtype: dtype.into(),
        shape: value
            .shape
            .iter()
            .map(|&v| usize::try_from(v).map_err(|_| invalid("tensor dimension overflows")))
            .collect::<Result<_>>()?,
    };
    view.nbytes()?;
    Ok(view)
}

/// Validate logical-resource rank bindings without requiring them on token cores.
pub fn validate_bundle(bundle: &wire::BundleIndex) -> Result<Option<PreparedTensorContract>> {
    let Some(value) = &bundle.tensor_contract else {
        return Ok(None);
    };
    if bundle.execution.is_some()
        || bundle.roce.is_some()
        || bundle
            .allocations
            .iter()
            .any(|a| a.memory == wire::Memory::Peer as i32)
    {
        return Err(invalid(
            "tensor contract requires an independent rank without token execution",
        ));
    }
    if bundle.allocations.iter().any(|a| a.sizing.is_some()) {
        return Err(invalid("tensor contracts require fixed allocations"));
    }
    let contract = PreparedTensorContract::parse(value)?;
    contract.resources(
        bundle.allocations.iter().map(|a| {
            (
                a.name.as_str(),
                a.nbytes,
                a.memory == wire::Memory::Device as i32 && a.sizing.is_none(),
            )
        }),
        bundle.graphs.iter().map(|g| g.name.as_str()),
    )?;
    Ok(Some(contract))
}

/// Validate payload rank bindings using the same declaration and resource rules.
pub fn validate_rank(
    rank: &crate::core::schema::wire::RankBody,
) -> Result<Option<PreparedTensorContract>> {
    let Some(value) = &rank.tensor_contract else {
        return Ok(None);
    };
    if rank.rank != 0
        || rank.execution.is_some()
        || rank.roce.is_some()
        || !rank.peers.is_empty()
        || rank
            .allocations
            .iter()
            .any(|a| a.memory == wire::Memory::Peer as i32)
    {
        return Err(invalid(
            "tensor contract requires one independent rank without token execution",
        ));
    }
    if rank.allocations.iter().any(|a| a.sizing.is_some()) {
        return Err(invalid("tensor contracts require fixed allocations"));
    }
    let contract = PreparedTensorContract::parse(value)?;
    contract.resources(
        rank.allocations.iter().map(|a| {
            (
                a.name.as_str(),
                a.nbytes,
                a.memory == wire::Memory::Device as i32 && a.sizing.is_none(),
            )
        }),
        rank.graphs.iter().map(|g| g.name.as_str()),
    )?;
    Ok(Some(contract))
}

/// Keep signed tensor programs separate from token-serving and dependent ranks.
pub fn validate_model(index: &wire::ModelIndex) -> Result<()> {
    if !index.ranks.iter().any(|r| r.tensor_contract.is_some()) {
        return Ok(());
    }
    if index.ranks.len() != 1
        || !index.peer_memory.is_empty()
        || index.cluster.is_some()
        || index.engram.is_some()
        || index.contract.is_some()
        || index.frontend.is_some()
        || index.sampling_defaults.is_some()
        || !index.assets.is_empty()
    {
        return Err(invalid(
            "tensor contract requires one independent rank without token-serving metadata",
        ));
    }
    validate_bundle(&index.ranks[0])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn contract() -> wire::TensorContract {
        let binding = |name: &str, offset, dtype, shape| wire::TensorBinding {
            name: name.into(),
            reference: Some(wire::Reference {
                allocation: "storage".into(),
                offset,
            }),
            dtype,
            shape,
        };
        wire::TensorContract {
            version: 1,
            graphs: vec!["encode".into(), "step".into(), "decode".into()],
            inputs: vec![binding(
                "noise",
                0,
                wire::TensorDtype::F16 as i32,
                vec![1, 4, 8, 8],
            )],
            outputs: vec![binding(
                "image",
                512,
                wire::TensorDtype::F32 as i32,
                vec![1, 3, 64, 64],
            )],
        }
    }

    fn bundle() -> wire::BundleIndex {
        wire::BundleIndex {
            allocations: vec![wire::Allocation {
                name: "storage".into(),
                nbytes: 49664,
                ..Default::default()
            }],
            graphs: contract()
                .graphs
                .iter()
                .map(|name| wire::GraphIndex {
                    name: name.clone(),
                    ..Default::default()
                })
                .collect(),
            tensor_contract: Some(contract()),
            ..Default::default()
        }
    }

    #[test]
    fn tensor_contract_roundtrip_preserves_order_and_bindings() {
        let original = bundle();
        let decoded = wire::BundleIndex::decode(original.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded.tensor_contract, original.tensor_contract);
        let value = validate_bundle(&decoded).unwrap().unwrap();
        assert_eq!(value.graphs(), &["encode", "step", "decode"]);
        assert_eq!(value.inputs()[0].nbytes().unwrap(), 512);
        assert_eq!(value.outputs()[0].nbytes().unwrap(), 49152);
    }

    #[test]
    fn rejects_invalid_shape_dtype_alignment_and_overflow() {
        for (shape, dtype, offset) in [
            (vec![], wire::TensorDtype::F16 as i32, 0),
            (vec![0], wire::TensorDtype::F16 as i32, 0),
            (vec![u64::MAX, 2], wire::TensorDtype::U8 as i32, 0),
            (vec![1], 0, 0),
            (vec![1], 123, 0),
            (vec![1], wire::TensorDtype::F32 as i32, 2),
            (vec![2], wire::TensorDtype::U8 as i32, u64::MAX),
        ] {
            let mut value = contract();
            value.inputs[0].shape = shape;
            value.inputs[0].dtype = dtype;
            value.inputs[0].reference.as_mut().unwrap().offset = offset;
            assert!(PreparedTensorContract::parse(&value).is_err());
        }
        let mut value = contract();
        value.outputs[0].name = "../image".into();
        assert!(PreparedTensorContract::parse(&value).is_err());
        value = contract();
        value.outputs[0].reference = None;
        assert!(PreparedTensorContract::parse(&value).is_err());
    }

    #[test]
    fn rejects_empty_or_duplicate_graphs_and_descriptors() {
        for graphs in [vec![], vec!["step".into(), "step".into()], vec!["".into()]] {
            let mut value = contract();
            value.graphs = graphs;
            assert!(PreparedTensorContract::parse(&value).is_err());
        }
        let mut value = contract();
        value.outputs[0].name = "noise".into();
        assert!(PreparedTensorContract::parse(&value).is_err());
        for version in [0, 2, u32::MAX] {
            value = contract();
            value.version = version;
            assert!(PreparedTensorContract::parse(&value).is_err());
        }
        value = contract();
        value.inputs.clear();
        assert!(PreparedTensorContract::parse(&value).is_err());
        value = contract();
        value.outputs.clear();
        assert!(PreparedTensorContract::parse(&value).is_err());
    }

    #[test]
    fn rejects_overlapping_inputs_but_accepts_adjacent_ranges() {
        let mut value = contract();
        let mut second = value.inputs[0].clone();
        second.name = "noise2".into();
        second.reference.as_mut().unwrap().offset = 510;
        value.inputs.push(second);
        assert!(PreparedTensorContract::parse(&value).is_err());
        value.inputs[1].reference.as_mut().unwrap().offset = 512;
        PreparedTensorContract::parse(&value).unwrap();
    }

    #[test]
    fn rejects_missing_graphs_and_nonfixed_or_out_of_bounds_storage() {
        for case in 0..8 {
            let mut value = bundle();
            match case {
                0 => {
                    value.graphs.pop();
                }
                1 => value.allocations[0].nbytes -= 1,
                2 => value.allocations[0].name = "other".into(),
                3 => value.allocations[0].memory = wire::Memory::MappedHost as i32,
                4 => value.allocations[0].sizing = Some(wire::Sizing::default()),
                5 => value.allocations.push(value.allocations[0].clone()),
                6 => value.graphs.push(value.graphs[0].clone()),
                7 => value.allocations[0].memory = wire::Memory::Peer as i32,
                _ => unreachable!(),
            }
            assert!(validate_bundle(&value).is_err(), "case {case}");
        }
    }

    #[test]
    fn rejects_unbound_dynamic_storage_but_allows_fixed_host_scratch() {
        let mut value = bundle();
        value.allocations.push(wire::Allocation {
            name: "scratch".into(),
            nbytes: 256,
            memory: wire::Memory::MappedHost as i32,
            ..Default::default()
        });
        validate_bundle(&value).unwrap();
        value.allocations[1].sizing = Some(wire::Sizing::default());
        assert!(
            validate_bundle(&value)
                .unwrap_err()
                .to_string()
                .contains("fixed allocations")
        );
        let mut rank = crate::core::schema::wire::RankBody {
            tensor_contract: value.tensor_contract.clone(),
            allocations: value
                .allocations
                .iter()
                .map(|a| crate::core::schema::wire::Allocation {
                    name: a.name.clone(),
                    nbytes: a.nbytes,
                    memory: a.memory,
                    sizing: a.sizing.clone(),
                    ..Default::default()
                })
                .collect(),
            graphs: value
                .graphs
                .iter()
                .map(|g| crate::core::schema::wire::Graph {
                    name: g.name.clone(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        assert!(
            validate_rank(&rank)
                .unwrap_err()
                .to_string()
                .contains("fixed allocations")
        );
        rank.allocations[1].sizing = None;
        validate_rank(&rank).unwrap();
    }

    #[test]
    fn rejects_payload_token_execution_and_peer_dependencies() {
        for case in 0..4 {
            let mut rank = crate::core::schema::wire::RankBody {
                tensor_contract: Some(contract()),
                ..Default::default()
            };
            match case {
                0 => rank.rank = 1,
                1 => rank.execution = Some(wire::Execution::default()),
                2 => rank.roce = Some(wire::RoceTransport::default()),
                3 => rank.allocations.push(crate::core::schema::wire::Allocation {
                    memory: wire::Memory::Peer as i32,
                    ..Default::default()
                }),
                _ => unreachable!(),
            }
            assert!(
                validate_rank(&rank)
                    .unwrap_err()
                    .to_string()
                    .contains("independent rank")
            );
        }
    }

    #[test]
    fn rejects_mixed_or_dependent_tensor_contracts_without_changing_token_cores() {
        let mut value = bundle();
        value.execution = Some(wire::Execution::default());
        assert!(validate_bundle(&value).is_err());
        value.tensor_contract = None;
        assert!(validate_bundle(&value).unwrap().is_none());
        for case in 0..5 {
            let mut index = wire::ModelIndex {
                ranks: vec![bundle()],
                ..Default::default()
            };
            match case {
                0 => index.ranks.push(bundle()),
                1 => index.contract = Some(wire::ServingContract::default()),
                2 => index.cluster = Some(wire::Cluster::default()),
                3 => index.peer_memory.push(wire::PeerMemory::default()),
                4 => index.ranks[0].roce = Some(wire::RoceTransport::default()),
                _ => unreachable!(),
            }
            assert!(validate_model(&index).is_err(), "case {case}");
        }
        validate_model(&wire::ModelIndex {
            ranks: vec![bundle()],
            ..Default::default()
        })
        .unwrap();
    }
}
