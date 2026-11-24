/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fixed tensor requests over authenticated, exported CUDA graphs.

use std::{collections::HashMap, path::PathBuf, time::Instant};

use serde::{Deserialize, Serialize};

use super::{NativeResources, format::invalid, tensor_contract::PreparedTensorContract};
use crate::{
    BufferSpan, Result,
    cuda::{Event, Stream},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorInput {
    pub name: String,
    pub file: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRequest {
    pub inputs: Vec<TensorInput>,
}

impl TensorRequest {
    /// Match files by name; only the signed contract determines binding order.
    pub fn bind(&self, contract: &PreparedTensorContract) -> Result<Vec<&TensorInput>> {
        if self.inputs.len() != contract.inputs().len() {
            return Err(invalid("input tensor count differs from signed contract"));
        }
        let mut inputs = HashMap::new();
        for input in &self.inputs {
            if inputs.insert(input.name.as_str(), input).is_some() {
                return Err(invalid(format!("duplicate input tensor {:?}", input.name)));
            }
            if input.file.as_os_str().is_empty() {
                return Err(invalid("input tensor file path is empty"));
            }
        }
        contract
            .inputs()
            .iter()
            .map(|view| {
                inputs.remove(view.name.as_str()).ok_or_else(|| {
                    invalid(format!("request is missing signed input {:?}", view.name))
                })
            })
            .collect()
    }
}

pub(super) fn range_address(span: &BufferSpan, offset: usize, nbytes: usize) -> Result<usize> {
    let end = offset
        .checked_add(nbytes)
        .ok_or_else(|| invalid("tensor allocation range overflows"))?;
    if nbytes == 0 || end > span.nbytes {
        return Err(invalid("tensor range lies outside allocation"));
    }
    span.address
        .checked_add(offset)
        .and_then(|address| address.checked_add(nbytes).map(|_| address))
        .ok_or_else(|| invalid("tensor device address overflows"))
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorTiming {
    pub input_upload_wall_ns: u64,
    pub replay_wall_ns: u64,
    pub replay_gpu_ns: u64,
    pub graph_gpu_ns: Vec<u64>,
}

/// Owns storage, a stream and timing events through confirmed replay drain.
/// Every operation is a graph launch or a byte copy; tensor arithmetic belongs
/// to the exported graphs.
pub struct TensorRunner {
    owner: Option<NativeResources>,
    stream: Option<Stream>,
    events: Vec<Event>,
    graphs: Vec<usize>,
    contract: PreparedTensorContract,
}

impl TensorRunner {
    pub fn new(owner: NativeResources) -> Result<Self> {
        let contract = owner.tensor_contract()?.clone();
        for view in contract.inputs().iter().chain(contract.outputs()) {
            range_address(
                &owner.allocation(&view.allocation)?,
                view.offset,
                view.nbytes()?,
            )?;
        }
        let bundle = owner.graph_bundle();
        let graphs = contract
            .graphs()
            .iter()
            .map(|name| {
                owner
                    .graph_names()
                    .iter()
                    .position(|candidate| candidate == name)
                    .map(|index| bundle.graph_execs[index])
                    .ok_or_else(|| invalid(format!("unknown tensor graph {name:?}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let stream = Stream::open(bundle.context, bundle.device)?;
        let events = (0..=graphs.len())
            .map(|_| stream.event_with_timing(true))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            owner: Some(owner),
            stream: Some(stream),
            events,
            graphs,
            contract,
        })
    }

    /// Upload all request inputs and execute every graph once without intergraph
    /// host synchronization. Outputs remain resident until explicitly downloaded.
    pub fn run(&mut self, inputs: &[Vec<u8>]) -> Result<TensorTiming> {
        if inputs.len() != self.contract.inputs().len() {
            return Err(invalid("input tensor count differs from signed contract"));
        }
        for (view, bytes) in self.contract.inputs().iter().zip(inputs) {
            if bytes.len() != view.nbytes()? {
                return Err(invalid(format!(
                    "input tensor {:?} byte extent differs from signed contract",
                    view.name
                )));
            }
        }
        let owner = self.owner.as_ref().unwrap();
        let stream = self.stream.as_ref().unwrap();
        let upload = Instant::now();
        for (view, bytes) in self.contract.inputs().iter().zip(inputs) {
            owner.upload_range(&view.allocation, view.offset, bytes)?;
        }
        let input_upload_wall_ns = nanos(upload.elapsed());
        let replay = Instant::now();
        stream.record(&self.events[0])?;
        for (index, graph) in self.graphs.iter().enumerate() {
            stream.launch(*graph)?;
            stream.record(&self.events[index + 1])?;
        }
        stream.synchronize()?;
        let replay_wall_ns = nanos(replay.elapsed());
        let graph_gpu_ns = self
            .events
            .windows(2)
            .map(|events| events[1].elapsed_ns(&events[0]))
            .collect::<Result<Vec<_>>>()?;
        let replay_gpu_ns = self.events.last().unwrap().elapsed_ns(&self.events[0])?;
        Ok(TensorTiming {
            input_upload_wall_ns,
            replay_wall_ns,
            replay_gpu_ns,
            graph_gpu_ns,
        })
    }

    pub fn outputs(&self) -> Result<Vec<Vec<u8>>> {
        let owner = self.owner.as_ref().unwrap();
        self.contract
            .outputs()
            .iter()
            .map(|view| owner.download_range(&view.allocation, view.offset, view.nbytes()?))
            .collect()
    }

    /// Confirm stream drain before releasing timing events and loaded resources.
    pub fn close(mut self) -> Result<()> {
        self.finish()
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(stream) = self.stream.take() {
            let owner = self.owner.as_ref().unwrap();
            // SAFETY: The context remains owned until its queued work drains.
            let bound = unsafe {
                owner
                    .driver
                    .check("cuCtxSetCurrent", (owner.driver.context_set)(owner.context))
            };
            if let Err(error) = bound.and_then(|_| stream.synchronize()) {
                std::mem::forget(stream);
                std::mem::forget(std::mem::take(&mut self.events));
                std::mem::forget(self.owner.take());
                return Err(error);
            }
            self.events.clear();
            stream.destroy()?;
        }
        Ok(())
    }
}

fn nanos(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

impl Drop for TensorRunner {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> PreparedTensorContract {
        use super::super::wire;
        let binding = |name: &str, offset, dtype| wire::TensorBinding {
            name: name.into(),
            reference: Some(wire::Reference {
                allocation: "storage".into(),
                offset,
            }),
            dtype: dtype as i32,
            shape: vec![1],
        };
        PreparedTensorContract::parse(&wire::TensorContract {
            version: 1,
            graphs: vec!["encode".into(), "step".into(), "decode".into()],
            inputs: vec![
                binding("noise", 0, wire::TensorDtype::F16),
                binding("guidance", 4, wire::TensorDtype::F32),
            ],
            outputs: vec![binding("image", 8, wire::TensorDtype::F32)],
        })
        .unwrap()
    }

    fn request(names: &[&str]) -> TensorRequest {
        TensorRequest {
            inputs: names
                .iter()
                .map(|name| TensorInput {
                    name: (*name).into(),
                    file: PathBuf::from(format!("{name}.bin")),
                })
                .collect(),
        }
    }

    #[test]
    fn binds_files_by_signed_names_in_contract_order() {
        let contract = contract();
        let request = request(&["guidance", "noise"]);
        let bound = request.bind(&contract).unwrap();
        assert_eq!(
            bound
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            ["noise", "guidance"]
        );
        assert_eq!(bound[0].file, PathBuf::from("noise.bin"));
    }

    #[test]
    fn rejects_missing_unknown_duplicate_and_empty_inputs() {
        let contract = contract();
        for names in [
            &[][..],
            &["noise"],
            &["noise", "guidance", "extra"],
            &["noise", "extra"],
            &["noise", "noise"],
        ] {
            assert!(request(names).bind(&contract).is_err(), "{names:?}");
        }
        let mut request = request(&["noise", "guidance"]);
        request.inputs[0].file = PathBuf::new();
        assert!(request.bind(&contract).is_err());
    }

    #[test]
    fn request_cannot_override_signed_execution_or_bindings() {
        let request = serde_json::json!({"inputs":[
            {"name":"noise", "file":"noise.bin"},
            {"name":"guidance", "file":"guidance.bin"}
        ]});
        assert!(serde_json::from_value::<TensorRequest>(request.clone()).is_ok());
        for (field, value) in [
            ("graphs", serde_json::json!(["decode"])),
            ("outputs", serde_json::json!([])),
        ] {
            let mut altered = request.clone();
            altered[field] = value;
            assert!(serde_json::from_value::<TensorRequest>(altered).is_err());
        }
        for (field, value) in [
            ("allocation", serde_json::json!("weights")),
            ("offset", serde_json::json!(8)),
            ("dtype", serde_json::json!("I64")),
            ("shape", serde_json::json!([1024])),
        ] {
            let mut altered = request.clone();
            altered["inputs"][0][field] = value;
            assert!(serde_json::from_value::<TensorRequest>(altered).is_err());
        }
    }

    #[test]
    fn checks_range_extent_and_address_overflow() {
        let span = BufferSpan {
            address: 4096,
            nbytes: 512,
        };
        assert_eq!(range_address(&span, 256, 256).unwrap(), 4352);
        for (offset, nbytes) in [(512, 1), (0, 0), (usize::MAX, 1)] {
            assert!(range_address(&span, offset, nbytes).is_err());
        }
        assert!(
            range_address(
                &BufferSpan {
                    address: usize::MAX,
                    nbytes: 2
                },
                0,
                2
            )
            .is_err()
        );
    }
}
