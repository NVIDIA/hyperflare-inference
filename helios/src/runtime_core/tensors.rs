/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fixed tensor requests over authenticated, exported CUDA graphs.

use std::{collections::HashSet, path::PathBuf, time::Instant};

use serde::{Deserialize, Serialize};

use super::{NativeResources, format::invalid};
use crate::{
    BufferSpan, Result,
    cuda::{Event, Stream},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TensorView {
    pub name: String,
    pub allocation: String,
    #[serde(default)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TensorInput {
    #[serde(flatten)]
    pub view: TensorView,
    pub file: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRequest {
    pub graphs: Vec<String>,
    pub inputs: Vec<TensorInput>,
    pub outputs: Vec<TensorView>,
}

impl TensorRequest {
    pub fn validate(&self) -> Result<()> {
        let mut graphs = HashSet::new();
        if self.graphs.is_empty()
            || self
                .graphs
                .iter()
                .any(|name| name.is_empty() || !graphs.insert(name))
        {
            return Err(invalid(
                "tensor requests require a nonempty sequence of unique graph names",
            ));
        }
        if self.inputs.is_empty() || self.outputs.is_empty() {
            return Err(invalid("tensor requests require inputs and outputs"));
        }
        let mut names = HashSet::new();
        for view in self
            .inputs
            .iter()
            .map(|input| &input.view)
            .chain(&self.outputs)
        {
            view.nbytes()?;
            if !names.insert(&view.name) {
                return Err(invalid(format!("duplicate tensor name {:?}", view.name)));
            }
        }
        for (index, input) in self.inputs.iter().enumerate() {
            if input.file.as_os_str().is_empty() {
                return Err(invalid("input tensor file path is empty"));
            }
            let end = input
                .view
                .offset
                .checked_add(input.view.nbytes()?)
                .ok_or_else(|| invalid("input tensor range overflows"))?;
            for other in &self.inputs[..index] {
                let other_end = other
                    .view
                    .offset
                    .checked_add(other.view.nbytes()?)
                    .ok_or_else(|| invalid("input tensor range overflows"))?;
                if input.view.allocation == other.view.allocation
                    && input.view.offset < other_end
                    && other.view.offset < end
                {
                    return Err(invalid("input tensor allocation ranges overlap"));
                }
            }
        }
        Ok(())
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
    inputs: Vec<TensorView>,
    outputs: Vec<TensorView>,
}

impl TensorRunner {
    pub fn new(owner: NativeResources, request: &TensorRequest) -> Result<Self> {
        request.validate()?;
        for view in request
            .inputs
            .iter()
            .map(|input| &input.view)
            .chain(&request.outputs)
        {
            range_address(
                &owner.allocation(&view.allocation)?,
                view.offset,
                view.nbytes()?,
            )?;
        }
        let bundle = owner.graph_bundle();
        let graphs = request
            .graphs
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
            inputs: request
                .inputs
                .iter()
                .map(|input| input.view.clone())
                .collect(),
            outputs: request.outputs.clone(),
        })
    }

    /// Upload all request inputs and execute every graph once without intergraph
    /// host synchronization. Outputs remain resident until explicitly downloaded.
    pub fn run(&mut self, inputs: &[Vec<u8>]) -> Result<TensorTiming> {
        if inputs.len() != self.inputs.len() {
            return Err(invalid("input tensor count differs from request"));
        }
        for (view, bytes) in self.inputs.iter().zip(inputs) {
            if bytes.len() != view.nbytes()? {
                return Err(invalid(format!(
                    "input tensor {:?} byte extent differs from request",
                    view.name
                )));
            }
        }
        let owner = self.owner.as_ref().unwrap();
        let stream = self.stream.as_ref().unwrap();
        let upload = Instant::now();
        for (view, bytes) in self.inputs.iter().zip(inputs) {
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
        self.outputs
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

    fn request() -> TensorRequest {
        serde_json::from_value(serde_json::json!({
            "graphs": ["encode", "step", "decode"],
            "inputs": [{"name":"noise","allocation":"storage","offset":0,"dtype":"F16","shape":[1,4,8,8],"file":"noise.bin"}],
            "outputs": [{"name":"image","allocation":"storage","offset":512,"dtype":"F32","shape":[1,3,64,64]}]
        })).unwrap()
    }

    #[test]
    fn validates_tensor_bytes_and_safe_output_names() {
        let mut value = request();
        value.validate().unwrap();
        assert_eq!(value.inputs[0].view.nbytes().unwrap(), 512);
        assert_eq!(value.outputs[0].nbytes().unwrap(), 49152);
        value.outputs[0].name = "../image".into();
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_invalid_shape_dtype_alignment_and_overflow() {
        for (shape, dtype, offset) in [
            (vec![], "F16", 0),
            (vec![0], "F16", 0),
            (vec![usize::MAX, 2], "U8", 0),
            (vec![1], "unknown", 0),
            (vec![1], "F32", 2),
        ] {
            let mut value = request();
            value.inputs[0].view.shape = shape;
            value.inputs[0].view.dtype = dtype.into();
            value.inputs[0].view.offset = offset;
            assert!(value.validate().is_err());
        }
    }

    #[test]
    fn rejects_empty_or_duplicate_graphs_and_tensor_names() {
        for graphs in [vec![], vec!["step".into(), "step".into()], vec!["".into()]] {
            let mut value = request();
            value.graphs = graphs;
            assert!(value.validate().is_err());
        }
        let mut value = request();
        value.outputs[0].name = "noise".into();
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_overlapping_input_ranges_but_accepts_adjacent_ranges() {
        let mut value = request();
        let mut second = value.inputs[0].clone();
        second.view.name = "noise2".into();
        second.view.offset = 510;
        value.inputs.push(second);
        assert!(value.validate().is_err());
        value.inputs[1].view.offset = 512;
        value.validate().unwrap();
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
