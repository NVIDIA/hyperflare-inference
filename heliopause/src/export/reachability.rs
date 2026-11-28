/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conservative retention of graph-reachable allocations with typed numeric leaves.

use std::collections::{HashMap, HashSet, VecDeque};

use super::{Allocation, ExportResult, Kernel, invalid, wire};

pub(crate) fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.retain(|(start, end)| start < end);
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut().filter(|last| start <= last.1) {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

pub(crate) fn subtract_ranges(start: u64, end: u64, excluded: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut cursor = start;
    let mut spans = Vec::new();
    let first = excluded.partition_point(|&(_, stop)| stop <= start);
    for &(begin, stop) in &excluded[first..] {
        if begin >= end {
            break;
        }
        if cursor < begin {
            spans.push((cursor, begin.min(end)));
        }
        cursor = cursor.max(stop.min(end));
    }
    if cursor < end {
        spans.push((cursor, end));
    }
    spans
}

/// A kernel argument value inside the recorded address envelope that resolves
/// to no live recorded allocation, including a value in an inactive range.
/// Such a value may be a stale pointer or a numeric constant; argument typing
/// cannot yet distinguish the two, so export reports these for operator review.
pub(crate) struct UnresolvedCandidate {
    pub graph: String,
    pub node: usize,
    pub function: usize,
    pub argument: usize,
    pub offset: usize,
}

struct UnresolvedEntry {
    first: UnresolvedCandidate,
    order: usize,
    occurrences: usize,
}

/// The call site a kernel argument blob was scanned for. Every occurrence of
/// the same argument bytes is recorded separately so a report can name the
/// graph, node, kernel, and parameter that carried an unresolved candidate.
pub(crate) struct Site<'a> {
    pub graph: &'a str,
    pub node: usize,
    pub function: usize,
    pub argument: usize,
}

/// The number of distinct addresses shown in a rank's report. The examples
/// retain the first call site for each address in graph and node order.
const UNRESOLVED_EXAMPLES: usize = 10;

pub(crate) struct Reachability {
    allocations: HashMap<String, Allocation>,
    intervals: Vec<(u64, u64, String)>,
    numeric: Vec<(u64, u64)>,
    inactive: Vec<(u64, u64)>,
    pub(crate) reached: HashSet<String>,
    pending: VecDeque<String>,
    blobs: HashMap<(Vec<u8>, Vec<(usize, usize)>), Vec<(usize, u64)>>,
    unresolved: HashMap<u64, UnresolvedEntry>,
    tail: Option<(String, u64, Vec<u8>)>,
}

impl Reachability {
    pub(crate) fn new(
        allocations: &[Allocation],
        numeric: &[(u64, u64)],
        inactive: &[(u64, u64)],
        roots: &[String],
    ) -> ExportResult<Self> {
        let mut this = Self {
            allocations: allocations
                .iter()
                .cloned()
                .map(|a| (a.name.clone(), a))
                .collect(),
            intervals: allocations
                .iter()
                .map(|a| (a.address, a.address + a.nbytes, a.name.clone()))
                .collect(),
            numeric: merge_ranges(numeric.to_vec()),
            inactive: merge_ranges(inactive.to_vec()),
            reached: HashSet::new(),
            pending: VecDeque::new(),
            blobs: HashMap::new(),
            unresolved: HashMap::new(),
            tail: None,
        };
        this.intervals.sort_unstable();
        for name in roots {
            this.reference(name)?;
        }
        Ok(this)
    }

    pub(crate) fn execution(&mut self, execution: &wire::Execution) -> ExportResult<()> {
        for buffer in &execution.buffers {
            self.reference(
                &buffer
                    .reference
                    .as_ref()
                    .ok_or_else(|| invalid("execution buffer lacks allocation reference"))?
                    .allocation,
            )?;
        }
        for worker in &execution.persistent_workers {
            if let Some(stop_flag) = &worker.stop_flag {
                self.reference(&stop_flag.allocation)?;
            }
            if let Some(metrics) = &worker.metrics {
                self.reference(&metrics.allocation)?;
            }
        }
        Ok(())
    }

    pub(crate) fn reference(&mut self, name: &str) -> ExportResult<()> {
        if !self.allocations.contains_key(name) {
            return Err(invalid(format!("unregistered graph allocation: {name}")));
        }
        if self.reached.insert(name.to_owned()) {
            self.pending.push_back(name.to_owned());
        }
        Ok(())
    }

    fn candidate(&mut self, value: u64) -> ExportResult<bool> {
        let inactive = self.inactive.partition_point(|&(start, _)| start <= value);
        if inactive > 0 && value < self.inactive[inactive - 1].1 {
            return Ok(true);
        }
        let index = self
            .intervals
            .partition_point(|&(start, _, _)| start <= value);
        if index > 0 && value < self.intervals[index - 1].1 {
            let name = self.intervals[index - 1].2.clone();
            self.reference(&name)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn blob(&mut self, data: &[u8], numeric: &[(usize, usize)], site: &Site) -> ExportResult<()> {
        let key = (data.to_vec(), numeric.to_vec());
        let misses = match self.blobs.get(&key) {
            Some(misses) => misses.clone(),
            None => {
                let excluded = merge_ranges(
                    numeric
                        .iter()
                        .map(|&(start, end)| (start as u64, end as u64))
                        .collect(),
                );
                let mut misses = Vec::new();
                for (start, end) in subtract_ranges(0, data.len() as u64, &excluded) {
                    self.scan(
                        &data[start as usize..end as usize],
                        start as usize,
                        Some(&mut misses),
                    )?;
                }
                self.blobs.insert(key, misses.clone());
                misses
            }
        };
        for (offset, address) in misses {
            let order = self.unresolved.len();
            self.unresolved
                .entry(address)
                .and_modify(|entry| entry.occurrences += 1)
                .or_insert_with(|| UnresolvedEntry {
                    first: UnresolvedCandidate {
                        graph: site.graph.to_owned(),
                        node: site.node,
                        function: site.function,
                        argument: site.argument,
                        offset,
                    },
                    order,
                    occurrences: 1,
                });
        }
        Ok(())
    }

    pub(crate) fn unresolved_count(&self) -> usize {
        self.unresolved
            .values()
            .map(|entry| entry.occurrences)
            .sum()
    }

    pub(crate) fn unresolved_unique_addresses(&self) -> usize {
        self.unresolved.len()
    }

    /// Lines describing the unresolved candidates: a summary line, one line per
    /// distinct address up to the example limit, and a final line when occurrences
    /// outnumber examples. The caller supplies the recorded functions and adds
    /// the export prefix.
    pub(crate) fn unresolved_report(&self, functions: &[wire::Function]) -> Vec<String> {
        let mut lines = Vec::new();
        let occurrences = self.unresolved_count();
        if occurrences == 0 {
            return lines;
        }
        lines.push(format!(
            "kernel argument scan: {} and {} fall inside the recorded address envelope but \
             outside every live recorded allocation, so each may be a stale pointer or a numeric \
             constant",
            with_count(occurrences, "occurrence", "occurrences"),
            with_count(
                self.unresolved_unique_addresses(),
                "distinct address",
                "distinct addresses"
            ),
        ));
        let mut examples: Vec<_> = self.unresolved.iter().collect();
        examples.sort_unstable_by_key(|(_, entry)| entry.order);
        for &(address, entry) in examples.iter().take(UNRESOLVED_EXAMPLES) {
            let address = *address;
            let candidate = &entry.first;
            let function = functions.get(candidate.function);
            lines.push(format!(
                "  graph {:?} node {} kernel {} ({}), argument {}, offset {}: {address:#x}",
                candidate.graph,
                candidate.node,
                function.map_or("unknown", |function| function.symbol.as_str()),
                function.map_or("unknown", |function| function.module.as_str()),
                candidate.argument,
                candidate.offset,
            ));
        }
        let hidden = occurrences - examples.len().min(UNRESOLVED_EXAMPLES);
        if hidden > 0 {
            lines.push(format!(
                "  … {} not shown",
                with_count(hidden, "more occurrence", "more occurrences")
            ));
        }
        lines
    }

    fn scan(
        &mut self,
        data: &[u8],
        base: usize,
        mut misses: Option<&mut Vec<(usize, u64)>>,
    ) -> ExportResult<()> {
        if self.intervals.is_empty() {
            return Ok(());
        }
        let floor = self.intervals[0].0;
        let ceiling = self.intervals.last().unwrap().1;
        for (offset, bytes) in data.windows(8).enumerate() {
            let pointer = u64::from_le_bytes(bytes.try_into().unwrap());
            if pointer >= floor && pointer < ceiling {
                let inactive = self
                    .inactive
                    .partition_point(|&(start, _)| start <= pointer);
                let in_free_block = inactive > 0 && pointer < self.inactive[inactive - 1].1;
                if (!self.candidate(pointer)? || in_free_block)
                    && let Some(misses) = misses.as_deref_mut()
                {
                    misses.push((base + offset, pointer));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn graph(
        &mut self,
        name: &str,
        nodes: &[wire::Node],
        kernels: &[Kernel],
    ) -> ExportResult<()> {
        use wire::node::Operation;
        for (node_index, node) in nodes.iter().enumerate() {
            match node
                .operation
                .as_ref()
                .ok_or_else(|| invalid("graph node lacks operation"))?
            {
                Operation::Kernel(node) => {
                    let kernel = kernels
                        .get(node.function as usize)
                        .ok_or_else(|| invalid("graph function index is out of bounds"))?;
                    if node.arguments.len() != kernel.arguments.len() {
                        return Err(invalid("kernel argument count differs from its function"));
                    }
                    for (position, (blob, argument)) in
                        node.arguments.iter().zip(&kernel.arguments).enumerate()
                    {
                        for relocation in &blob.relocations {
                            self.reference(
                                &relocation
                                    .target
                                    .as_ref()
                                    .ok_or_else(|| invalid("relocation lacks target"))?
                                    .allocation,
                            )?;
                        }
                        self.blob(
                            &blob.data,
                            &argument.numeric,
                            &Site {
                                graph: name,
                                node: node_index,
                                function: node.function as usize,
                                argument: position,
                            },
                        )?;
                    }
                    for attribute in &node.attributes {
                        if let Some(value) = &attribute.value {
                            for relocation in &value.relocations {
                                self.reference(
                                    &relocation
                                        .target
                                        .as_ref()
                                        .ok_or_else(|| {
                                            invalid("attribute relocation lacks target")
                                        })?
                                        .allocation,
                                )?;
                            }
                        }
                    }
                }
                Operation::Copy(copy) => {
                    self.reference(
                        &copy
                            .source
                            .as_ref()
                            .ok_or_else(|| invalid("copy lacks source"))?
                            .allocation,
                    )?;
                    self.reference(
                        &copy
                            .destination
                            .as_ref()
                            .ok_or_else(|| invalid("copy lacks destination"))?
                            .allocation,
                    )?;
                }
                Operation::Memset(memset) => self.reference(
                    &memset
                        .destination
                        .as_ref()
                        .ok_or_else(|| invalid("memset lacks destination"))?
                        .allocation,
                )?,
                Operation::Empty(_) | Operation::EventWait(_) => {}
            }
        }
        Ok(())
    }

    fn scan_spans(&mut self, address: u64, data: &[u8], numeric: bool) -> ExportResult<()> {
        if numeric {
            let end = address
                .checked_add(data.len() as u64)
                .ok_or_else(|| invalid("allocation scan extent overflow"))?;
            for (start, stop) in subtract_ranges(address, end, &self.numeric) {
                self.scan(
                    &data[(start - address) as usize..(stop - address) as usize],
                    0,
                    None,
                )?;
            }
            Ok(())
        } else {
            self.scan(data, 0, None)
        }
    }

    fn scan_contents(
        &mut self,
        name: &str,
        address: u64,
        data: &[u8],
        numeric: bool,
    ) -> ExportResult<()> {
        let mut boundary = [0u8; 14];
        let mut previous = 0;
        if let Some((previous_name, previous_end, tail)) = self.tail.take() {
            if previous_name == name && previous_end == address {
                previous = tail.len();
                boundary[..previous].copy_from_slice(&tail);
            }
        }
        let prefix = data.len().min(7);
        boundary[previous..previous + prefix].copy_from_slice(&data[..prefix]);
        if previous > 0 {
            let start = address
                .checked_sub(previous as u64)
                .ok_or_else(|| invalid("allocation scan boundary underflow"))?;
            self.scan_spans(start, &boundary[..previous + prefix], numeric)?;
        }
        self.scan_spans(address, data, numeric)?;
        let tail = if data.len() >= 7 {
            &data[data.len() - 7..]
        } else {
            &boundary[(previous + prefix).saturating_sub(7)..previous + prefix]
        };
        self.tail = Some((
            name.to_owned(),
            address
                .checked_add(data.len() as u64)
                .ok_or_else(|| invalid("allocation scan extent overflow"))?,
            tail.to_vec(),
        ));
        Ok(())
    }

    pub(crate) fn contents(
        &mut self,
        allocation: &Allocation,
        offset: u64,
        data: &[u8],
    ) -> ExportResult<()> {
        let address = allocation
            .address
            .checked_add(offset)
            .ok_or_else(|| invalid("allocation scan address overflow"))?;
        self.scan_contents(&allocation.name, address, data, true)
    }

    pub(crate) fn opaque_contents(
        &mut self,
        name: &str,
        offset: u64,
        data: &[u8],
    ) -> ExportResult<()> {
        self.scan_contents(name, offset, data, false)
    }

    pub(crate) fn take(&mut self) -> ExportResult<Option<Allocation>> {
        let Some(name) = self.pending.pop_front() else {
            return Ok(None);
        };
        let allocation = self.allocations[&name].clone();
        for relocation in &allocation.relocations {
            self.reference(&relocation.target.allocation)?;
        }
        Ok(Some(allocation))
    }
}

fn with_count(count: usize, one: &str, many: &str) -> String {
    if count == 1 {
        format!("1 {one}")
    } else {
        format!("{count} {many}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn allocation(name: &str, address: u64) -> Allocation {
        Allocation {
            name: name.into(),
            address,
            nbytes: 256,
            alignment: 256,
            initialize: "snapshot".into(),
            relocations: vec![],
            preserve_address: true,
            memory: "device".into(),
            snapshot_spans: None,
            sizing: None,
        }
    }
    #[test]
    fn unaligned_pointer_across_chunks_retains_target() {
        let allocations = [allocation("source", 0x1000), allocation("target", 0x2000)];
        let mut reach = Reachability::new(&allocations, &[], &[], &["source".into()]).unwrap();
        let mut bytes = vec![0; 19];
        bytes[5..13].copy_from_slice(&0x2040u64.to_le_bytes());
        reach.contents(&allocations[0], 0, &bytes[..9]).unwrap();
        reach.contents(&allocations[0], 9, &bytes[9..]).unwrap();
        assert!(reach.reached.contains("target"));
    }
    #[test]
    fn arbitrary_chunk_widths_preserve_pointer_hits_across_numeric_ranges_and_gaps() {
        let mut source = allocation("source", 0x0800_0000_0000);
        source.nbytes = 256;
        let mut allocations = vec![source.clone()];
        for index in 0..6 {
            allocations.push(allocation(
                &format!("target-{index}"),
                0x0800_1000_0000 + index * 0x1000,
            ));
        }
        let mut state = 17u32;
        let mut data = (0..256)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect::<Vec<_>>();
        for (index, offset) in [5, 19, 61, 111, 178, 226].into_iter().enumerate() {
            data[offset..offset + 8]
                .copy_from_slice(&(allocations[index + 1].address + 13).to_le_bytes());
        }
        let numeric = [
            (source.address, source.address + 5),
            (source.address + 16, source.address + 28),
            (source.address + 107, source.address + 114),
        ];
        let expected: HashSet<String> = ["target-0", "target-4", "target-5"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        for width in 1..=31 {
            let mut reach = Reachability::new(&allocations, &numeric, &[], &[]).unwrap();
            for (start, end) in [(0, 59), (65, 170), (176, 256)] {
                for offset in (start..end).step_by(width) {
                    reach
                        .contents(
                            &source,
                            offset as u64,
                            &data[offset..(offset + width).min(end)],
                        )
                        .unwrap();
                }
            }
            assert_eq!(reach.reached, expected, "chunk width {width}");
        }
    }

    #[test]
    fn numeric_leaves_and_inactive_candidates_do_not_retain() {
        let allocations = [
            allocation("source", 0x1000),
            allocation("numeric", 0x2000),
            allocation("inactive", 0x3000),
        ];
        let mut reach =
            Reachability::new(&allocations, &[(0x1000, 0x1008)], &[(0x3000, 0x3100)], &[]).unwrap();
        let bytes = [0x2000u64.to_le_bytes(), 0x3000u64.to_le_bytes()].concat();
        reach.contents(&allocations[0], 0, &bytes).unwrap();
        assert!(reach.reached.is_empty());
        reach.reference("inactive").unwrap();
        assert!(reach.reached.contains("inactive"));
    }
    #[test]
    fn opaque_globals_scan_across_chunks_without_numeric_tensor_classification() {
        let allocations = [allocation("target", 0x2000)];
        let mut reach = Reachability::new(&allocations, &[(0, 32)], &[], &[]).unwrap();
        let mut bytes = vec![0; 15];
        bytes[3..11].copy_from_slice(&0x2020u64.to_le_bytes());
        reach
            .opaque_contents("module-global", 0, &bytes[..7])
            .unwrap();
        reach
            .opaque_contents("module-global", 7, &bytes[7..])
            .unwrap();
        assert!(reach.reached.contains("target"));
    }

    #[test]
    fn numeric_argument_holes_are_conservative() {
        let allocations = [allocation("target", 0x2000)];
        let mut reach = Reachability::new(&allocations, &[], &[], &[]).unwrap();
        let bytes = [0x2000u64.to_le_bytes(), 0x2000u64.to_le_bytes()].concat();
        reach.blob(&bytes, &[(0, 8)], &SITE).unwrap();
        assert!(reach.reached.contains("target"));
    }

    #[test]
    fn numeric_scale_and_adjacent_bytes_never_form_a_pointer() {
        let source = allocation("scale", 0x200a295e5000);
        let target = allocation("unused", 0x200a3f800000);
        let mut reach = Reachability::new(
            &[source.clone(), target.clone()],
            &[(source.address, source.address + 4)],
            &[],
            &[source.name.clone()],
        )
        .unwrap();
        let bytes = target.address.to_le_bytes();
        assert_eq!(f32::from_le_bytes(bytes[..4].try_into().unwrap()), 1.0);
        reach.contents(&source, 0, &bytes).unwrap();
        assert_eq!(reach.reached, HashSet::from([source.name]));
    }

    #[test]
    fn pointer_table_beside_numeric_values_survives_chunk_boundaries() {
        let source = allocation("mixed", 0x200000000000);
        let live = allocation("live", source.address + 256);
        let dead = allocation("dead", source.address + 512);
        let mut reach = Reachability::new(
            &[source.clone(), live.clone(), dead.clone()],
            &[(source.address, source.address + 16)],
            &[],
            &[source.name.clone()],
        )
        .unwrap();
        let bytes = [
            dead.address.to_le_bytes().as_slice(),
            dead.address.to_le_bytes().as_slice(),
            b"padding",
            (live.address + 1).to_le_bytes().as_slice(),
        ]
        .concat();
        reach.contents(&source, 0, &bytes[..26]).unwrap();
        reach.contents(&source, 26, &bytes[26..]).unwrap();
        assert_eq!(reach.reached, HashSet::from([source.name, live.name]));
    }

    #[test]
    fn opaque_arguments_and_disjoint_snapshot_spans_do_not_bridge() {
        let source = allocation("source", 0x200000000000);
        let target = allocation("target", source.address + 256);
        let bytes = target.address.to_le_bytes();
        let mut reach =
            Reachability::new(&[source.clone(), target.clone()], &[], &[], &[]).unwrap();
        reach.blob(&bytes[..4], &[], &SITE).unwrap();
        reach.blob(&bytes[4..], &[], &SITE).unwrap();
        reach.contents(&source, 0, &bytes[..4]).unwrap();
        reach.contents(&source, 128, &bytes[4..]).unwrap();
        assert!(reach.reached.is_empty());
        reach
            .blob(
                &[b"abc".as_slice(), bytes.as_slice(), b"xyz".as_slice()].concat(),
                &[],
                &SITE,
            )
            .unwrap();
        assert_eq!(reach.reached, HashSet::from([target.name]));
    }

    #[test]
    fn typed_scalar_and_pointer_parameters_keep_independent_roles() {
        let scalar = allocation("scalar_bits", 0x200000000000);
        let input = allocation("input", scalar.address + 256);
        let kernel = Kernel {
            function: 1,
            module: "module".into(),
            arguments: vec![
                super::super::Argument {
                    size: 8,
                    relocations: vec![],
                    numeric: vec![(0, 8)],
                },
                super::super::Argument {
                    size: 8,
                    relocations: vec![],
                    numeric: vec![],
                },
            ],
        };
        let mut node = wire::Kernel {
            function: 0,
            arguments: vec![
                wire::Blob {
                    data: scalar.address.to_le_bytes().to_vec(),
                    relocations: vec![],
                },
                wire::Blob {
                    data: input.address.to_le_bytes().to_vec(),
                    relocations: vec![],
                },
            ],
            ..Default::default()
        };
        let mut reach = Reachability::new(&[scalar.clone(), input.clone()], &[], &[], &[]).unwrap();
        reach
            .graph(
                "graph",
                &[wire::Node {
                    operation: Some(wire::node::Operation::Kernel(node.clone())),
                }],
                &[kernel.clone()],
            )
            .unwrap();
        assert_eq!(reach.reached, HashSet::from([input.name.clone()]));
        node.arguments[1].data = node.arguments[0].data.clone();
        reach
            .graph(
                "graph",
                &[wire::Node {
                    operation: Some(wire::node::Operation::Kernel(node)),
                }],
                &[kernel],
            )
            .unwrap();
        assert_eq!(reach.reached, HashSet::from([input.name, scalar.name]));
    }

    fn kernel_graph(argument_data: &[u8], numeric: Vec<(usize, usize)>) -> (wire::Node, Kernel) {
        let kernel = Kernel {
            function: 1,
            module: "module".into(),
            arguments: vec![super::super::Argument {
                size: argument_data.len(),
                relocations: vec![],
                numeric,
            }],
        };
        let node = wire::Node {
            operation: Some(wire::node::Operation::Kernel(wire::Kernel {
                function: 0,
                arguments: vec![wire::Blob {
                    data: argument_data.to_vec(),
                    relocations: vec![],
                }],
                ..Default::default()
            })),
        };
        (node, kernel)
    }

    const SITE: Site = Site {
        graph: "graph",
        node: 0,
        function: 0,
        argument: 0,
    };

    fn first(reach: &Reachability, address: u64) -> &UnresolvedCandidate {
        &reach.unresolved[&address].first
    }

    #[test]
    fn kernel_argument_gap_candidate_is_reported_with_site_context() {
        let low = allocation("low", 0x2000);
        let high = allocation("high", 0x4000);
        let (node, kernel) = kernel_graph(&0x3008u64.to_le_bytes(), vec![]);
        let mut reach = Reachability::new(&[low, high], &[], &[], &[]).unwrap();
        reach.graph("prefill-4096", &[node], &[kernel]).unwrap();
        assert_eq!(reach.unresolved_count(), 1);
        assert_eq!(reach.unresolved.len(), 1);
        let candidate = first(&reach, 0x3008);
        assert_eq!(candidate.graph, "prefill-4096");
        assert_eq!(candidate.node, 0);
        assert_eq!(candidate.function, 0);
        assert_eq!(candidate.argument, 0);
        assert_eq!(candidate.offset, 0);
        assert_eq!(reach.unresolved_unique_addresses(), 1);
    }

    #[test]
    fn unresolved_examples_are_capped_while_occurrences_are_counted() {
        let low = allocation("low", 0x2000);
        let high = allocation("high", 0x4000);
        let data: Vec<u8> = (0..12u64)
            .flat_map(|index| (0x3000 + 8 * index).to_le_bytes())
            .collect();
        let (node, kernel) = kernel_graph(&data, vec![]);
        let mut reach = Reachability::new(&[low, high], &[], &[], &[]).unwrap();
        reach.graph("graph", &[node], &[kernel]).unwrap();
        assert_eq!(reach.unresolved_count(), 12);
        assert_eq!(reach.unresolved.len(), 12);
        assert_eq!(reach.unresolved_unique_addresses(), 12);
        for index in 0..12 {
            let address = 0x3000 + 8 * index as u64;
            assert_eq!(first(&reach, address).graph, "graph");
            assert_eq!(first(&reach, address).offset, 8 * index);
            assert_eq!(reach.unresolved[&address].order, index);
            assert_eq!(reach.unresolved[&address].occurrences, 1);
        }
    }

    #[test]
    fn kernel_argument_candidate_offset_counts_from_the_argument_start() {
        let low = allocation("low", 0x2000);
        let high = allocation("high", 0x4000);
        let data = [
            0x3008u64.to_le_bytes().as_slice(),
            &[0u8; 8],
            0x3010u64.to_le_bytes().as_slice(),
        ]
        .concat();
        let (node, kernel) = kernel_graph(&data, vec![(0, 8)]);
        let mut reach = Reachability::new(&[low, high], &[], &[], &[]).unwrap();
        reach.graph("graph", &[node], &[kernel]).unwrap();
        assert_eq!(reach.unresolved_count(), 1);
        let candidate = first(&reach, 0x3010);
        assert_eq!(candidate.offset, 16);
        assert_eq!(candidate.argument, 0);
        assert_eq!(candidate.node, 0);
    }

    #[test]
    fn shared_argument_bytes_record_one_example_per_address() {
        let low = allocation("low", 0x2000);
        let high = allocation("high", 0x4000);
        let (node, kernel) = kernel_graph(&0x3008u64.to_le_bytes(), vec![]);
        let mut reach = Reachability::new(&[low, high], &[], &[], &[]).unwrap();
        reach
            .graph("graph-a", &[node.clone()], &[kernel.clone()])
            .unwrap();
        reach.graph("graph-b", &[node], &[kernel]).unwrap();
        assert_eq!(reach.unresolved_count(), 2);
        assert_eq!(reach.unresolved.len(), 1);
        assert_eq!(first(&reach, 0x3008).graph, "graph-a");
        assert_eq!(reach.unresolved[&0x3008].occurrences, 2);
        assert_eq!(reach.unresolved_unique_addresses(), 1);
    }

    #[test]
    fn unresolved_report_names_the_kernel_and_its_symbol() {
        let low = allocation("low", 0x2000);
        let high = allocation("high", 0x4000);
        let (node, kernel) = kernel_graph(&0x3008u64.to_le_bytes(), vec![]);
        let mut reach = Reachability::new(&[low, high], &[], &[], &[]).unwrap();
        reach.graph("prefill-8192", &[node], &[kernel]).unwrap();
        let functions = vec![
            wire::Function {
                module: "module".into(),
                symbol: "cublasLt::splitKreduce_kernel".into(),
                parameter_sizes: vec![8],
                attributes: vec![],
            },
            wire::Function {
                module: "wrong-module".into(),
                symbol: "wrong-symbol".into(),
                parameter_sizes: vec![8],
                attributes: vec![],
            },
        ];
        assert_eq!(
            reach.unresolved_report(&functions),
            vec![
                "kernel argument scan: 1 occurrence and 1 distinct address fall inside the \
                 recorded address envelope but outside every live recorded allocation, so each may be \
                 a stale pointer or a numeric constant"
                    .to_owned(),
                "  graph \"prefill-8192\" node 0 kernel cublasLt::splitKreduce_kernel (module), \
                 argument 0, offset 0: 0x3008"
                    .to_owned(),
            ]
        );
    }

    #[test]
    fn unresolved_report_bounds_examples_and_counts_hidden_occurrences() {
        let low = allocation("low", 0x2000);
        let high = allocation("high", 0x4000);
        let data: Vec<u8> = (0..12u64)
            .rev()
            .flat_map(|index| (0x3000 + 8 * index).to_le_bytes())
            .collect();
        let (node, kernel) = kernel_graph(&data, vec![]);
        let mut reach = Reachability::new(&[low, high], &[], &[], &[]).unwrap();
        reach
            .graph("graph-a", &[node.clone()], &[kernel.clone()])
            .unwrap();
        reach.graph("graph-b", &[node], &[kernel]).unwrap();
        let lines = reach.unresolved_report(&[]);
        assert_eq!(lines.len(), 12);
        assert_eq!(
            lines[0],
            "kernel argument scan: 24 occurrences and 12 distinct addresses fall inside the \
             recorded address envelope but outside every live recorded allocation, so each may be a \
             stale pointer or a numeric constant"
        );
        assert!(lines[1].contains("graph \"graph-a\" node 0 kernel unknown (unknown)"));
        assert!(lines[1].ends_with("0x3058"));
        assert!(lines[10].ends_with("0x3010"));
        assert_eq!(lines[11], "  … 14 more occurrences not shown");
    }

    #[test]
    fn unresolved_report_is_empty_without_candidates() {
        let reach = Reachability::new(&[allocation("only", 0x2000)], &[], &[], &[]).unwrap();
        assert!(reach.unresolved_report(&[]).is_empty());
    }

    #[test]
    fn kernel_argument_boundaries_and_external_values_follow_the_recorded_contract() {
        let low = allocation("low", 0x2000);
        let high = allocation("high", 0x4000);
        for (value, reported) in [
            (low.address, false),
            (low.address + 255, false),
            (low.address + 256, true),
            (high.address - 1, true),
            (high.address, false),
            (0x1000, false),
            (high.address + 256, false),
        ] {
            let (node, kernel) = kernel_graph(&value.to_le_bytes(), vec![]);
            let mut reach = Reachability::new(&[low.clone(), high.clone()], &[], &[], &[]).unwrap();
            reach.graph("graph", &[node], &[kernel]).unwrap();
            assert_eq!(
                reach.unresolved.contains_key(&value),
                reported,
                "value {value:#x}"
            );
        }
    }

    #[test]
    fn kernel_argument_inactive_ranges_are_reported_but_numeric_ranges_stay_lenient() {
        let low = allocation("low", 0x2000);
        let high = allocation("high", 0x4000);
        let (node, kernel) = kernel_graph(&0x3008u64.to_le_bytes(), vec![]);
        let mut reach =
            Reachability::new(&[low.clone(), high.clone()], &[], &[(0x3000, 0x3100)], &[]).unwrap();
        reach.graph("graph", &[node], &[kernel]).unwrap();
        assert_eq!(reach.unresolved_count(), 1);
        assert!(reach.unresolved.contains_key(&0x3008));
        let (node, kernel) = kernel_graph(&0x3008u64.to_le_bytes(), vec![(0, 8)]);
        let mut reach = Reachability::new(&[low, high], &[], &[], &[]).unwrap();
        reach.graph("graph", &[node], &[kernel]).unwrap();
        assert!(reach.unresolved.is_empty());
    }

    #[test]
    fn allocation_contents_still_tolerate_gap_candidates() {
        let low = allocation("low", 0x2000);
        let high = allocation("high", 0x4000);
        let mut reach = Reachability::new(&[low.clone(), high], &[], &[], &[]).unwrap();
        reach.contents(&low, 0, &0x3008u64.to_le_bytes()).unwrap();
        assert!(reach.reached.is_empty());
        assert!(reach.unresolved.is_empty());
    }

    #[test]
    fn freed_interval_boundaries_report_kernel_candidates_without_retaining_storage() {
        let mut storage = allocation("segment", 0x200000000000);
        storage.nbytes = 4096;
        for (offset, retained) in [
            (0, true),
            (1023, true),
            (1024, false),
            (2048, false),
            (3071, false),
            (3072, true),
        ] {
            let mut reach = Reachability::new(
                &[storage.clone()],
                &[],
                &[(storage.address + 1024, storage.address + 3072)],
                &[],
            )
            .unwrap();
            reach
                .blob(
                    &[
                        b"unaligned".as_slice(),
                        (storage.address + offset).to_le_bytes().as_slice(),
                    ]
                    .concat(),
                    &[],
                    &SITE,
                )
                .unwrap();
            assert_eq!(!reach.reached.is_empty(), retained);
            assert_eq!(
                reach.unresolved.contains_key(&(storage.address + offset)),
                !retained,
            );
            reach.reference(&storage.name).unwrap();
            assert!(reach.reached.contains(&storage.name));
        }
    }

    #[test]
    fn unaligned_numeric_token_ids_do_not_retain_address_shaped_bits() {
        let mut source = allocation("indices", 0x200000000000);
        source.nbytes = 256 * 8;
        let target = allocation("unrelated", 50 << 16);
        let bytes = (0..256u64).flat_map(u64::to_le_bytes).collect::<Vec<_>>();
        let mut raw = Reachability::new(&[source.clone(), target.clone()], &[], &[], &[]).unwrap();
        raw.contents(&source, 0, &bytes).unwrap();
        assert!(raw.reached.contains(&target.name));
        let mut typed = Reachability::new(
            &[source.clone(), target],
            &[(source.address, source.address + source.nbytes)],
            &[],
            &[],
        )
        .unwrap();
        typed.contents(&source, 0, &bytes).unwrap();
        assert!(typed.reached.is_empty());
    }

    #[test]
    fn persistent_worker_execution_roots_retain_stop_flag_allocation() {
        let mut stop_alloc = allocation("host-stop", 0x0c00_0000_0000);
        stop_alloc.memory = "mapped_host".into();
        stop_alloc.nbytes = 4096;
        let execution = wire::Execution {
            persistent_workers: vec![wire::PersistentWorker {
                worker_sm_count: 8,
                graph: "worker_launch".into(),
                stop_flag: Some(wire::Reference {
                    allocation: "host-stop".into(),
                    offset: 0,
                }),
                metrics: None,
                metrics_count: 0,
            }],
            ..Default::default()
        };
        let mut reach = Reachability::new(&[stop_alloc], &[], &[], &[]).unwrap();
        reach.execution(&execution).unwrap();
        assert!(reach.reached.contains("host-stop"));
    }

    #[test]
    fn persistent_worker_descriptors_retain_mapped_host_planes_when_not_numeric() {
        let mut host_planes = Vec::new();
        for (idx, label) in ["plane-a", "plane-b", "plane-c", "plane-d"]
            .into_iter()
            .enumerate()
        {
            let mut alloc = allocation(
                &format!("host-{label}"),
                0x0c00_0000_0000 + idx as u64 * 0x10000,
            );
            alloc.memory = "mapped_host".into();
            alloc.nbytes = 4096;
            host_planes.push(alloc);
        }
        let mut worker_descs = allocation("worker_descs", 0x0800_0000_0000);
        worker_descs.nbytes = 192;
        let mut control_table = allocation("control_table", 0x0800_0001_0000);
        control_table.nbytes = 192;

        let mut allocations = host_planes.clone();
        allocations.push(worker_descs.clone());
        allocations.push(control_table.clone());

        let mut desc_bytes = vec![0u8; 192];
        for (idx, plane) in host_planes.iter().enumerate() {
            desc_bytes[idx * 8..(idx + 1) * 8].copy_from_slice(&plane.address.to_le_bytes());
        }
        let mut ctrl_bytes = vec![0u8; 192];
        for (idx, plane) in host_planes.iter().enumerate() {
            let off = 8 + idx * 8;
            ctrl_bytes[off..off + 8].copy_from_slice(&plane.address.to_le_bytes());
        }

        // Omitting pointer-bearing descriptor allocations from numeric ranges retains every mapped-host plane.
        let mut worker_reach =
            Reachability::new(&allocations, &[], &[], &["worker_descs".into()]).unwrap();
        worker_reach.contents(&worker_descs, 0, &desc_bytes).unwrap();
        let mut serve_reach =
            Reachability::new(&allocations, &[], &[], &["control_table".into()]).unwrap();
        serve_reach.contents(&control_table, 0, &ctrl_bytes).unwrap();
        for plane in &host_planes {
            assert!(
                worker_reach.reached.contains(&plane.name),
                "worker descriptor missed {}",
                plane.name
            );
            assert!(
                serve_reach.reached.contains(&plane.name),
                "control table missed {}",
                plane.name
            );
        }

        // Classifying pointer-bearing descriptor allocations as numeric drops every mapped-host plane.
        let numeric = [
            (worker_descs.address, worker_descs.address + worker_descs.nbytes),
            (
                control_table.address,
                control_table.address + control_table.nbytes,
            ),
        ];
        let mut dropped_worker =
            Reachability::new(&allocations, &numeric, &[], &["worker_descs".into()]).unwrap();
        dropped_worker.contents(&worker_descs, 0, &desc_bytes).unwrap();
        let mut dropped_serve =
            Reachability::new(&allocations, &numeric, &[], &["control_table".into()]).unwrap();
        dropped_serve.contents(&control_table, 0, &ctrl_bytes).unwrap();
        for plane in &host_planes {
            assert!(!dropped_worker.reached.contains(&plane.name));
            assert!(!dropped_serve.reached.contains(&plane.name));
        }
    }
}
