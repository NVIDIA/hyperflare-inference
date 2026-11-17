//! Exercise real DMA and fixed captured addresses; this does not qualify a MoE kernel.
use super::*;
use crate::runtime_core::driver::{CopyEndpoint, CopyParams};
use std::{
    ptr,
    time::{Duration, Instant},
};

struct Harness {
    driver: Driver,
    context: Handle,
    device: i32,
    allocations: Vec<u64>,
    hosts: Vec<cuda::PinnedHost>,
    graph: Handle,
    executable: Handle,
}
impl Harness {
    fn new() -> Self {
        let driver = Driver::load().unwrap();
        let (mut device, mut context) = (-1, ptr::null_mut());
        unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            driver
                .check("cuDeviceGet", (driver.device)(&mut device, 0))
                .unwrap();
            driver
                .check(
                    "cuCtxCreate",
                    (driver.context_create)(&mut context, 0, device),
                )
                .unwrap();
        }
        Self {
            driver,
            context,
            device,
            allocations: vec![],
            hosts: vec![],
            graph: ptr::null_mut(),
            executable: ptr::null_mut(),
        }
    }
    fn allocate(&mut self, bytes: usize) -> u64 {
        let mut address = 0;
        self.driver
            .check("cuMemAlloc", unsafe {
                (self.driver.allocate)(&mut address, bytes)
            })
            .unwrap();
        self.allocations.push(address);
        address
    }
    fn upload(&self, address: u64, bytes: &[u8]) {
        self.driver
            .check("cuMemcpyHtoD", unsafe {
                (self.driver.upload)(address, bytes.as_ptr().cast(), bytes.len())
            })
            .unwrap();
    }
    fn download(&self, address: u64, bytes: usize) -> Vec<u8> {
        let mut result = vec![0; bytes];
        self.driver
            .check("cuMemcpyDtoH", unsafe {
                (self.driver.download)(result.as_mut_ptr().cast(), address, bytes)
            })
            .unwrap();
        result
    }
    fn capture(&mut self, ranges: &[(u64, usize)], output: u64) {
        self.driver
            .check("cuGraphCreate", unsafe {
                (self.driver.graph_create)(&mut self.graph, 0)
            })
            .unwrap();
        let mut offset = 0;
        for &(address, bytes) in ranges {
            let params = CopyParams {
                source: CopyEndpoint {
                    memory_type: 2,
                    device: address,
                    ..Default::default()
                },
                destination: CopyEndpoint {
                    memory_type: 2,
                    device: output + offset,
                    ..Default::default()
                },
                width: bytes,
                height: 1,
                depth: 1,
            };
            let mut node = ptr::null_mut();
            self.driver
                .check("cuGraphAddMemcpyNode", unsafe {
                    (self.driver.graph_copy)(
                        &mut node,
                        self.graph,
                        ptr::null(),
                        0,
                        &params,
                        self.context,
                    )
                })
                .unwrap();
            offset += bytes as u64;
        }
        self.driver
            .check("cuGraphInstantiate", unsafe {
                (self.driver.graph_instantiate)(&mut self.executable, self.graph, 0)
            })
            .unwrap();
    }
}
impl Drop for Harness {
    fn drop(&mut self) {
        unsafe {
            if (self.driver.context_set)(self.context) != 0 || (self.driver.context_sync)() != 0 {
                std::mem::forget(std::mem::take(&mut self.hosts));
                return;
            }
            self.hosts.clear();
            if !self.executable.is_null() {
                (self.driver.executable_destroy)(self.executable);
            }
            if !self.graph.is_null() {
                (self.driver.graph_destroy)(self.graph);
            }
            for address in &self.allocations {
                (self.driver.free)(*address);
            }
            (self.driver.context_destroy)(self.context);
        }
    }
}
fn expert(expert: usize) -> ExpertId {
    ExpertId { layer: 3, expert }
}
fn plan(generation: u64, promote: Vec<usize>, evict: Vec<usize>) -> Plan {
    Plan {
        version: 1,
        catalog_id: "synthetic-rows".into(),
        generation,
        valid_until_ns: u64::MAX,
        promote: promote.into_iter().map(expert).collect(),
        evict: evict.into_iter().map(expert).collect(),
    }
}
fn settle(controller: &mut Controller<Rows>) -> Progress {
    let end = Instant::now() + Duration::from_secs(5);
    loop {
        // This test owns the context and excludes any other graph submitter.
        let result = unsafe { controller.progress(0) }.unwrap();
        if result != Progress::WaitingForTransfer {
            return result;
        }
        assert!(Instant::now() < end, "expert row DMA did not complete");
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn payload(expert: usize, part: usize, bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|i| (i + 31 * expert + 17 * part) as u8)
        .collect()
}

// Make transfer completion precede the executor's next readiness query. This
// controls only timing; staging, publication and reader ownership use Rows.
struct CompletedRows(Rows);
unsafe impl Backend for CompletedRows {
    fn regions(&self) -> &[Region] { self.0.regions() }
    fn initial_location(&self, region: usize) -> Location { self.0.initial_location(region) }
    fn has_backing(&self, region: usize) -> bool { self.0.has_backing(region) }
    fn region_group(&self, region: usize) -> usize { self.0.region_group(region) }
    fn slot_group(&self, slot: usize) -> usize { self.0.slot_group(slot) }
    fn slots(&self) -> usize { self.0.slots() }
    fn slot_bytes(&self) -> usize { self.0.slot_bytes() }
    fn slot_capacity(&self, slot: usize) -> usize { self.0.slot_capacity(slot) }
    fn slot_accepts(&self, region: usize, slot: usize) -> bool { self.0.slot_accepts(region, slot) }
    fn backing_bytes(&self) -> usize { self.0.backing_bytes() }
    fn transfer_bytes(&self, region: usize) -> usize { self.0.transfer_bytes(region) }
    fn capabilities(&self) -> Capabilities { self.0.capabilities() }
    fn address(&self, region: usize) -> u64 { self.0.address(region) }
    fn stage(&mut self, copies: &[(usize, usize)]) -> Result<()> {
        self.0.stage(copies)?;
        self.0.stream.synchronize().map_err(|e| Error::Poisoned(e.to_string()))
    }
    fn ready(&mut self) -> Result<bool> { self.0.ready() }
    fn copy_elapsed_ns(&self) -> Result<Option<u64>> { self.0.copy_elapsed_ns() }
    fn discard(&mut self) -> Result<()> { self.0.discard() }
    unsafe fn publish(&mut self, bindings: &[Binding]) -> Result<()> {
        unsafe { self.0.publish(bindings) }
    }
}

#[test]
#[ignore = "requires an explicitly assigned idle GPU and FLARE_RUN_CUDA_TESTS=1"]
fn residency_publication_cannot_revoke_an_admitted_graph_submission() {
    use crate::execution::*;
    use super::super::execution::ExecutionBinding;

    assert_eq!(std::env::var("FLARE_RUN_CUDA_TESTS").as_deref(), Ok("1"));
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").expect("select the assigned GPU UUID");
    assert!(visible.starts_with("GPU-") && !visible.contains(','));
    let mut h = Harness::new();
    let stream = cuda::Stream::open(h.context as usize, h.device).unwrap();
    let mut host = stream.mapped_host_buffer(64).unwrap();
    host.bytes_mut().fill(19);
    let weights = h.allocate(64);
    h.upload(weights, host.bytes());
    let mapping = h.allocate(4);
    h.upload(mapping, &1i32.to_ne_bytes());
    let backend = unsafe { Rows::new(h.context as usize, h.device, vec![Layer {
        layer: 3, experts: 1, host_experts: vec![0], hbm_experts: vec![0],
        mapping_address: mapping, planes: vec![Plane {
            host_address: stream.mapped_address(&host).unwrap() as u64,
            hbm_address: weights, row_bytes: 64,
        }],
    }]) }.unwrap();
    h.hosts.push(host);
    let binding = ExecutionBinding::new(
        Controller::new("synthetic-rows".into(), CompletedRows(backend)).unwrap());
    let output = h.allocate(4);
    h.capture(&[(mapping, 4)], output);
    let mut buffers = Vec::new();
    for (name, dtype, shape, strides, bytes) in [
        ("frame_index", Dtype::Int32, vec![1], vec![1], 4),
        ("submission_sequence", Dtype::Int64, vec![1], vec![1], 8),
        ("completion_records", Dtype::Int64, vec![2, 1, 24], vec![24, 24, 1], 384),
    ] {
        let address = h.allocate(bytes);
        h.upload(address, &vec![0; bytes]);
        buffers.push(BufferDescriptor { name: name.into(), address: address as usize,
            nbytes: bytes, dtype, shape, strides });
    }
    let manifest = ExecutionManifest {
        abi_version: ABI_VERSION, completion_token_capacity: 8,
        context: h.context as usize, device: h.device, max_active: 1,
        input_frame_bytes: 16, buffers, residency: Some(binding.clone()),
        engram: None, collectives: None, cache: None, offload: None, tuning_graphs: vec![],
        graphs: vec![GraphDescriptor {
            name: "read-expert-map".into(), kind: GraphKind::Decode,
            sampling_mode: SamplingMode::General, handle: h.executable as usize,
            batch_capacity: 1, token_capacity: 1, prefill_capacity: 0,
            draft_tokens: 0, packed_token_capacity: 0, confidence_width: 0,
            estimated_duration_ns: 1_000, tuned_bindings: vec![],
            input_bindings: vec![
                InputBinding { buffer_name: "frame_index".into(), host_offset: 0, nbytes: 4 },
                InputBinding { buffer_name: "submission_sequence".into(), host_offset: 8, nbytes: 8 },
            ],
        }],
    };
    let mut executor = unsafe { GraphExecutor::new(manifest) }.unwrap();
    let first = executor.submit(0, &[], false).unwrap().unwrap();
    binding.submit(plan(0, vec![], vec![0])).unwrap();
    assert!(executor.can_submit(false));
    let reserved = executor.next_fence().unwrap();
    assert_eq!(executor.submit(0, &[], false).unwrap(), Some(reserved),
        "a ready movement must not revoke scheduler admission inside submit");
    assert_eq!(executor.outstanding(), 2);
    let deadline = Instant::now() + Duration::from_secs(5);
    for fence in [first, reserved] {
        loop {
            assert!(Instant::now() < deadline);
            if let Some(completion) = executor.poll().unwrap() {
                assert_eq!(completion.fence, fence);
                break;
            }
        }
        assert!(!executor.can_submit(false));
        assert_eq!(h.download(mapping, 4), 1i32.to_ne_bytes());
    }
    assert_eq!(h.download(output, 4), 1i32.to_ne_bytes());
    assert!(executor.poll().unwrap().is_none());
    assert_eq!(binding.snapshot().unwrap().generation, 1);
    assert_eq!(h.download(mapping, 4), 0i32.to_ne_bytes());
    assert!(executor.can_submit(false));
    let resumed = executor.submit(0, &[], false).unwrap().unwrap();
    loop {
        assert!(Instant::now() < deadline);
        if let Some(completion) = executor.poll().unwrap() {
            assert_eq!(completion.fence, resumed);
            break;
        }
    }
    assert_eq!(h.download(output, 4), 0i32.to_ne_bytes());
    assert!(executor.close().drained);
    unsafe { binding.close() }.unwrap();
}

#[test]
#[ignore = "requires an explicitly assigned idle GPU and FLARE_RUN_CUDA_TESTS=1"]
fn row_dma_preserves_captured_addresses_and_reader_gated_mapping() {
    assert_eq!(std::env::var("FLARE_RUN_CUDA_TESTS").as_deref(), Ok("1"));
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").expect("select the assigned GPU UUID");
    assert!(visible.starts_with("GPU-") && !visible.contains(','));
    for sizes in [[37, 17, 65, 9], [11_796_480, 737_280, 5_898_240, 368_640]] {
        let mut h = Harness::new();
        let stream = cuda::Stream::open(h.context as usize, h.device).unwrap();
        let mut planes = vec![];
        for (part, bytes) in sizes.into_iter().enumerate() {
            let mut host = stream.mapped_host_buffer(bytes * 3).unwrap();
            for row in 0..3 {
                host.bytes_mut()[row * bytes..(row + 1) * bytes].copy_from_slice(&payload(
                    row + 1,
                    part,
                    bytes,
                ));
            }
            let address = h.allocate(bytes * 3);
            for row in 0..3 {
                h.upload(address + (row * bytes) as u64, &payload(row, part, bytes));
            }
            planes.push(Plane {
                host_address: stream.mapped_address(&host).unwrap() as u64,
                hbm_address: address,
                row_bytes: bytes,
            });
            h.hosts.push(host);
        }
        let mapping = h.allocate(16);
        h.upload(
            mapping,
            &[3i32, 4, 5, 2]
                .into_iter()
                .flat_map(i32::to_ne_bytes)
                .collect::<Vec<_>>(),
        );
        let layer = Layer {
            layer: 3,
            experts: 4,
            host_experts: vec![1, 2, 3],
            hbm_experts: vec![0, 1, 2],
            planes: planes.clone(),
            mapping_address: mapping,
        };
        let mut wrong_order = layer.clone();
        wrong_order.hbm_experts.swap(0, 1);
        assert!(matches!(
            unsafe { Rows::new(h.context as usize, h.device, vec![wrong_order]) },
            Err(Error::Invalid(_))
        ));
        let backend = unsafe { Rows::new(h.context as usize, h.device, vec![layer]) }.unwrap();
        assert!(!backend.capabilities().cold_moe_execution_qualified);
        let mut controller = Controller::new("synthetic-rows".into(), backend).unwrap();
        assert_eq!(controller.snapshot().free_slots, 0);
        let constraints = controller.placement_constraints();
        assert!(!constraints[&expert(0)].evictable);
        assert!(constraints[&expert(1)].evictable);
        assert!(constraints.values().all(|c| c.group == 3));
        assert!(matches!(
            controller.begin(plan(0, vec![3], vec![0]), 0),
            Err(Error::Invalid(_))
        ));
        // Retire one cached copy to make a complete spare row bundle.
        controller.begin(plan(0, vec![], vec![2]), 0).unwrap();
        assert_eq!(settle(&mut controller), Progress::Committed(1));
        assert_eq!(controller.snapshot().free_slots, 4);
        assert_eq!(controller.snapshot().free_slots_by_group, [(3, 4)].into());
        let mut ranges = vec![(mapping, 16)];
        ranges.extend(planes.iter().map(|p| (p.hbm_address, p.row_bytes * 3)));
        let output_bytes = ranges.iter().map(|(_, n)| n).sum();
        let output = h.allocate(output_bytes);
        h.capture(&ranges, output);
        stream.launch(h.executable as usize).unwrap();
        stream.synchronize().unwrap();
        let mut free_before = 0;
        let mut total = 0;
        h.driver
            .check("cuMemGetInfo", unsafe {
                (h.driver.memory_info)(&mut free_before, &mut total)
            })
            .unwrap();
        let mut resident = 1;
        let mut generation = 1;
        let mut expected_map = vec![3i32, 4, 1, 2];
        for iteration in 0..20 {
            let promote = if resident == 1 { 3 } else { 1 };
            let reader = controller.acquire_reader().unwrap();
            stream.launch(h.executable as usize).unwrap();
            stream.synchronize().unwrap();
            controller
                .begin(plan(generation, vec![promote], vec![resident]), 0)
                .unwrap();
            assert_eq!(settle(&mut controller), Progress::WaitingForReaders);
            assert_eq!(
                h.download(mapping, 16),
                expected_map
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>()
            );
            unsafe {
                reader.complete();
            }
            generation += 1;
            assert_eq!(settle(&mut controller), Progress::Committed(generation));
            expected_map[resident] = if resident == 1 { 0 } else { 2 };
            expected_map[promote] = if iteration % 2 == 0 { 5 } else { 4 };
            resident = promote;
            let reader = controller.acquire_reader().unwrap();
            stream.launch(h.executable as usize).unwrap();
            stream.synchronize().unwrap();
            unsafe {
                reader.complete();
            }
            let image = h.download(output, output_bytes);
            assert_eq!(
                image[..16],
                expected_map
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>()
            );
            let mut offset = 16;
            for (part, plane) in planes.iter().enumerate() {
                let row = expected_map[resident] as usize - 3;
                assert_eq!(
                    image[offset + row * plane.row_bytes..offset + (row + 1) * plane.row_bytes],
                    payload(resident, part, plane.row_bytes)
                );
                assert_eq!(
                    image[offset..offset + plane.row_bytes],
                    payload(0, part, plane.row_bytes)
                );
                offset += plane.row_bytes * 3;
            }
            assert_eq!(controller.snapshot().free_slots, 4);
        }
        let mut free_after = 0;
        h.driver
            .check("cuMemGetInfo", unsafe {
                (h.driver.memory_info)(&mut free_after, &mut total)
            })
            .unwrap();
        assert_eq!(
            free_before, free_after,
            "row exchange must not allocate device memory"
        );
        assert_eq!(
            controller.statistics().staged_bytes,
            20 * sizes.iter().sum::<usize>() as u64
        );
        for (part, host) in h.hosts.iter().enumerate() {
            for row in 0..3 {
                assert_eq!(
                    host.bytes()[row * sizes[part]..(row + 1) * sizes[part]],
                    payload(row + 1, part, sizes[part])
                );
            }
        }
        println!(
            "row bytes {:?}: 20 exact exchanges, pinned/backing bytes unchanged, fixed graph and allocation",
            sizes
        );
        let binding = execution::ExecutionBinding::new(controller);
        assert_eq!(binding.snapshot().unwrap().generation, generation);
        drop(binding);
        drop(stream);
    }
}
