use super::*;
use crate::runtime_core::wire as runtime;
use prost::Message;

fn fixture() -> wire::RankBody {
    let allocations = [
        ("host", 0x10000, 48, 1),
        ("hbm", 0x20000, 48, 0),
        ("map", 0x30000, 12, 0),
    ]
    .into_iter()
    .map(|(name, address, nbytes, memory)| wire::Allocation {
        name: name.into(),
        address: Some(address),
        nbytes,
        memory,
        ..Default::default()
    })
    .collect::<Vec<_>>();
    let initializers = allocations
        .iter()
        .map(|a| wire::Initializer {
            allocation: a.name.clone(),
            source: Some(wire::PayloadUse {
                role: if a.name == "map" {
                    wire::PayloadRole::InitialState
                } else {
                    wire::PayloadRole::Weights
                } as i32,
                slice: Some(wire::PayloadSlice {
                    nbytes: a.nbytes,
                    ..Default::default()
                }),
            }),
            ..Default::default()
        })
        .collect();
    let views = ["host", "hbm"]
        .into_iter()
        .enumerate()
        .map(|(allocation, name)| wire::PreparedWeightView {
            name: name.into(),
            allocation: allocation as u64,
            dtype: "uint8".into(),
            element_bytes: 1,
            shape: vec![2, 4, 6],
            stride: vec![24, 1, 4],
            nbytes: 48,
            ..Default::default()
        })
        .collect();
    wire::RankBody {
        allocations,
        initializers,
        prepared_weights: Some(wire::PreparedWeightCatalog {
            version: 1,
            views,
            ..Default::default()
        }),
        execution: Some(runtime::Execution {
            buffers: vec![runtime::Buffer {
                name: "map".into(),
                dtype: "int32".into(),
                shape: vec![3],
                strides: vec![1],
                nbytes: 12,
                reference: Some(runtime::Reference {
                    allocation: "map".into(),
                    offset: 0,
                }),
            }],
            expert_rows: Some(runtime::ExpertRowCatalog {
                version: 1,
                layers: vec![runtime::ExpertRowLayer {
                    layer: 7,
                    experts: 3,
                    host_experts: vec![1, 2],
                    hbm_experts: vec![0, 1],
                    mapping_buffer: "map".into(),
                    planes: vec![runtime::ExpertRowPlane {
                        host_weight_view: "host".into(),
                        hbm_weight_view: "hbm".into(),
                    }],
                }],
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn row_geometry_resolves_transposed_planes_and_compressed_catalogs() {
    let mut body = fixture();
    for compressed in [false, true] {
        if compressed {
            let bytes = body.prepared_weights.take().unwrap().encode_to_vec();
            body.compressed_prepared_weights = Some(wire::CompressedPreparedWeightCatalog {
                zstd: zstd::stream::encode_all(bytes.as_slice(), 3).unwrap(),
                decoded_nbytes: bytes.len() as u64,
            });
        }
        let encoded = body.encode_to_vec();
        let restored = wire::RankBody::decode(encoded.as_slice()).unwrap();
        let layers = resolve_body(&restored).unwrap();
        assert_eq!(layers[0].host_experts, [1, 2]);
        assert_eq!(layers[0].hbm_experts, [0, 1]);
        assert_eq!(layers[0].planes[0].row_bytes, 24);
        assert_eq!(layers[0].planes[0].host_address, 0x10000);
        assert_eq!(layers[0].planes[0].hbm_address, 0x20000);
        let layout = Layout::new(layers).unwrap();
        assert_eq!(layout.maps(&layout.initial).unwrap(), [vec![2, 3, 1]]);
    }
}

#[test]
fn missing_declaration_preserves_wire_compatibility_but_cannot_enable_movement() {
    let execution = runtime::Execution::decode(&[][..]).unwrap();
    assert!(execution.expert_rows.is_none());
    assert!(execution.encode_to_vec().is_empty());
    let mut body = fixture();
    body.execution.as_mut().unwrap().expert_rows = None;
    assert!(resolve_body(&body).is_err());
}

#[test]
fn unsafe_row_bindings_fail_before_cuda() {
    for fault in 0..18 {
        let mut body = fixture();
        let execution = body.execution.as_mut().unwrap();
        let rows = execution.expert_rows.as_mut().unwrap();
        match fault {
            0 => rows.version = 2,
            1 => rows.layers[0].host_experts[1] = 1,
            2 => rows.layers[0].planes[0].host_weight_view = "missing".into(),
            3 => execution.buffers[0].dtype = "float32".into(),
            4 => execution.buffers[0].reference.as_mut().unwrap().offset = 4,
            5 => body.allocations[0].memory = 0,
            6 => body.allocations[1].nbytes = 47,
            7 => body.prepared_weights.as_mut().unwrap().views[1].stride = vec![24, 6, 1],
            8 => {
                for view in &mut body.prepared_weights.as_mut().unwrap().views {
                    view.stride[2] = 3;
                }
            }
            9 => {
                body.initializers[0].source.as_mut().unwrap().role =
                    wire::PayloadRole::InitialState as i32
            }
            10 => {
                body.initializers[1]
                    .source
                    .as_mut()
                    .unwrap()
                    .slice
                    .as_mut()
                    .unwrap()
                    .nbytes = 47
            }
            11 => body.allocations[1]
                .relocations
                .push(runtime::Relocation::default()),
            12 => body.allocations[1].address = Some(0x10000),
            13 => rows.layers.push(rows.layers[0].clone()),
            14 => rows.layers[0].hbm_experts[0] = 3,
            15 => execution.graphs.push(runtime::ExecutionGraph {
                input_bindings: vec![runtime::InputBinding {
                    buffer_name: "map".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            16 => body.allocations[0].address = Some(u64::MAX - 3),
            17 => body.initializers.retain(|i| i.allocation != "map"),
            _ => unreachable!(),
        }
        assert!(resolve_body(&body).is_err(), "fault {fault}");
    }
}
