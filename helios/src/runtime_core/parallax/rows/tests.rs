use super::*;

fn layers() -> Vec<Layer> {
    [3, 7]
        .into_iter()
        .map(|layer| Layer {
            layer,
            experts: 5,
            host_experts: vec![1, 2, 3, 4],
            hbm_experts: vec![0, 1, 2, 3],
            planes: vec![],
            mapping_address: 0,
        })
        .collect()
}
fn canonical(layers: &[Layer]) -> Vec<ExpertId> {
    layers
        .iter()
        .flat_map(|l| {
            (0..l.experts).map(move |expert| ExpertId {
                layer: l.layer,
                expert,
            })
        })
        .collect()
}

#[test]
fn startup_spares_retire_only_backed_rows_in_each_layer() {
    let layers = layers();
    let ids = canonical(&layers);
    let evict = spare_experts(&layers, &ids, 1).unwrap();
    assert_eq!(
        evict,
        [
            ExpertId {
                layer: 3,
                expert: 3
            },
            ExpertId {
                layer: 7,
                expert: 3
            }
        ]
    );
    let evict = spare_experts(&layers, &ids, 2).unwrap();
    assert_eq!(evict.len(), 4);
    assert!(evict.iter().all(|e| e.expert == 2 || e.expert == 3));
    for count in [0, 3, usize::MAX] {
        assert!(spare_experts(&layers, &ids, count).is_err());
    }
    let mut incomplete = ids.clone();
    incomplete.pop();
    assert!(spare_experts(&layers, &incomplete, 1).is_err());
    let mut duplicate = ids.clone();
    duplicate[1] = duplicate[0];
    assert!(spare_experts(&layers, &duplicate, 1).is_err());
    let mut pinned = layers.clone();
    pinned[1].host_experts = vec![4];
    assert!(spare_experts(&pinned, &ids, 1).is_err());
}

#[test]
fn residency_configuration_preserves_virtual_memory_and_distinguishes_rows() {
    let ids = canonical(&layers());
    let rows = serde_json::json!({"version": 2, "package_id": "00".repeat(32), "catalog_id": "test",
        "canonical": ids, "spare_rows_per_layer": 1});
    let parsed: ResidencyOptions = serde_json::from_value(rows.clone()).unwrap();
    assert!(matches!(parsed, ResidencyOptions::Rows(_)));
    assert_eq!(parsed.canonical(), ids);
    assert_eq!(parsed.catalog_id(), "test");
    let legacy = serde_json::json!({"version": 1, "catalog_id": "test", "canonical": ids,
        "storage": {"version": 1, "package_id": "00".repeat(32), "rank": 0, "planes": []},
        "hbm_slots": 2, "hbm_bytes": 100, "backing_bytes": 200, "initial_hbm": []});
    let parsed: ResidencyOptions = serde_json::from_value(legacy.clone()).unwrap();
    assert!(matches!(parsed, ResidencyOptions::VirtualMemory(_)));
    assert_eq!(serde_json::to_value(parsed).unwrap(), legacy);
    let mut mixed = rows;
    mixed["hbm_slots"] = 2.into();
    assert!(serde_json::from_value::<ResidencyOptions>(mixed).is_err());
}
