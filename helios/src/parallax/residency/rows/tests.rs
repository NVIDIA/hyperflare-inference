use super::*;

fn layer() -> Layer {
    Layer {
        layer: 3,
        experts: 3,
        host_experts: vec![1, 2],
        hbm_experts: vec![0, 1],
        planes: [32, 16, 64, 8]
            .into_iter()
            .enumerate()
            .map(|(part, row_bytes)| Plane {
                host_address: 0x10000 + part as u64 * 0x10000,
                hbm_address: 0x80000 + part as u64 * 0x10000,
                row_bytes,
            })
            .collect(),
        mapping_address: 0x1000,
    }
}

#[test]
fn row_maps_preserve_canonical_ids_and_all_planes() {
    let l = Layout::new(vec![layer()]).unwrap();
    assert_eq!(l.maps(&l.initial).unwrap(), vec![vec![2, 3, 1]]);
    assert!(l.backing[..4].iter().all(Option::is_none));
    let mut locations = l.initial.clone();
    locations[4..8].fill(Location::Backing);
    for part in 0..4 {
        locations[8 + part] = Location::Hbm(4 + part);
    }
    assert_eq!(l.maps(&locations).unwrap(), vec![vec![2, 0, 3]]);
    locations[8] = Location::Backing;
    assert!(l.maps(&locations).is_err());
}

#[test]
fn rows_cannot_mix_layers_parts_or_live_slots() {
    let mut second = layer();
    second.layer = 9;
    second.mapping_address += 0x100000;
    for p in &mut second.planes {
        p.host_address += 0x100000;
        p.hbm_address += 0x100000;
    }
    let l = Layout::new(vec![layer(), second]).unwrap();
    assert!(l.accepts(8, 4));
    assert!(!l.accepts(8, 5));
    assert!(!l.accepts(8, 12));
    let mut locations = l.initial.clone();
    for part in 0..4 {
        locations[8 + part] = Location::Hbm(4 + part);
    }
    assert!(
        l.maps(&locations).is_err(),
        "cannot publish an occupied row twice"
    );
    locations[4..8].fill(Location::Backing);
    assert!(l.maps(&locations).is_ok());
    locations[9] = Location::Hbm(1);
    assert!(
        l.maps(&locations).is_err(),
        "one canonical expert cannot combine different physical rows"
    );
}

#[test]
fn invalid_slot_indices_in_any_plane_fail_without_indexing_occupancy() {
    let layout = Layout::new(vec![layer()]).unwrap();
    for part in 0..4 {
        for slot in [layout.slots.len(), usize::MAX] {
            let mut locations = layout.initial.clone();
            locations[4 + part] = Location::Hbm(slot);
            assert!(layout.maps(&locations).is_err());
        }
    }
}

#[test]
fn invalid_geometry_is_rejected_without_cuda() {
    let mut cases = vec![];
    let mut p = layer();
    p.host_experts.push(1);
    cases.push(p);
    let mut p = layer();
    p.host_experts = vec![1];
    cases.push(p);
    let mut p = layer();
    p.planes[1].hbm_address = p.planes[0].hbm_address;
    cases.push(p);
    let mut p = layer();
    p.mapping_address = p.planes[0].hbm_address;
    cases.push(p);
    let mut p = layer();
    p.planes[0].row_bytes = usize::MAX;
    cases.push(p);
    let mut p = layer();
    p.planes[0].host_address = u64::MAX - 3;
    cases.push(p);
    let mut p = layer();
    p.mapping_address += 1;
    cases.push(p);
    for p in cases {
        assert!(Layout::new(vec![p]).is_err());
    }
    assert!(Layout::new(vec![layer(), layer()]).is_err());
}

#[test]
fn map_publication_restores_every_changed_layer_after_partial_failure() {
    let before = vec![vec![2, 3, 1], vec![2, 3, 1]];
    let after = vec![vec![2, 0, 3], vec![2, 0, 3]];
    let changed = [0, 1].into();
    let mut visible = before.clone();
    let mut calls = 0;
    let result = replace_maps(&before, &after, &changed, |maps, layers| {
        calls += 1;
        for &layer in layers {
            visible[layer] = maps[layer].clone();
            if calls == 1 {
                return Err(Error::Backend("partial upload".into()));
            }
        }
        Ok(())
    });
    assert!(matches!(result, Err(Error::Backend(_))));
    assert_eq!(calls, 2);
    assert_eq!(visible, before);
}

#[test]
fn failed_drain_or_rollback_poison_mapping_publication() {
    let before = vec![vec![2, 3, 1]];
    let after = vec![vec![2, 0, 3]];
    for failed_drain in [false, true] {
        let mut calls = 0;
        let result = replace_maps(&before, &after, &[0].into(), |_, _| {
            calls += 1;
            if failed_drain {
                Err(Error::Poisoned("unconfirmed upload".into()))
            } else {
                Err(Error::Backend("upload failed".into()))
            }
        });
        assert!(matches!(result, Err(Error::Poisoned(_))));
        assert_eq!(calls, if failed_drain { 1 } else { 2 });
    }
}

/// CPU-only measurement of the two full-map validations performed by publication.
#[test]
#[ignore = "manual release-mode CPU timing; does not use CUDA"]
fn publication_map_validation_cost() {
    let layers = (0..40).map(|layer| {
        let resident = if layer < 20 { 320 } else { 256 };
        Layer {
            layer, experts: 384,
            host_experts: (resident - 96..384).collect(),
            hbm_experts: (0..resident).collect(),
            mapping_address: 0x1000 + layer as u64 * 0x1000,
            planes: [8192, 512, 4096, 256].into_iter().enumerate().map(|(part, row_bytes)| {
                let base = 0x1_0000_0000 + layer as u64 * 0x1000_0000 + part as u64 * 0x100_0000;
                Plane { host_address: base, hbm_address: base + 0x80_0000, row_bytes }
            }).collect(),
        }
    }).collect();
    let layout = Layout::new(layers).unwrap();
    let mut changed = layout.initial.clone();
    for (index, layer) in layout.layers.iter().enumerate() {
        let victim = layer.host_experts[0];
        let promoted = layer.hbm_experts.len();
        let start = layout.starts[index];
        for part in 0..layer.planes.len() {
            changed[start + promoted * layer.planes.len() + part] =
                changed[start + victim * layer.planes.len() + part];
            changed[start + victim * layer.planes.len() + part] = Location::Backing;
        }
    }
    let expected_before = layout.maps(&layout.initial).unwrap();
    let expected_after = layout.maps(&changed).unwrap();
    assert_ne!(expected_before, expected_after);
    let mut times = vec![];
    for index in 0..110 {
        let started = std::time::Instant::now();
        let before = layout.maps(std::hint::black_box(&layout.initial)).unwrap();
        let after = layout.maps(std::hint::black_box(&changed)).unwrap();
        let elapsed = started.elapsed().as_nanos();
        assert_eq!(before, expected_before);
        assert_eq!(after, expected_after);
        if index >= 10 { times.push(elapsed); }
    }
    times.sort_unstable();
    println!("SYNTHETIC CPU map validation: {} regions, {} slots, 100 samples; pair p50={} ns p95={} ns p99={} ns. Excludes CUDA upload and reader drain.",
        layout.regions.len(), layout.slots.len(), times[49], times[94], times[98]);
}
