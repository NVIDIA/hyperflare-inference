//! Validate canonical row identities, disjoint storage and complete plane bundles.
use super::{ExpertId, Location, Region, Result, invalid};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct Plane {
    pub host_address: u64,
    pub hbm_address: u64,
    /// Physical span of one row, including its prepared internal layout.
    pub row_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub layer: usize,
    pub experts: usize,
    pub host_experts: Vec<usize>,
    pub hbm_experts: Vec<usize>,
    pub planes: Vec<Plane>,
    /// Device int32[experts]: host rows first, followed by HBM rows.
    pub mapping_address: u64,
}

pub(super) struct Slot {
    pub(super) layer: usize,
    pub(super) part: usize,
    pub(super) row: usize,
    pub(super) address: u64,
    pub(super) bytes: usize,
}
pub(super) struct Layout {
    pub(super) layers: Vec<Layer>,
    pub(super) regions: Vec<Region>,
    pub(super) backing: Vec<Option<u64>>,
    pub(super) initial: Vec<Location>,
    pub(super) slots: Vec<Slot>,
    pub(super) starts: Vec<usize>,
    pub(super) host_rows: Vec<BTreeMap<usize, usize>>,
    pub(super) map_offsets: Vec<usize>,
    pub(super) map_bytes: usize,
    pub(super) backing_bytes: usize,
}

impl Layout {
    pub(super) fn new(layers: Vec<Layer>) -> Result<Self> {
        if layers.is_empty() {
            return Err(invalid("expert row storage requires layers"));
        }
        let mut result = Self {
            layers,
            regions: vec![],
            backing: vec![],
            initial: vec![],
            slots: vec![],
            starts: vec![],
            host_rows: vec![],
            map_offsets: vec![],
            map_bytes: 0,
            backing_bytes: 0,
        };
        let mut ranges = vec![];
        let mut identities = BTreeSet::new();
        for (index, layer) in result.layers.iter().enumerate() {
            let rows = layer
                .host_experts
                .len()
                .checked_add(layer.hbm_experts.len())
                .ok_or_else(|| invalid("expert row count overflows"))?;
            if !identities.insert(layer.layer)
                || layer.experts == 0
                || layer.planes.is_empty()
                || layer.hbm_experts.is_empty()
                || rows > i32::MAX as usize
            {
                return Err(invalid("invalid expert row geometry"));
            }
            let host: BTreeMap<_, _> = layer
                .host_experts
                .iter()
                .copied()
                .enumerate()
                .map(|(r, e)| (e, r))
                .collect();
            let hbm: BTreeMap<_, _> = layer
                .hbm_experts
                .iter()
                .copied()
                .enumerate()
                .map(|(r, e)| (e, r))
                .collect();
            if host.len() != layer.host_experts.len()
                || hbm.len() != layer.hbm_experts.len()
                || host.keys().chain(hbm.keys()).any(|e| *e >= layer.experts)
                || (0..layer.experts).any(|e| !host.contains_key(&e) && !hbm.contains_key(&e))
            {
                return Err(invalid(
                    "physical rows must cover canonical experts once per tier",
                ));
            }
            result.starts.push(result.regions.len());
            result.map_offsets.push(result.map_bytes);
            let map_bytes = layer
                .experts
                .checked_mul(4)
                .ok_or_else(|| invalid("expert map size overflows"))?;
            result.map_bytes = result
                .map_bytes
                .checked_add(map_bytes)
                .ok_or_else(|| invalid("expert map size overflows"))?;
            if layer.mapping_address % 4 != 0 {
                return Err(invalid("expert map is not int32 aligned"));
            }
            add_range(&mut ranges, layer.mapping_address, map_bytes)?;
            let slot_start = result.slots.len();
            for plane in &layer.planes {
                if plane.row_bytes == 0 {
                    return Err(invalid("expert plane has empty rows"));
                }
                let host_bytes = plane
                    .row_bytes
                    .checked_mul(host.len())
                    .ok_or_else(|| invalid("host plane size overflows"))?;
                let hbm_bytes = plane
                    .row_bytes
                    .checked_mul(hbm.len())
                    .ok_or_else(|| invalid("HBM plane size overflows"))?;
                if host_bytes != 0 {
                    add_range(&mut ranges, plane.host_address, host_bytes)?;
                }
                add_range(&mut ranges, plane.hbm_address, hbm_bytes)?;
                result.backing_bytes = result
                    .backing_bytes
                    .checked_add(host_bytes)
                    .ok_or_else(|| invalid("backing size overflows"))?;
            }
            for row in 0..hbm.len() {
                for (part, plane) in layer.planes.iter().enumerate() {
                    result.slots.push(Slot {
                        layer: index,
                        part,
                        row,
                        address: plane.hbm_address + (row * plane.row_bytes) as u64,
                        bytes: plane.row_bytes,
                    });
                }
            }
            for expert in 0..layer.experts {
                for (part, plane) in layer.planes.iter().enumerate() {
                    let initial = hbm
                        .get(&expert)
                        .map(|row| Location::Hbm(slot_start + row * layer.planes.len() + part))
                        .unwrap_or(Location::Backing);
                    let backing = host
                        .get(&expert)
                        .map(|row| plane.host_address + (row * plane.row_bytes) as u64);
                    let row = hbm.get(&expert).or_else(|| host.get(&expert)).unwrap();
                    result.regions.push(Region {
                        expert: ExpertId {
                            layer: layer.layer,
                            expert,
                        },
                        part,
                        offset: row * plane.row_bytes,
                        bytes: plane.row_bytes,
                    });
                    result.backing.push(backing);
                    result.initial.push(initial);
                }
            }
            result.host_rows.push(host);
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(invalid(
                "expert planes and mapping buffers must not overlap",
            ));
        }
        result.maps(&result.initial)?;
        Ok(result)
    }

    pub(super) fn accepts(&self, region: usize, slot: usize) -> bool {
        let (Some(r), Some(s)) = (self.regions.get(region), self.slots.get(slot)) else {
            return false;
        };
        self.layers[s.layer].layer == r.expert.layer && s.part == r.part && s.bytes == r.bytes
    }

    pub(super) fn maps(&self, locations: &[Location]) -> Result<Vec<Vec<i32>>> {
        if locations.len() != self.regions.len() {
            return Err(invalid("expert location count differs from layout"));
        }
        // Slots are dense layout indices. Validate their geometry before indexing.
        let mut occupied = vec![false; self.slots.len()];
        let mut maps = vec![];
        for (index, layer) in self.layers.iter().enumerate() {
            let mut map = vec![];
            for expert in 0..layer.experts {
                let start = self.starts[index] + expert * layer.planes.len();
                let value = match locations[start] {
                    Location::Backing => *self.host_rows[index]
                        .get(&expert)
                        .ok_or_else(|| invalid("expert has no immutable backing row"))?,
                    Location::Hbm(slot) => {
                        self.slots
                            .get(slot)
                            .ok_or_else(|| invalid("expert HBM slot is absent"))?
                            .row
                            + layer.host_experts.len()
                    }
                };
                for region in start..start + layer.planes.len() {
                    match locations[region] {
                        Location::Backing if locations[start] == Location::Backing => {}
                        Location::Hbm(slot)
                            if self.accepts(region, slot)
                                && !std::mem::replace(&mut occupied[slot], true)
                                && self.slots[slot].row + layer.host_experts.len() == value
                                && matches!(locations[start], Location::Hbm(_)) => {}
                        _ => {
                            return Err(invalid(
                                "all expert planes must publish the same physical row",
                            ));
                        }
                    }
                }
                map.push(value as i32);
            }
            maps.push(map);
        }
        Ok(maps)
    }
}

fn add_range(ranges: &mut Vec<(u64, u64)>, address: u64, bytes: usize) -> Result<()> {
    let end = address
        .checked_add(bytes as u64)
        .filter(|_| address != 0 && bytes != 0)
        .ok_or_else(|| invalid("invalid expert storage address range"))?;
    ranges.push((address, end));
    Ok(())
}
