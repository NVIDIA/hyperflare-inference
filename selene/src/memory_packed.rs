// SPDX-License-Identifier: Apache-2.0

use crate::{
    Result,
    codec_copy::Packet,
    geometry::invalid,
    shared::wire::{PackedOffer, PackedUnit},
    slab::{SlabPool, Slot},
};

/// Page packets and row packets occupy separate runs so dropping the row frees its slots.
#[derive(Clone, Debug, PartialEq)]
pub struct Retained {
    pub offer: PackedOffer,
    pub pages: usize,
    offsets: Vec<usize>,
}
impl Eq for Retained {}

impl Retained {
    pub(crate) fn install(
        layout: &crate::Layout,
        pool: &SlabPool,
        page_slots: &mut Vec<Slot>,
        row_slots: Option<&mut Vec<Slot>>,
        first: u64,
        pages: usize,
        mut packets: Vec<Packet>,
        old: Option<&Self>,
    ) -> Result<Option<Self>> {
        if let Some(prior) = old
            && prior
                .offer
                .units
                .iter()
                .filter(|u| u.kind == 0)
                .all(|u| u.representation == 0)
            && packets.iter().all(|p| p.unit.representation == 0)
        {
            // A literal refill can turn the entire record raw. Materialize shared pages
            // before changing their slots, so a failed allocation leaves the old record readable.
            for p in prior.read(
                &pool.addresses(page_slots),
                &[],
                pool.page_bytes(),
                0,
                pages,
                false,
            )? {
                for rank in
                    std::iter::once(p.unit.stored_rank).chain(p.unit.replicas.iter().copied())
                {
                    let mut unit = p.unit.clone();
                    unit.stored_rank = rank;
                    unit.replicas.clear();
                    packets.push(Packet {
                        unit,
                        data: p.data.clone(),
                    });
                }
            }
            let needed = pages.saturating_sub(page_slots.len());
            if needed > 0 {
                page_slots.extend(
                    pool.allocate(needed)
                        .ok_or_else(|| invalid("no slots for a literal refill"))?,
                );
            }
            return Self::install(
                layout, pool, page_slots, row_slots, first, pages, packets, None,
            );
        }
        // A capture of literal shares uses the ordinary raw slots and envelope.
        if old.is_none() && packets.iter().all(|p| p.unit.representation == 0) {
            let addresses = row_slots
                .as_deref()
                .map(|s| pool.addresses(s))
                .unwrap_or_default();
            let ranks = layout.ranks.len();
            if packets.len() != (pages + usize::from(!addresses.is_empty())) * ranks {
                return Err(invalid("incomplete literal mirror"));
            }
            packets.sort_by_key(|p| (p.unit.kind, p.unit.logical_index, p.unit.stored_rank));
            for (i, p) in packets.iter().enumerate() {
                let rank = i % ranks;
                let page = i / ranks;
                let row = page == pages;
                let want = if row {
                    layout.row_share(rank)
                } else {
                    layout.page_share(rank)
                };
                if p.unit.kind != i32::from(row)
                    || p.unit.stored_rank != rank as u32
                    || p.data.len() as u64 != want
                    || p.unit.logical_index != if row { 0 } else { first + page as u64 }
                {
                    return Err(invalid("literal mirror has the wrong rank or length"));
                }
                let stride = layout.page_share(rank) as usize;
                let page_address;
                let destinations = if row {
                    addresses.as_slice()
                } else {
                    page_address = [pool.slot_address(page_slots[page])];
                    &page_address
                };
                for (i, &base) in destinations
                    .iter()
                    .take(want.div_ceil(stride as u64) as usize)
                    .enumerate()
                {
                    let offset = i * stride;
                    let count = stride.min(p.data.len() - offset);
                    // The provisional record owns these destinations through installation.
                    let out = unsafe {
                        std::slice::from_raw_parts_mut(
                            (base + layout.ranks[rank].page_offset as usize) as *mut u8,
                            count,
                        )
                    };
                    p.data.copy_to(offset, out);
                }
            }
            return Ok(None);
        }
        packets.sort_by_key(|p| (p.unit.kind, p.unit.logical_index, p.unit.stored_rank));
        // Share identical rank payloads only after a byte comparison.
        let mut unique: Vec<Packet> = Vec::new();
        for p in packets {
            let u = &p.unit;
            if let Some(prior) = unique
                .iter_mut()
                .rev()
                .take_while(|x| x.unit.kind == u.kind && x.unit.logical_index == u.logical_index)
                .find(|x| {
                    x.unit.original_length == u.original_length
                        && x.unit.representation == u.representation
                        && x.unit.profile == u.profile
                        && x.unit.original_crc == u.original_crc
                        && x.data == p.data
                })
            {
                prior.unit.replicas.push(u.stored_rank);
            } else {
                unique.push(p);
            }
        }
        let mut held = old.cloned().unwrap_or_else(|| Self {
            offer: PackedOffer {
                first_page: first,
                units: Vec::new(),
            },
            pages,
            offsets: Vec::new(),
        });
        if old.is_some() {
            held.drop_row();
        }
        let (page_packets, row_packets) =
            unique.split_at(unique.partition_point(|p| p.unit.kind == 0));
        let mut write = |packets: &[Packet], slots: &mut Vec<Slot>| -> Result<()> {
            let length = packets
                .iter()
                .try_fold(0usize, |n, p| n.checked_add(p.data.len()))
                .ok_or_else(|| invalid("mirror length overflow"))?;
            let used = length.div_ceil(pool.page_bytes());
            if used > slots.len() {
                return Err(invalid("encoded mirror exceeds reserved slots"));
            }
            let addresses = pool.addresses(slots);
            let mut offset = 0;
            for p in packets {
                held.offer.units.push(p.unit.clone());
                held.offsets.push(offset);
                access(
                    &addresses,
                    pool.page_bytes(),
                    offset,
                    p.data.len(),
                    |base, pos, n| {
                        // The store owns these slots exclusively until publication.
                        let out = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, n) };
                        p.data.copy_to(pos, out);
                    },
                )?;
                offset += p.data.len();
            }
            pool.release(slots.drain(used..));
            Ok(())
        };
        if old.is_none() {
            write(page_packets, page_slots)?;
        }
        if let Some(slots) = row_slots {
            write(row_packets, slots)?;
        }
        Ok(Some(held))
    }

    pub(crate) fn scattered(layout: &crate::shared::packed::Layout, width: usize) -> Self {
        let offer = layout.offer().clone();
        let mut offsets = Vec::with_capacity(offer.units.len());
        let mut pos = 0;
        let mut last = None;
        for u in &offer.units {
            let coord = (u.kind, u.logical_index);
            if last != Some(coord) {
                pos = if u.kind == 0 {
                    (u.logical_index - offer.first_page) as usize * width
                } else {
                    0
                };
            }
            last = Some(coord);
            offsets.push(pos);
            pos += u.stored_length as usize;
        }
        Self {
            offer,
            pages: layout.page_count(),
            offsets,
        }
    }

    pub(crate) fn drop_row(&mut self) {
        let n = self.offer.units.partition_point(|u| u.kind == 0);
        self.offsets.truncate(n);
        self.offer.units.truncate(n);
    }
    pub(crate) fn units(
        &self,
        first: usize,
        pages: usize,
        row: bool,
    ) -> impl Iterator<Item = (usize, &PackedUnit)> {
        let start = self.offer.first_page + first as u64;
        let end = start + pages as u64;
        self.offer.units.iter().enumerate().filter(move |(_, u)| {
            if u.kind == 1 {
                row
            } else {
                (start..end).contains(&u.logical_index)
            }
        })
    }
    pub(crate) fn pinned(
        &self,
        page_slots: &[usize],
        row_slots: &[usize],
        width: usize,
        first: usize,
        pages: usize,
        row: bool,
    ) -> Result<Vec<crate::codec_copy::PinnedPacket>> {
        self.units(first, pages, row)
            .map(|(i, unit)| {
                pinned_unit(
                    unit,
                    if unit.kind == 0 {
                        page_slots
                    } else {
                        row_slots
                    },
                    width,
                    self.offsets[i],
                )
            })
            .collect()
    }

    pub(crate) fn read(
        &self,
        page_slots: &[usize],
        row_slots: &[usize],
        width: usize,
        first: usize,
        pages: usize,
        row: bool,
    ) -> Result<Vec<Packet>> {
        self.units(first, pages, row)
            .map(|(i, u)| {
                let mut data = vec![0; u.stored_length as usize];
                access(
                    if u.kind == 0 { page_slots } else { row_slots },
                    width,
                    self.offsets[i],
                    data.len(),
                    |base, pos, n| {
                        // The restore/export holds a pin on this immutable record.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                base as *const u8,
                                data[pos..].as_mut_ptr(),
                                n,
                            );
                        }
                    },
                )?;
                Ok(Packet {
                    unit: u.clone(),
                    data: data.into(),
                })
            })
            .collect()
    }
}

pub(crate) fn pinned_unit(
    unit: &PackedUnit,
    slots: &[usize],
    width: usize,
    offset: usize,
) -> Result<crate::codec_copy::PinnedPacket> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    access(
        slots,
        width,
        offset,
        unit.stored_length as usize,
        |address, _, len| {
            if let Some((base, n)) = spans.last_mut()
                && *base + *n == address
            {
                *n += len;
            } else {
                spans.push((address, len));
            }
        },
    )?;
    Ok(crate::codec_copy::PinnedPacket {
        unit: unit.clone(),
        spans,
    })
}

fn access(
    slots: &[usize],
    width: usize,
    offset: usize,
    length: usize,
    mut visit: impl FnMut(usize, usize, usize),
) -> Result<()> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| invalid("packed slot overflow"))?;
    if end > slots.len().saturating_mul(width) {
        return Err(invalid("packed bytes outside retained slots"));
    }
    let mut pos = offset;
    while pos < end {
        let within = pos % width;
        let n = (end - pos).min(width - within);
        visit(slots[pos / width] + within, pos - offset, n);
        pos += n;
    }
    Ok(())
}
