// SPDX-License-Identifier: Apache-2.0

//! Stored rank shares passed between a host store and its device copier.
use crate::shared::{packed, wire::PackedUnit};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct PinnedPacket {
    pub unit: PackedUnit,
    pub(crate) spans: Vec<(usize, usize)>,
}

impl PinnedPacket {
    /// # Safety
    /// The ranges must stay registered and immutable through the last copier fence.
    pub unsafe fn from_registered(unit: PackedUnit, spans: Vec<(usize, usize)>) -> Self {
        Self { unit, spans }
    }
    /// # Safety
    /// The copier must still own the immutable slab pin or transport window lease.
    pub unsafe fn chunks(&self) -> Vec<&[u8]> {
        self.spans
            .iter()
            .map(|&(address, length)| unsafe {
                std::slice::from_raw_parts(address as *const u8, length)
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct Restore {
    pub first: u64,
    pub packets: Vec<PinnedPacket>,
}

#[derive(Clone, Debug)]
pub struct Window(pub Arc<Restore>);
impl PartialEq for Window {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Window {}
impl Window {
    pub fn restore_pinned(first: u64, packets: Vec<PinnedPacket>) -> Self {
        Self(Arc::new(Restore { first, packets }))
    }
}

pub fn profile(geometry: &[crate::CacheGeometry]) -> Option<[u8; 32]> {
    if geometry
        .iter()
        .any(|rank| rank.placement == crate::offload::KvCompression::Off)
    {
        return None;
    }
    checked_profile(geometry).ok()
}

pub fn checked_profile(geometry: &[crate::CacheGeometry]) -> std::result::Result<[u8; 32], String> {
    let mut expected = None;
    for (rank, g) in geometry.iter().enumerate() {
        let actual = g
            .checked_codec_profile()
            .map_err(|e| format!("rank {rank}: {e}"))?;
        if expected.is_some_and(|p| p != actual) {
            return Err(format!(
                "codec field layout on rank {rank} differs from rank 0"
            ));
        }
        expected = Some(actual);
    }
    expected.ok_or_else(|| "no cache ranks".to_owned())
}

pub fn reader_caps(profile: Option<[u8; 32]>, rdma: bool) -> crate::shared::wire::CodecCaps {
    let mut caps = packed::raw_reader();
    if let Some(profile) = profile.filter(|_| !rdma) {
        caps.receive_profiles.push(profile.to_vec());
    }
    caps
}
