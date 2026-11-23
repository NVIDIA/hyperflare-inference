// SPDX-License-Identifier: Apache-2.0

//! Stored rank shares passed between a host store and its device copier.
use crate::shared::packed;

pub fn reader_caps(profile: Option<[u8; 32]>, rdma: bool) -> crate::shared::wire::CodecCaps {
    let mut caps = packed::raw_reader();
    if let Some(profile) = profile.filter(|_| !rdma) {
        caps.receive_profiles.push(profile.to_vec());
    }
    caps
}
