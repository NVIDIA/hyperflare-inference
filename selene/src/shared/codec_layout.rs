// SPDX-License-Identifier: Apache-2.0

use super::{
    codec::{Encoding, Field, Plan},
    wire::{CodecField, CodecLayout, CodecPlane},
};
use crate::geometry::{CacheGeometry, Plane};
use blake2::{Blake2b, Digest};

const REVISION: &str = "ZKV1/v1/64-pivots";

impl From<&CacheGeometry> for CodecLayout {
    fn from(g: &CacheGeometry) -> Self {
        let plane = |p: &Plane| CodecPlane {
            name: p.name.clone(),
            unit_bytes: p.unit_bytes,
            fields: g
                .codec_fields
                .iter()
                .filter(|f| f.buffer == p.name)
                .map(|f| CodecField {
                    offset: f.offset,
                    length: f.length,
                    encoding: f.encoding,
                })
                .collect(),
        };
        Self {
            revision: REVISION.into(),
            page_tokens: g.page_tokens,
            planes: g.planes.iter().map(plane).collect(),
            row: Some(plane(&g.row)),
        }
    }
}

fn encoding(id: u32) -> Option<Encoding> {
    Some(match id {
        0 => Encoding::Bytes,
        1 => Encoding::Float32Matrix128,
        2 => Encoding::Bfloat16,
        20 => Encoding::Convolution20,
        32 => Encoding::Convolution32,
        40 => Encoding::Latent528,
        41 => Encoding::Index132,
        42 => Encoding::Fp8Page512,
        45 => Encoding::WindowFp4,
        47 => Encoding::LatentFp4,
        _ => return None,
    })
}

impl CodecLayout {
    pub fn profile(&self) -> Result<[u8; 32], String> {
        self.plans_and_profile().map(|(_, digest)| digest).map_err(|error| error.to_string())
    }

    pub fn plans(&self) -> std::io::Result<[Plan; 2]> {
        self.plans_and_profile().map(|(plans, _)| plans)
    }

    pub fn plans_and_profile(&self) -> std::io::Result<([Plan; 2], [u8; 32])> {
        use std::io::Error;
        if self.revision != REVISION {
            return Err(Error::other("unsupported codec layout revision"));
        }
        let row = self.row.as_ref().ok_or_else(|| Error::other("missing codec row"))?;
        let mut hash = Blake2b::<blake2::digest::consts::U32>::new();
        hash.update(self.revision.as_bytes());
        hash.update(self.page_tokens.to_le_bytes());
        let mut plan = |planes: &[CodecPlane]| {
            let mut fields = Vec::new();
            let mut base = 0u64;
            for plane in planes {
                if plane.fields.is_empty() {
                    return Err(Error::other(format!("codec layout has no field for plane {}", plane.name)));
                }
                let padded = crate::geometry::padded_bytes(plane.unit_bytes)
                    .and_then(|size| base.checked_add(size))
                    .ok_or_else(|| Error::other("codec unit size overflow"))?;
                for field in &plane.fields {
                    if field.offset.checked_add(field.length).is_none_or(|end| end > plane.unit_bytes) {
                        return Err(Error::other(format!("codec field exceeds plane {}", plane.name)));
                    }
                    fields.push(Field {
                        offset: usize::try_from(base + field.offset).map_err(Error::other)?,
                        length: usize::try_from(field.length).map_err(Error::other)?,
                        encoding: encoding(field.encoding).ok_or_else(|| Error::other("unsupported codec field encoding"))?,
                    });
                    hash.update((plane.name.len() as u64).to_le_bytes());
                    hash.update(plane.name.as_bytes());
                    hash.update(field.offset.to_le_bytes());
                    hash.update(field.length.to_le_bytes());
                    hash.update(field.encoding.to_le_bytes());
                }
                hash.update(plane.unit_bytes.to_le_bytes());
                base = padded;
            }
            Plan::new(usize::try_from(base).map_err(Error::other)?, &fields)
        };
        let plans = [plan(&self.planes)?, plan(std::slice::from_ref(row))?];
        Ok((plans, hash.finalize().into()))
    }
}
