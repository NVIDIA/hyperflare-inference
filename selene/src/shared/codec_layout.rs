// SPDX-License-Identifier: Apache-2.0

use super::{
    codec::{Encoding, Field, Plan},
    wire::{CodecField, CodecLayout, CodecPlane},
};
use crate::geometry::{BLOCK_BYTES, CacheGeometry, Plane};
use blake2::{Blake2b, Digest};

const REVISION: &str = "SGC1/v1/64-pivots";

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

fn encoding(id: u32) -> Option<(Encoding, u64)> {
    Some(match id {
        0 => (Encoding::Bytes, 1),
        1 => (Encoding::Float32Matrix128, 65536),
        2 => (Encoding::Bfloat16, 2),
        20 => (Encoding::Convolution20, 20),
        32 => (Encoding::Convolution32, 32),
        40 => (Encoding::Latent528, 135168),
        41 => (Encoding::Index132, 8448),
        42 => (Encoding::Fp8Page512, 32768),
        45 => (Encoding::WindowFp4, 33792),
        47 => (Encoding::LatentFp4, 18432),
        _ => return None,
    })
}

impl CodecLayout {
    pub fn profile(&self) -> Result<[u8; 32], String> {
        if self
            .planes
            .iter()
            .chain(self.row.iter())
            .all(|p| p.fields.is_empty())
        {
            return Err("codec layout has no fields".into());
        }
        if self.revision != REVISION {
            return Err("unsupported codec layout revision".into());
        }
        let row = self.row.as_ref().ok_or("missing codec row")?;
        let padded = |n: u64| {
            n.checked_add(BLOCK_BYTES - 1)
                .map(|v| v & !(BLOCK_BYTES - 1))
        };
        let pages = self
            .planes
            .iter()
            .try_fold(0u64, |n, p| n.checked_add(padded(p.unit_bytes)?));
        if !pages.is_some_and(|n| n <= 256 << 20)
            || !padded(row.unit_bytes).is_some_and(|n| n <= 256 << 20)
        {
            return Err("page or row share exceeds the 256 MiB codec limit".into());
        }
        let mut hash = Blake2b::<blake2::digest::consts::U32>::new();
        hash.update(self.revision.as_bytes());
        hash.update(self.page_tokens.to_le_bytes());
        for plane in self.planes.iter().chain([row]) {
            let mut end = 0;
            let mut count = 0;
            for f in &plane.fields {
                let next = f.offset.checked_add(f.length).ok_or_else(|| {
                    format!("codec field {} at {} overflows", plane.name, f.offset)
                })?;
                let (_, block) = encoding(f.encoding).ok_or_else(|| {
                    format!(
                        "codec field {} at {} has unsupported encoding {}",
                        plane.name, f.offset, f.encoding
                    )
                })?;
                if f.length == 0
                    || f.offset < end
                    || next > plane.unit_bytes
                    || f.length % block != 0
                    || (f.encoding == 1 && f.offset % 4 != 0)
                {
                    return Err(format!(
                        "codec field {} at {} with length {} is invalid for encoding {} and a {}-byte plane",
                        plane.name, f.offset, f.length, f.encoding, plane.unit_bytes
                    ));
                }
                hash.update((plane.name.len() as u64).to_le_bytes());
                hash.update(plane.name.as_bytes());
                hash.update(f.offset.to_le_bytes());
                hash.update(f.length.to_le_bytes());
                hash.update(f.encoding.to_le_bytes());
                end = next;
                count += 1;
            }
            if count == 0 {
                return Err(format!(
                    "codec layout has no field for plane {}",
                    plane.name
                ));
            }
            hash.update(plane.unit_bytes.to_le_bytes());
        }
        Ok(hash.finalize().into())
    }

    pub fn plans(&self) -> std::io::Result<[Plan; 2]> {
        fn plan(planes: &[CodecPlane]) -> std::io::Result<Plan> {
            let mut fields = Vec::new();
            let mut base = 0u64;
            for plane in planes {
                for f in &plane.fields {
                    let (encoding, _) = encoding(f.encoding)
                        .ok_or_else(|| std::io::Error::other("unqualified KV field"))?;
                    fields.push(Field {
                        offset: (base + f.offset) as usize,
                        length: f.length as usize,
                        encoding,
                    });
                }
                base += plane.unit_bytes.next_multiple_of(BLOCK_BYTES);
            }
            Plan::new(base as usize, &fields)
        }
        self.profile().map_err(std::io::Error::other)?;
        Ok([plan(&self.planes)?, plan(self.row.as_slice())?])
    }
}
