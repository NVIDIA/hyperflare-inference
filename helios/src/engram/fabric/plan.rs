/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! RDMA READ planning for one row-cache miss gather.
//!
//! The row cache passes its misses as sorted, unique row IDs and one output buffer laid out as
//! `[count × weight rows][count × scale rows]`. Consecutive IDs are contiguous in both the
//! store's plane and the output, so each run of consecutive IDs is one READ per plane, split
//! only where a run exceeds the store's per-request limit.

/// One plane to read: rows of `row_bytes` at `remote` under `rkey`, written to the output
/// starting at byte offset `local`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plane {
    pub row_bytes: usize,
    pub rkey: u32,
    pub remote: u64,
    pub local: usize,
}

/// One RDMA READ of `nbytes` from `remote` into the output at byte offset `local`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Read {
    pub rkey: u32,
    pub remote: u64,
    pub local: usize,
    pub nbytes: u32,
}

/// Append the READs for `ids` to `reads`. `ids` must be strictly increasing and nonnegative,
/// as the row cache produces them; `max_read` caps every READ and must be nonzero.
pub(crate) fn plan(
    ids: &[i64],
    planes: &[Plane],
    max_read: usize,
    reads: &mut Vec<Read>,
) -> Result<(), &'static str> {
    if max_read == 0 || max_read > u32::MAX as usize {
        return Err("Engram store READ limit is out of range");
    }
    if ids.first().is_some_and(|id| *id < 0) || ids.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err("Engram store gathers require sorted, unique, nonnegative row IDs");
    }
    for plane in planes {
        if plane.row_bytes == 0 {
            return Err("Engram store plane has empty rows");
        }
        let mut begin = 0;
        while begin < ids.len() {
            let first = ids[begin];
            let mut end = begin + 1;
            while end < ids.len() && ids[end] == first + (end - begin) as i64 {
                end += 1;
            }
            let bytes = (end - begin)
                .checked_mul(plane.row_bytes)
                .ok_or("Engram store READ exceeds the host range")?;
            let mut local = begin
                .checked_mul(plane.row_bytes)
                .and_then(|offset| plane.local.checked_add(offset))
                .ok_or("Engram store READ exceeds the host range")?;
            let mut remote = (first as u64)
                .checked_mul(plane.row_bytes as u64)
                .and_then(|offset| plane.remote.checked_add(offset))
                .ok_or("Engram store READ exceeds the remote range")?;
            let mut remaining = bytes;
            while remaining > 0 {
                let nbytes = remaining.min(max_read);
                reads.push(Read { rkey: plane.rkey, remote, local, nbytes: nbytes as u32 });
                local += nbytes;
                remote += nbytes as u64;
                remaining -= nbytes;
            }
            begin = end;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planes(count: usize) -> [Plane; 2] {
        [
            Plane { row_bytes: 256, rkey: 1, remote: 0x1000, local: 0 },
            Plane { row_bytes: 8, rkey: 2, remote: 0x9000_0000, local: count * 256 },
        ]
    }

    #[test]
    fn adjacent_ids_merge_and_separated_ids_do_not() {
        let ids = [3, 4, 5, 9, 11, 12];
        let mut reads = Vec::new();
        plan(&ids, &planes(ids.len()), 1 << 20, &mut reads).unwrap();
        assert_eq!(
            reads,
            [
                Read { rkey: 1, remote: 0x1000 + 3 * 256, local: 0, nbytes: 3 * 256 },
                Read { rkey: 1, remote: 0x1000 + 9 * 256, local: 3 * 256, nbytes: 256 },
                Read { rkey: 1, remote: 0x1000 + 11 * 256, local: 4 * 256, nbytes: 2 * 256 },
                Read { rkey: 2, remote: 0x9000_0000 + 3 * 8, local: 6 * 256, nbytes: 3 * 8 },
                Read { rkey: 2, remote: 0x9000_0000 + 9 * 8, local: 6 * 256 + 3 * 8, nbytes: 8 },
                Read { rkey: 2, remote: 0x9000_0000 + 11 * 8, local: 6 * 256 + 4 * 8, nbytes: 2 * 8 },
            ]
        );
    }

    #[test]
    fn runs_longer_than_the_read_limit_split_without_gaps() {
        let ids: Vec<i64> = (100..110).collect();
        let mut reads = Vec::new();
        plan(&ids, &planes(ids.len())[..1], 1000, &mut reads).unwrap();
        assert_eq!(reads.iter().map(|read| read.nbytes).collect::<Vec<_>>(), [1000, 1000, 560]);
        for pair in reads.windows(2) {
            assert_eq!(pair[0].local + pair[0].nbytes as usize, pair[1].local);
            assert_eq!(pair[0].remote + u64::from(pair[0].nbytes), pair[1].remote);
        }
        assert_eq!(reads[0].remote, 0x1000 + 100 * 256);
    }

    #[test]
    fn invalid_id_order_and_limits_are_rejected() {
        let mut reads = Vec::new();
        for ids in [&[2i64, 2][..], &[3, 1], &[-1, 4]] {
            assert!(plan(ids, &planes(2), 4096, &mut reads).is_err(), "{ids:?}");
        }
        assert!(plan(&[1], &planes(1), 0, &mut reads).is_err());
        let empty = Plane { row_bytes: 0, rkey: 0, remote: 0, local: 0 };
        assert!(plan(&[1], &[empty], 4096, &mut reads).is_err());
        assert!(reads.is_empty());
        plan(&[], &planes(0), 4096, &mut reads).unwrap();
        assert!(reads.is_empty());
    }

    /// Executing the plan as byte copies from the planes reproduces the miss layout the row
    /// cache expects, including unaligned row sizes and runs split mid-row.
    #[test]
    fn executed_reads_reproduce_the_row_cache_miss_layout() {
        let (weights, scales, rows) = (80usize, 10usize, 64usize);
        let weight_plane: Vec<u8> = (0..rows * weights).map(|i| (i * 7 % 251) as u8).collect();
        let scale_plane: Vec<u8> = (0..rows * scales).map(|i| (i * 13 % 241) as u8).collect();
        let ids = [0i64, 1, 2, 7, 8, 30, 63];
        let count = ids.len();
        let mut expected = vec![0u8; count * (weights + scales)];
        for (row, id) in ids.iter().map(|id| *id as usize).enumerate() {
            expected[row * weights..(row + 1) * weights]
                .copy_from_slice(&weight_plane[id * weights..(id + 1) * weights]);
            expected[count * weights + row * scales..count * weights + (row + 1) * scales]
                .copy_from_slice(&scale_plane[id * scales..(id + 1) * scales]);
        }
        let planes = [
            Plane { row_bytes: weights, rkey: 0, remote: 0, local: 0 },
            Plane { row_bytes: scales, rkey: 1, remote: 0, local: count * weights },
        ];
        for max_read in [7, 64, 4096] {
            let mut reads = Vec::new();
            plan(&ids, &planes, max_read, &mut reads).unwrap();
            assert!(reads.iter().all(|read| read.nbytes as usize <= max_read));
            let mut output = vec![0xaau8; expected.len()];
            for read in &reads {
                let source = if read.rkey == 0 { &weight_plane } else { &scale_plane };
                let remote = read.remote as usize;
                output[read.local..read.local + read.nbytes as usize]
                    .copy_from_slice(&source[remote..remote + read.nbytes as usize]);
            }
            assert_eq!(output, expected, "max_read {max_read}");
        }
    }
}
