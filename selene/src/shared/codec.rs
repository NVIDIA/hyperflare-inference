// SPDX-License-Identifier: Apache-2.0

//! Lossless packets for one rank's share of a cache page or continuation row.

use super::packed::{crc32, crc32_chunks};
use rayon::prelude::*;
use std::io::{Error, ErrorKind, Result};

pub static MIRROR_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static ORIGINAL_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

const MAGIC: u32 = u32::from_le_bytes(*b"SGC1");
const HEADER: usize = 16;
const ENTRY: usize = 40;
const MAX_UNIT: usize = 256 << 20;
const CHUNK: usize = 65536;

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

/// Each field describes a contiguous interval of the padded rank share.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub offset: usize,
    pub length: usize,
    pub encoding: Encoding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Bytes,
    Bfloat16,
    Float32Matrix128,
    Convolution20,
    Convolution32,
    Latent528,
    Index132,
    Fp8Page512,
    WindowFp4,
    LatentFp4,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Frame {
    base: i32,
    full: i32,
    sub: i32,
    n: i32,
    kind: i32,
    rank: i32,
    layer: i32,
    page: i32,
    work: i32,
    arena: i32,
    reserved: i32,
}

impl Frame {
    fn shape(&self) -> [i32; 8] {
        [
            self.base, self.full, self.sub, self.n, self.kind, self.rank, self.layer, self.page,
        ]
    }

    fn write_entry(&self, entry: &mut [u8], pos: usize, len: usize) {
        let words = self.shape().map(|v| v as u32);
        let words = words.into_iter().chain([pos as u32, len as u32]);
        for (dst, word) in entry.chunks_exact_mut(4).zip(words) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
    }

    fn raw_bytes(&self) -> usize {
        self.n as usize * if self.kind == 1 { 4 } else { 1 }
    }
}

unsafe extern "C" {
    fn kv_cpu_frame(
        frame: *const Frame,
        raw: *mut u8,
        packet: *mut u8,
        capacity: i32,
        scratch: *mut u32,
        decoding: bool,
    ) -> i32;
}

#[derive(Clone, Debug)]
pub struct Plan {
    frames: Vec<Frame>,
    original_bytes: usize,
    work: usize,
    arena: usize,
}

impl Plan {
    /// Padding and untyped intervals are included byte for byte.
    pub fn new(original_bytes: usize, fields: &[Field]) -> Result<Self> {
        if original_bytes == 0 || original_bytes > MAX_UNIT {
            return Err(invalid("codec unit size outside supported range"));
        }
        let mut plan = Self {
            frames: vec![],
            original_bytes,
            work: 0,
            arena: 0,
        };
        let mut end = 0;
        for field in fields {
            if field.offset < end
                || field.length == 0
                || field
                    .offset
                    .checked_add(field.length)
                    .is_none_or(|n| n > original_bytes)
            {
                return Err(invalid("invalid codec field extent"));
            }
            plan.bytes(end, field.offset - end)?;
            plan.field(*field)?;
            end = field.offset + field.length;
        }
        plan.bytes(end, original_bytes - end)?;
        Ok(plan)
    }

    fn frame(&mut self, base: usize, full: usize, sub: usize, n: usize, kind: i32) -> Result<()> {
        if self.frames.len() >= 65536 {
            return Err(invalid("too many codec frames"));
        }
        let frame = Frame {
            base: base as i32,
            full: full as i32,
            sub: sub as i32,
            n: n as i32,
            kind,
            rank: 0,
            layer: 0,
            page: 0,
            work: self.work as i32,
            arena: self.arena as i32,
            reserved: 0,
        };
        let aligned = frame.raw_bytes().next_multiple_of(4);
        self.work += aligned;
        self.arena += aligned + 4;
        self.frames.push(frame);
        Ok(())
    }

    fn bytes(&mut self, offset: usize, size: usize) -> Result<()> {
        for start in (0..size).step_by(CHUNK) {
            let n = (size - start).min(CHUNK);
            self.frame(offset + start, n, 0, n, 0)?;
        }
        Ok(())
    }

    fn field(&mut self, field: Field) -> Result<()> {
        use Encoding::*;
        let Field {
            offset,
            length,
            encoding,
        } = field;
        let multiple = |size| {
            if length.is_multiple_of(size) {
                Ok(())
            } else {
                Err(invalid("field length disagrees with codec geometry"))
            }
        };
        match encoding {
            Bytes => self.bytes(offset, length)?,
            Float32Matrix128 => {
                multiple(CHUNK)?;
                if offset % 4 != 0 {
                    return Err(invalid("unaligned codec matrix"));
                }
                for sub in (0..length).step_by(CHUNK) {
                    self.frame(offset, length, sub, 16384, 1)?;
                }
            }
            Bfloat16 | Convolution20 | Convolution32 => {
                let width = match encoding {
                    Bfloat16 => 2,
                    Convolution20 => 20,
                    _ => 32,
                };
                multiple(width)?;
                let rows = length / width;
                for lane in 0..width {
                    for sub in (0..rows).step_by(32768) {
                        self.frame(
                            offset,
                            length,
                            lane * rows + sub,
                            (rows - sub).min(32768),
                            width as i32,
                        )?;
                    }
                }
            }
            Latent528 | Index132 | Fp8Page512 => {
                let (stride, n, kind) = match encoding {
                    Latent528 => (135168, 131072, 40),
                    Index132 => (8448, 8192, 41),
                    _ => (32768, 32768, 42),
                };
                multiple(stride)?;
                for sub in (0..length).step_by(stride) {
                    self.frame(offset, length, sub, n, kind)?;
                    if kind != 42 {
                        let count = n / 128;
                        for lane in 0..4 {
                            self.frame(offset, length, sub + lane * count, count, kind + 3)?;
                        }
                    }
                }
            }
            WindowFp4 | LatentFp4 => {
                let (stride, n, kind) = if encoding == WindowFp4 {
                    (33792, 32768, 45)
                } else {
                    (18432, 16384, 47)
                };
                multiple(stride)?;
                for base in (0..length).step_by(stride) {
                    self.frame(offset + base, stride, 0, n, kind)?;
                    self.frame(offset + base, stride, 0, stride - n, kind + 1)?;
                }
            }
        }
        Ok(())
    }

    /// The same immutable frame descriptors feed the GPU codec.
    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }

    pub fn original_bytes(&self) -> usize {
        self.original_bytes
    }

    pub fn workspace_bytes(&self) -> usize {
        self.work
    }

    pub fn arena_bytes(&self) -> usize {
        self.arena
    }

    pub fn metadata(&self) -> usize {
        HEADER + self.frames.len() * ENTRY
    }

    pub fn max_packet(&self) -> usize {
        self.metadata() + self.arena
    }

    fn entries(&self, packet: &[u8], total: usize) -> Result<Vec<std::ops::Range<usize>>> {
        let word = |p| u32::from_le_bytes(packet[p..p + 4].try_into().unwrap());
        if packet.len() < self.metadata()
            || total < self.metadata()
            || total > self.max_packet()
            || word(0) != MAGIC
            || word(4) != 1
            || word(8) as usize != self.frames.len()
            || word(12) as usize != self.original_bytes
        {
            return Err(invalid("invalid codec packet header"));
        }
        let mut pos = self.metadata();
        let mut ranges = Vec::with_capacity(self.frames.len());
        for (frame, entry) in self
            .frames
            .iter()
            .zip(packet[HEADER..self.metadata()].chunks_exact(ENTRY))
        {
            let mut values = entry
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
            for expected in frame.shape() {
                if values.next() != Some(expected as u32) {
                    return Err(invalid("codec frame differs from prepared layout"));
                }
            }
            let start = values.next().unwrap() as usize;
            let len = values.next().unwrap() as usize;
            if start != pos || !(4..=frame.raw_bytes() + 4).contains(&len) || len > total - pos {
                return Err(invalid("invalid codec frame payload extent"));
            }
            ranges.push(pos..pos + len);
            pos += len;
        }
        if pos != total {
            return Err(invalid("trailing codec packet bytes"));
        }
        Ok(ranges)
    }

    /// Validate the frame directory and return the stored byte count.
    pub fn validate_chunks(&self, chunks: &[&[u8]]) -> Result<usize> {
        let input = Source::new(chunks)?;
        self.entries(&input.range(0..self.metadata())?, input.len)?;
        Ok(input.len)
    }

    fn encode_in(&self, source: &[u8], pool: Option<&rayon::ThreadPool>) -> Result<Vec<u8>> {
        if source.len() != self.original_bytes {
            return Err(invalid("codec input size mismatch"));
        }
        let mut out = vec![0; self.max_packet()];
        for (chunk, word) in out[..HEADER].chunks_exact_mut(4).zip([
            MAGIC,
            1,
            self.frames.len() as u32,
            source.len() as u32,
        ]) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        let mut pos = self.metadata();
        let encode = |scratch: &mut Vec<u32>, frame: &Frame, output: *mut u8| {
            let cap = frame.raw_bytes() + 4;
            scratch.resize(frame.raw_bytes().div_ceil(4), 0);
            // The caller reserves raw_bytes + 4 output bytes per frame. Encoding only reads source.
            let n = unsafe {
                kv_cpu_frame(
                    frame,
                    source.as_ptr().cast_mut(),
                    output,
                    cap as i32,
                    scratch.as_mut_ptr(),
                    false,
                )
            };
            if n < 4 || n as usize > cap {
                Err(invalid("codec encode failed"))
            } else {
                Ok(n as usize)
            }
        };
        let (directory, payload) = out.split_at_mut(self.metadata());
        let entries = directory[HEADER..].chunks_exact_mut(ENTRY);
        if let Some(pool) = pool {
            // Each arena offset reserves a separate maximum-sized output until all frames finish.
            let base = payload.as_mut_ptr() as usize;
            let sizes: Vec<usize> = pool.install(|| {
                self.frames
                    .par_iter()
                    .map_init(Vec::new, |scratch, frame| {
                        encode(scratch, frame, (base + frame.arena as usize) as *mut u8)
                    })
                    .collect::<Result<_>>()
            })?;
            for ((frame, length), entry) in self.frames.iter().zip(sizes).zip(entries) {
                let start = frame.arena as usize;
                payload.copy_within(start..start + length, pos - self.metadata());
                frame.write_entry(entry, pos, length);
                pos += length;
            }
        } else {
            let mut scratch = Vec::new();
            for (frame, entry) in self.frames.iter().zip(entries) {
                let offset = pos - self.metadata();
                let output = &mut payload[offset..offset + frame.raw_bytes() + 4];
                let length = encode(&mut scratch, frame, output.as_mut_ptr())?;
                frame.write_entry(entry, pos, length);
                pos += length;
            }
        }
        out.truncate(pos);
        Ok(out)
    }

    fn decode_chunks(
        &self,
        chunks: &[&[u8]],
        dst: &mut [u8],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<()> {
        if dst.len() != self.original_bytes {
            return Err(invalid("codec destination size mismatch"));
        }
        let input = Source::new(chunks)?;
        let ranges = self.entries(&input.range(0..self.metadata())?, input.len)?;
        let base = dst.as_mut_ptr() as usize;
        let decode = |scratch: &mut Vec<u32>, (frame, range): (&Frame, std::ops::Range<usize>)| {
            let input = input.range(range)?;
            scratch.resize(frame.raw_bytes().div_ceil(4), 0);
            // Prepared frames partition output bytes, including transposed fields and padding.
            let n = unsafe {
                kv_cpu_frame(
                    frame,
                    base as *mut u8,
                    input.as_ptr().cast_mut(),
                    input.len() as i32,
                    scratch.as_mut_ptr(),
                    true,
                )
            };
            if n < 0 {
                Err(invalid("invalid entropy data in codec frame"))
            } else {
                Ok(())
            }
        };
        match pool {
            Some(pool) => pool.install(|| {
                self.frames.par_iter().zip(ranges).try_for_each_init(Vec::new, decode)
            }),
            None => {
                let mut scratch = Vec::new();
                self.frames.iter().zip(ranges).try_for_each(|frame| decode(&mut scratch, frame))
            }
        }
    }

    /// Encode a complete rank share and compare its decoded bytes before returning the packet.
    /// Without a frame pool the calling thread processes the unit itself.
    pub fn encode_verified(
        &self,
        source: &[u8],
        verification: &mut [u8],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Vec<u8>> {
        let packet = self.encode_in(source, pool)?;
        self.decode_chunks(&[&packet], verification, pool)?;
        let exact = match pool {
            Some(pool) => pool.install(|| {
                source.par_chunks(CHUNK).zip(verification.par_chunks(CHUNK)).all(|(a, b)| a == b)
            }),
            None => source == verification,
        };
        if !exact {
            MIRROR_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(invalid("codec round-trip verification failed"));
        }
        Ok(packet)
    }

    /// Check the stored CRC before decoding and the original CRC before releasing the output.
    pub fn decode_checked(
        &self,
        chunks: &[&[u8]],
        dst: &mut [u8],
        stored_crc: u32,
        original_crc: u32,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<()> {
        if crc32_chunks(chunks) != stored_crc {
            return Err(invalid("stored codec checksum mismatch"));
        }
        self.decode_chunks(chunks, dst, pool)?;
        if crc32(dst) != original_crc {
            ORIGINAL_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(invalid("original codec checksum mismatch"));
        }
        Ok(())
    }
}

struct Source<'a> {
    chunks: &'a [&'a [u8]],
    ends: Vec<usize>,
    len: usize,
}

impl<'a> Source<'a> {
    fn new(chunks: &'a [&'a [u8]]) -> Result<Self> {
        let mut len = 0usize;
        let ends = chunks
            .iter()
            .map(|s| {
                len = len.checked_add(s.len()).ok_or_else(|| invalid("packet length overflow"))?;
                Ok(len)
            })
            .collect::<Result<_>>()?;
        Ok(Self { chunks, ends, len })
    }

    fn range(&self, range: std::ops::Range<usize>) -> Result<std::borrow::Cow<'a, [u8]>> {
        if range.end > self.len || range.is_empty() {
            return Err(invalid("short codec packet"));
        }
        let i = self.ends.partition_point(|&end| end <= range.start);
        let start = range.start - (self.ends[i] - self.chunks[i].len());
        let first = &self.chunks[i][start..];
        if first.len() >= range.len() {
            return Ok((&first[..range.len()]).into());
        }
        // Only a frame crossing a slab boundary needs local scratch.
        let mut frame = Vec::with_capacity(range.len());
        frame.extend_from_slice(first);
        for chunk in &self.chunks[i + 1..] {
            let n = chunk.len().min(range.len() - frame.len());
            frame.extend_from_slice(&chunk[..n]);
            if frame.len() == range.len() {
                break;
            }
        }
        Ok(frame.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn kv_cpu_frame_scalar(
            frame: *const Frame,
            raw: *mut u8,
            packet: *mut u8,
            capacity: i32,
            scratch: *mut u32,
            decoding: bool,
        ) -> i32;
    }

    #[test]
    fn a_fragmented_packet_checks_both_crcs_and_rejects_bad_tables_and_truncation() {
        let packet = include_bytes!("../../tests/fixtures/codec/page0.sgc");
        let plan = Plan::new(4096, &[]).unwrap();
        let mut dst = vec![0; 4096];
        let stored = 0x6c3ff1e0;
        let original = 0x476cc8d3;
        let spans = packet.chunks(19).collect::<Vec<_>>();
        plan.decode_checked(&spans, &mut dst, stored, original, None).unwrap();
        assert_eq!(crc32(&dst), original);
        assert!(plan.decode_checked(&spans, &mut dst, stored ^ 1, original, None).is_err());
        assert!(plan.decode_checked(&spans, &mut dst, stored, original ^ 1, None).is_err());
        let mut damaged = packet.to_vec();
        let entropy = plan.metadata();
        assert_eq!(damaged[entropy], 2);
        damaged[entropy + 4..entropy + 36].fill(0);
        assert!(plan.decode_checked(&[&damaged], &mut dst, crc32(&damaged), original, None).is_err());
        let mut bad_state = packet.to_vec();
        let terminal = bad_state.len() - 4;
        bad_state[terminal..].fill(0);
        assert!(plan.decode_checked(&[&bad_state], &mut dst, crc32(&bad_state), original, None).is_err());
        for cut in 0..packet.len() {
            let truncated = &packet[..cut];
            assert!(plan.decode_checked(&[truncated], &mut dst, crc32(truncated), original, None).is_err());
        }
    }

    #[test]
    fn a_lane_cannot_consume_bytes_from_its_neighbor() {
        let mut packet = include_bytes!("../../tests/fixtures/codec/page0.sgc").to_vec();
        let plan = Plan::new(4096, &[]).unwrap();
        // This fixture has a 32-byte symbol mask and two 2-byte frequencies.
        let directory = plan.metadata() + 4 + 32 + 4;
        assert_eq!(&packet[directory..directory + 4], &[36, 0, 36, 0]);
        packet[directory] -= 1;
        packet[directory + 2] += 1;
        let mut dst = vec![0; 4096];
        assert!(plan.decode_chunks(&[&packet], &mut dst, None).is_err());
        let frame = &mut packet[plan.metadata()..];
        let mut scratch = vec![0u32; dst.len() / 4];
        // Only the two lane lengths changed. The destination and scratch cover the full frame.
        let decoded = unsafe {
            kv_cpu_frame_scalar(
                &plan.frames[0],
                dst.as_mut_ptr(),
                frame.as_mut_ptr(),
                frame.len() as i32,
                scratch.as_mut_ptr(),
                true,
            )
        };
        assert!(decoded < 0);
    }

    #[test]
    fn partial_lane_groups_restore_only_the_requested_bytes() {
        for size in [1, 255, 256, 257, 1023, 1024, 1025, 4095, 4097, 16385, 65535, 65537] {
            let raw: Vec<u8> = (0..size).map(|i| ((i / 3 + i / 29) % 7) as u8).collect();
            let plan = Plan::new(size, &[]).unwrap();
            let mut guarded = vec![0xff; size + 2];
            let packet = plan.encode_verified(&raw, &mut guarded[1..=size], None).unwrap();
            guarded[1..=size].fill(0xa5);
            plan.decode_checked(
                &[&packet],
                &mut guarded[1..=size],
                crc32(&packet),
                crc32(&raw),
                None,
            )
            .unwrap();
            assert_eq!(&guarded[1..=size], raw);
            assert_eq!((guarded[0], guarded[size + 1]), (0xff, 0xff));
        }
    }

    #[test]
    fn large_row_matches_the_reference_packet() {
        let packet = include_bytes!("../../tests/fixtures/codec/large-row.sgc");
        let raw = vec![0u8; 9 << 20];
        let mut dst = vec![1; raw.len()];
        let plan = Plan::new(raw.len(), &[]).unwrap();
        assert!(plan.encode_verified(&raw, &mut dst, None).unwrap() == packet);
        plan.decode_checked(&[packet], &mut dst, 0x96bd12e4, 0x63214bad, None).unwrap();
        assert!(dst == raw);
    }

    #[test]
    fn canonical_packets_round_trip() {
        let fixtures: [&[u8]; 3] = [
            include_bytes!("../../tests/fixtures/codec/page0.sgc"),
            include_bytes!("../../tests/fixtures/codec/page1.sgc"),
            include_bytes!("../../tests/fixtures/codec/row.sgc"),
        ];
        for (i, expected) in fixtures.into_iter().enumerate() {
            let source: Vec<u8> = (0..4096)
                .map(|j| ((j / (8 + 2 * i) + i) % (2 + i)) as u8)
                .collect();
            let mut back = vec![255; 4096];
            let plan = Plan::new(source.len(), &[]).unwrap();
            let bytes = plan.encode_verified(&source, &mut back, None).unwrap();
            assert_eq!(bytes, expected);
            assert_eq!(source, back);
            let mut damaged = bytes.clone();
            damaged[16] ^= 1;
            assert!(plan.decode_checked(&[&damaged], &mut back, crc32(&damaged), crc32(&source), None).is_err());
        }
    }

    #[test]
    fn typed_units_preserve_bits_including_padding() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for (kind, size) in [
            (Encoding::Float32Matrix128, 65536),
            (Encoding::Latent528, 135168),
            (Encoding::Index132, 8448),
            (Encoding::Fp8Page512, 32768),
            (Encoding::Convolution20, 20000),
            (Encoding::Convolution32, 32768),
            (Encoding::Bfloat16, 6144),
            (Encoding::WindowFp4, 33792),
            (Encoding::LatentFp4, 18432),
        ] {
            let mut state = 2026u32;
            let mut raw: Vec<u8> = (0..size + 16)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 24) as u8
                })
                .collect();
            raw[4..20]
                .copy_from_slice(&[0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 128, 127, 37, 2, 192, 127]);
            let plan = Plan::new(
                raw.len(),
                &[Field {
                    offset: 4,
                    length: size,
                    encoding: kind,
                }],
            )
            .unwrap();
            let mut dst = vec![1; raw.len()];
            let packet = plan.encode_verified(&raw, &mut dst, None).unwrap();
            assert_eq!(raw, dst);
            dst.fill(0xa5);
            let parallel = plan.encode_verified(&raw, &mut dst, Some(&pool)).unwrap();
            assert_eq!(packet, parallel);
            assert_eq!(raw, dst);
            let mut scalar_output = vec![0; raw.len()];
            let ranges = plan.entries(&packet, packet.len()).unwrap();
            for (frame, range) in plan.frames.iter().zip(ranges) {
                let mut buffer = vec![0; frame.raw_bytes() + 4];
                let mut work = vec![0u32; frame.raw_bytes().div_ceil(4)];
                // The validated frame partitions raw and scalar_output. Both native calls
                // get a full frame of scratch and the exact packet capacity.
                let encoded = unsafe {
                    kv_cpu_frame_scalar(
                        frame,
                        raw.as_ptr().cast_mut(),
                        buffer.as_mut_ptr(),
                        buffer.len() as i32,
                        work.as_mut_ptr(),
                        false,
                    )
                };
                assert_eq!(encoded as usize, range.len());
                assert_eq!(&buffer[..range.len()], &packet[range.clone()]);
                let decoded = unsafe {
                    kv_cpu_frame_scalar(
                        frame,
                        scalar_output.as_mut_ptr(),
                        buffer.as_mut_ptr(),
                        encoded,
                        work.as_mut_ptr(),
                        true,
                    )
                };
                assert_eq!(decoded as usize, frame.raw_bytes());
            }
            assert_eq!(raw, scalar_output);
        }
    }
}
