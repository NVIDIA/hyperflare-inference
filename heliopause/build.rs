/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

// Pure-std FIPS 180-4 SHA-256 matching heliopause/src/export/source_identity.rs
// without adding build-dependencies.
struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    total_bytes: u64,
}

impl Sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                0x1f83d9ab, 0x5be0cd19,
            ],
            block: [0; 64],
            block_len: 0,
            total_bytes: 0,
        }
    }

    fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(Self::K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, delta) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(delta);
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_bytes += data.len() as u64;
        if self.block_len > 0 {
            let take = (64 - self.block_len).min(data.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&data[..take]);
            self.block_len += take;
            data = &data[take..];
            if self.block_len == 64 {
                Self::compress(&mut self.state, &self.block);
                self.block_len = 0;
            }
        }
        while data.len() >= 64 {
            let chunk: &[u8; 64] = data[..64].try_into().unwrap();
            Self::compress(&mut self.state, chunk);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.block[..data.len()].copy_from_slice(data);
            self.block_len = data.len();
        }
    }

    fn finalize_hex(mut self) -> String {
        let bits = self.total_bytes * 8;
        self.update(&[0x80]);
        if self.block_len > 56 {
            self.block[self.block_len..].fill(0);
            Self::compress(&mut self.state, &self.block);
            self.block_len = 0;
        }
        self.block[self.block_len..56].fill(0);
        self.block[56..64].copy_from_slice(&bits.to_be_bytes());
        Self::compress(&mut self.state, &self.block);
        let mut out = String::with_capacity(64);
        for word in self.state {
            use std::fmt::Write;
            let _ = write!(&mut out, "{word:08x}");
        }
        out
    }
}

fn collect_sources(directory: &Path, files: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", directory.display());
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect_sources(&path, files);
        } else if path.is_file()
            && path
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|value| matches!(value, "py" | "rs" | "c" | "cu" | "cuh" | "h"))
        {
            println!("cargo:rerun-if-changed={}", path.display());
            files.push(path);
        }
    }
}

fn sort_key(path: &Path) -> Vec<String> {
    path.components()
        .map(|part| {
            let value = part.as_os_str().to_str().expect("UTF-8 source path");
            #[cfg(windows)]
            let value = value.to_lowercase();
            #[cfg(not(windows))]
            let value = value.to_owned();
            value
        })
        .collect()
}

fn hash_tree(root: &Path) -> String {
    let mut files = Vec::new();
    collect_sources(root, &mut files);
    let mut keyed: Vec<(Vec<String>, PathBuf)> = files
        .into_iter()
        .map(|path| (sort_key(&path), path))
        .collect();
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 << 10];
    for (_, path) in keyed {
        let relative = path.strip_prefix(root).expect("path under root");
        digest.update(relative.to_str().expect("UTF-8 relative path").as_bytes());
        let mut file = File::open(&path).expect("open source file");
        loop {
            let count = file.read(&mut buffer).expect("read source file");
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
    }
    digest.finalize_hex()
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(openssl3)");
    if std::env::var("DEP_OPENSSL_VERSION_NUMBER")
        .ok()
        .and_then(|value| u64::from_str_radix(&value, 16).ok())
        .is_some_and(|version| version >= 0x3000_0000)
    {
        println!("cargo:rustc-cfg=openssl3");
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let heliopause_src = manifest_dir.join("src");
    let helios_src = manifest_dir.join("../helios/src");
    println!(
        "cargo:rustc-env=FLARE_HELIOPAUSE_SOURCE_IDENTITY={}",
        hash_tree(&heliopause_src)
    );
    println!(
        "cargo:rustc-env=FLARE_HELIOS_SOURCE_IDENTITY={}",
        hash_tree(&helios_src)
    );
}
