/*
 * SPDX-License-Identifier: Apache-2.0
 */

use flare_selene::{
    PrefixKey,
    record::{self, DirectoryLock, Header, RankSizes},
};
use std::{fs, io::Write};
const DIRECTORY: &[u8] = include_bytes!("fixtures/codec/suffix-directory.pb");
const PAGE0: &[u8] = include_bytes!("fixtures/codec/page0.sgc");
const PAGE1: &[u8] = include_bytes!("fixtures/codec/page1.sgc");
const ROW: &[u8] = include_bytes!("fixtures/codec/row.sgc");

fn key(n: u64) -> PrefixKey {
    PrefixKey {
        consumed_tokens: n * 64,
        digest: [n as u8; 16],
    }
}

#[test]
fn scan_recovers_one_chain_spanning_v1_v2_v3_and_v4() {
    let directory =
        std::env::temp_dir().join(format!("selene-mixed-codec-root-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let lock = DirectoryLock::acquire(&directory).unwrap();
    let ranks = vec![RankSizes {
        planes: vec![4000],
        row_bytes: 1000,
    }];
    let header = |n: u64, p: Option<u64>| {
        Header::from_sizes(
            64,
            ranks.clone(),
            [1; 32],
            [2; 16],
            key(n),
            p.map(key),
            [3; 16],
            (n - p.unwrap_or(0)) as u32,
            0,
        )
        .unwrap()
    };
    for version in 1..=3u32 {
        let n = u64::from(version);
        let h = header(n, (n > 1).then_some(n - 1));
        let mut block = h.encode().unwrap();
        block[8..12].copy_from_slice(&version.to_le_bytes());
        if version <= 2 {
            if version == 2 {
                block[12..16].copy_from_slice(&(1u32 << 31).to_le_bytes())
            }
            block[181..189].copy_from_slice(&4000u64.to_le_bytes());
            block[189..197].copy_from_slice(&1000u64.to_le_bytes());
        }
        let mut f = fs::File::create(directory.join(record::file_name(&h.key))).unwrap();
        f.write_all(&block).unwrap();
        f.set_len(h.length(true)).unwrap();
    }
    let h = header(5, Some(3));
    let mut fixed = h.encode().unwrap();
    fixed[8..12].copy_from_slice(&4u32.to_le_bytes());
    let pages = 4096 + DIRECTORY.len() as u64;
    let row = pages + (PAGE0.len() + PAGE1.len()) as u64;
    fixed[141..149].copy_from_slice(&pages.to_le_bytes());
    fixed[149..157].copy_from_slice(&row.to_le_bytes());
    fixed[173..].fill(0);
    fixed[173..181].copy_from_slice(&4096u64.to_le_bytes());
    fixed[181..185].copy_from_slice(&(DIRECTORY.len() as u32).to_le_bytes());
    fixed[185..189].copy_from_slice(&0x500789bbu32.to_le_bytes());
    fixed[189..197].copy_from_slice(&(row + ROW.len() as u64).to_le_bytes());
    let file = directory.join(record::file_name(&h.key));
    fs::write(&file, [&fixed[..], DIRECTORY, PAGE0, PAGE1, ROW].concat()).unwrap();
    let result = record::scan(&lock, &[1; 32], &[2; 16], 64, &ranks).unwrap();
    let survived = file.is_file();
    let keys: Vec<_> = result.iter().map(|r| (r.header.key, r.row)).collect();
    drop(lock);
    fs::remove_dir_all(&directory).unwrap();
    assert!(survived, "the scanner deleted the v4 checkpoint");
    assert_eq!(
        keys,
        [
            (key(1), true),
            (key(2), true),
            (key(3), true),
            (key(5), true)
        ]
    );
}
