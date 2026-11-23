/*
 * SPDX-License-Identifier: Apache-2.0
 */

mod support;

use flare_selene::shared::{packed, record::Header};
use selene_manager::proto::v2;
use support::key;

const PAGE0: &[u8] = include_bytes!("../../selene/tests/fixtures/codec/page0.sgc");
const PAGE1: &[u8] = include_bytes!("../../selene/tests/fixtures/codec/page1.sgc");
const ROW: &[u8] = include_bytes!("../../selene/tests/fixtures/codec/row.sgc");

fn payload(unit: &v2::PackedUnit) -> &'static [u8] {
    match unit.original_crc {
        0x476cc8d3 => PAGE0,
        0xed7164fa => PAGE1,
        0xb2b29b40 => ROW,
        _ => panic!("unexpected unit"),
    }
}

fn offer() -> v2::PackedOffer {
    let units = [(0, 0, 0x476cc8d3), (0, 1, 0xed7164fa), (1, 0, 0xb2b29b40)]
        .into_iter()
        .map(|(kind, logical_index, original_crc)| {
            let mut u = v2::PackedUnit {
                kind,
                logical_index,
                original_crc,
                representation: 1,
                profile: vec![3; 32],
                original_length: 4096,
                ..Default::default()
            };
            let bytes = payload(&u);
            u.stored_crc = packed::crc32(bytes);
            u.stored_length = bytes.len() as u64;
            u
        })
        .collect();
    v2::PackedOffer {
        first_page: 0,
        units,
    }
}

#[test]
fn directory_bounds_integrity_and_physical_window_limits() {
    let ranks = support::descriptor().ranks;
    let offer = offer();
    for mutate in [
        |o: &mut v2::PackedOffer| o.units[0].stored_length = u64::MAX,
        |o: &mut v2::PackedOffer| o.units[0].original_length = 17,
        |o: &mut v2::PackedOffer| o.units[0].replicas.push(0),
        |o: &mut v2::PackedOffer| o.units.swap(0, 2),
        |o: &mut v2::PackedOffer| o.units[0].profile.clear(),
        |o: &mut v2::PackedOffer| o.units[0].raw_xxh3 = 1,
    ] {
        let mut bad = offer.clone();
        mutate(&mut bad);
        assert!(packed::Layout::new(bad, &ranks, 0, 2).is_err());
    }
    let layout = packed::Layout::new(offer.clone(), &ranks, 0, 2).unwrap();
    assert_eq!(
        layout
            .windows(0, true, 1000)
            .iter()
            .map(|w| w.length)
            .collect::<Vec<_>>(),
        [704, 994, 1208]
    );
    assert_eq!(layout.windows(1, true, 2000)[0].offset, 704);
    let header =
        selene_manager::record::new_header(&support::descriptor(), key(128), None, [4; 16], 2)
            .unwrap()
            .with_packed(offer.clone())
            .unwrap();
    let metadata = header.packed_metadata().unwrap();
    assert!(Header::decode(&metadata[..metadata.len() - 1]).is_err());
    let mut corrupt = metadata.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(Header::decode(&corrupt).is_err());
    let mut huge = metadata;
    huge[181..185].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Header::read_metadata(&*huge).is_err());
    let mut damaged = PAGE1.to_vec();
    damaged[70] ^= 1;
    assert!(packed::verify_units(&offer.units[1..2], &damaged).is_err());
    let mut literal = offer.units[0].clone();
    literal.representation = 0;
    literal.profile.clear();
    literal.stored_crc = 0;
    literal.original_crc = 0;
    literal.stored_length = 4096;
    let bytes = vec![17; 4096];
    literal.raw_xxh3 = flare_selene::record::checksum(&bytes);
    let mut mixed = offer;
    mixed.units[0] = literal;
    let layout = packed::Layout::new(mixed, &ranks, 0, 2).unwrap();
    assert!(packed::verify_units(layout.units(0, 1, false).unwrap(), &bytes).is_ok());
    assert!(packed::verify_units(layout.units(0, 1, false).unwrap(), &vec![18; 4096]).is_err());
}
