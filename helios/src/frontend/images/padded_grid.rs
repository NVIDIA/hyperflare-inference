/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unresized padded-grid preparation.
//!
//! Images keep their decoded size. Retained alpha is composited onto an 8-pixel
//! checkerboard of 204 and 153 as `(rgb·a + background·(255 − a) + 127) / 255`. The
//! image is padded on the bottom and right to whole 40-pixel token cells with −1 on the
//! 0–255 scale, and every value is normalized as `(v − 127.5) / 127.5`. Each cell holds
//! 2×2 native 20-pixel patches. Patches follow token raster order and, within a token,
//! merge-block row-major order; each patch has 1,200 features in pixel row, pixel column,
//! channel order. Grids count native patches. A learned 32×32 position table is sampled
//! with four bilinear taps at half-pixel centres (`align_corners=False`), clamped to the
//! table edge. Sequence offsets count one class token before each image's patches.

use std::borrow::Cow;

use super::{
    ImageGeometry, ImageLimits, ProcessedImages, RgbImage, check_image_patches,
    check_input_extent,
};
use crate::Result;

#[cfg(test)]
#[path = "../../../tests/support/frontend.rs"]
mod test_support;

const PATCH: usize = 20;
const MERGE: usize = 2;
const CELL: usize = PATCH * MERGE;
const POSITION_SIDE: usize = 32;
pub(super) const FEATURES: usize = PATCH * PATCH * 3;
const CHECKER_SQUARE: usize = 8;
const CHECKER_LIGHT: u32 = 204;
const CHECKER_DARK: u32 = 153;
const PAD_VALUE: f32 = -1.0;

pub(super) fn geometry(limits: ImageLimits, width: usize, height: usize) -> Result<ImageGeometry> {
    check_input_extent(limits, width, height)?;
    let (rows, columns) = (height.div_ceil(CELL), width.div_ceil(CELL));
    let tokens = rows.checked_mul(columns).ok_or("image token count overflows")?;
    let patches = tokens
        .checked_mul(MERGE * MERGE)
        .ok_or("image patch count overflows")?;
    check_image_patches(limits, patches)?;
    Ok(ImageGeometry {
        height: rows * CELL,
        width: columns * CELL,
        grid_thw: [1, (rows * MERGE) as i64, (columns * MERGE) as i64],
        patches,
        tokens,
    })
}

/// RGB with retained alpha composited onto the checkerboard anchored at the top-left pixel.
fn composited(image: &RgbImage) -> Result<Cow<'_, [u8]>> {
    let Some(alpha) = &image.alpha else {
        return Ok(Cow::Borrowed(&image.pixels));
    };
    if alpha.len() != image.width * image.height {
        return Err("image alpha extent differs from its pixels".into());
    }
    let mut pixels = Vec::with_capacity(image.pixels.len());
    for (index, (rgb, alpha)) in image.pixels.chunks_exact(3).zip(alpha).enumerate() {
        let (y, x) = (index / image.width, index % image.width);
        let background = if (y / CHECKER_SQUARE + x / CHECKER_SQUARE) % 2 == 0 {
            CHECKER_LIGHT
        } else {
            CHECKER_DARK
        };
        let alpha = u32::from(*alpha);
        pixels.extend(
            rgb.iter()
                .map(|value| ((u32::from(*value) * alpha + background * (255 - alpha) + 127) / 255) as u8),
        );
    }
    Ok(Cow::Owned(pixels))
}

/// Two taps and weights along one axis of `size` native patches, in the float32 operation
/// order of Transformers `_interpolation_axis_taps_weights` with `align_corners=False`.
fn interpolation_axis(index: usize, size: usize) -> ([usize; 2], [f32; 2]) {
    let source = (index as f32 + 0.5) * POSITION_SIDE as f32 / size as f32 - 0.5;
    let floor = source.floor();
    let distance = source - floor;
    let tap = |offset: i64| (floor as i64 + offset).clamp(0, POSITION_SIDE as i64 - 1) as usize;
    (
        [tap(0), tap(1)],
        [(1.0 - distance).max(0.0), (1.0 - (distance - 1.0).abs()).max(0.0)],
    )
}

pub(super) fn process(limits: ImageLimits, images: &[RgbImage]) -> Result<ProcessedImages> {
    limits.check_batch(images.len(), 0)?;
    let geometry = images
        .iter()
        .map(|image| geometry(limits, image.width, image.height))
        .collect::<Result<Vec<_>>>()?;
    let patches = geometry
        .iter()
        .try_fold(0usize, |sum, image| sum.checked_add(image.patches))
        .ok_or("image patch count overflows")?;
    limits.check_batch(images.len(), patches)?;
    let mut result = ProcessedImages {
        geometry,
        cu_seqlens: vec![0],
        ..Default::default()
    };
    result.reserve_outputs(patches, FEATURES, true, images.len(), images.len() + 1)?;
    let normalized: [f32; 256] = std::array::from_fn(|value| (value as f32 - 127.5) / 127.5);
    let padding = (PAD_VALUE - 127.5) / 127.5;
    for (image, geometry) in images.iter().zip(&result.geometry) {
        let pixels = composited(image)?;
        let grid_h = geometry.height / PATCH;
        let grid_w = geometry.width / PATCH;
        for block_row in 0..grid_h / MERGE {
            for block_col in 0..grid_w / MERGE {
                for dy in 0..MERGE {
                    for dx in 0..MERGE {
                        let (row, col) = (block_row * MERGE + dy, block_col * MERGE + dx);
                        result.position_ids.push([row as i64, col as i64]);
                        let (rows, row_weights) = interpolation_axis(row, grid_h);
                        let (cols, col_weights) = interpolation_axis(col, grid_w);
                        result.interpolation_indices.push([
                            (rows[0] * POSITION_SIDE + cols[0]) as i64,
                            (rows[0] * POSITION_SIDE + cols[1]) as i64,
                            (rows[1] * POSITION_SIDE + cols[0]) as i64,
                            (rows[1] * POSITION_SIDE + cols[1]) as i64,
                        ]);
                        result.interpolation_weights.push([
                            row_weights[0] * col_weights[0],
                            row_weights[0] * col_weights[1],
                            row_weights[1] * col_weights[0],
                            row_weights[1] * col_weights[1],
                        ]);
                        for y in row * PATCH..(row + 1) * PATCH {
                            for x in col * PATCH..(col + 1) * PATCH {
                                if y < image.height && x < image.width {
                                    let offset = (y * image.width + x) * 3;
                                    result.pixel_values.extend(
                                        pixels[offset..offset + 3]
                                            .iter()
                                            .map(|value| normalized[*value as usize]),
                                    );
                                } else {
                                    result.pixel_values.extend([padding; 3]);
                                }
                            }
                        }
                    }
                }
            }
        }
        let sequence = i32::try_from(geometry.patches + 1)
            .ok()
            .and_then(|length| result.cu_seqlens.last().unwrap().checked_add(length))
            .ok_or("image sequence offsets exceed int32")?;
        result.cu_seqlens.push(sequence);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::images::{ImageProcessor, ImageSettings};
    use serde_json::Value;
    use std::{fs, path::PathBuf};

    fn fixture_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/frontend")
    }

    fn fixture() -> Value {
        serde_json::from_slice(&fs::read(fixture_root().join("inkling2_images.json")).unwrap()).unwrap()
    }

    fn processor() -> ImageProcessor {
        ImageProcessor::load(
            ImageSettings::PaddedGrid {},
            ImageLimits {
                max_patches_per_image: 42436,
                max_total_patches: 65536,
                patches_per_feature: 4,
                ..ImageLimits::default()
            },
        ).unwrap()
    }

    fn pattern(width: usize, height: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        (0..width * height * 3).map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        }).collect()
    }

    fn inputs(processor: &ImageProcessor, case: &Value) -> Vec<RgbImage> {
        case["inputs"].as_array().unwrap().iter().map(|input| {
            if let Some(pattern_case) = input.get("pattern") {
                let [width, height, seed] = ["width", "height", "seed"]
                    .map(|key| pattern_case[key].as_u64().unwrap() as usize);
                return RgbImage::new(width, height, pattern(width, height, seed as u32)).unwrap();
            }
            let bytes = fs::read(fixture_root().join(input["file"].as_str().unwrap())).unwrap();
            let image = match input["format"].as_str().unwrap() {
                "PNG" => processor.decode_png(&bytes),
                "JPEG" => processor.decode_jpeg(&bytes),
                "WEBP" => processor.decode_webp(&bytes),
                format => panic!("unexpected fixture format {format}"),
            }.unwrap();
            assert_eq!([image.width(), image.height()],
                [input["width"].as_u64().unwrap() as usize, input["height"].as_u64().unwrap() as usize]);
            image
        }).collect()
    }

    #[test]
    fn reference_processor_pixels_order_positions_and_interpolation_match() {
        let fixture = fixture();
        let processor = processor();
        let mut config = test_support::model_config(crate::model::testing::metadata());
        config.contract.frontend.image_layout = Some(crate::model::contract::ImageLayoutKind::ClassToken);
        config.contract.vision.features_per_request = 16384;
        config.contract.vision.patches_per_feature = 4;
        config.contract.vision.patch_capacities = vec![65536];
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 24);
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let images = inputs(&processor, case);
            for (index, image) in images.iter().enumerate() {
                let expected = fs::read(fixture_root().join(case["composited_rgb"][index].as_str().unwrap())).unwrap();
                let actual = composited(image).unwrap();
                assert_eq!(actual.len(), expected.len(), "{name}");
                let error = actual.iter().zip(&expected).map(|(a,b)| a.abs_diff(*b)).max().unwrap();
                let tolerance = case["inputs"][index]["max_channel_error"].as_u64().unwrap_or(0);
                assert!(u64::from(error) <= tolerance, "{name} decoded/composited channel error {error}");
            }
            let processed = processor.process(&images).unwrap();
            let grids: Vec<_> = processed.geometry.iter()
                .map(|g| [1, g.grid_thw[1] / 2, g.grid_thw[2] / 2]).collect();
            assert_eq!(serde_json::json!(grids), case["grid_thw"], "{name}");
            assert_eq!(serde_json::json!(processed.geometry.iter().map(|g| g.tokens).collect::<Vec<_>>()),
                case["tokens"], "{name}");
            assert_eq!(serde_json::json!(processed.cu_seqlens), case["cu_seqlens"], "{name}");
            let bytes = fs::read(fixture_root().join(case["pixel_values"].as_str().unwrap())).unwrap();
            let expected: Vec<f32> = bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect();
            assert_eq!(processed.pixel_values.len(), expected.len(), "{name}");
            let channel_error = case["inputs"].as_array().unwrap().iter()
                .map(|v| v["max_channel_error"].as_u64().unwrap_or(0)).max().unwrap() as f32;
            let error = processed.pixel_values.iter().zip(&expected)
                .map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
            if channel_error == 0.0 {
                assert_eq!(processed.pixel_values, expected, "{name} pixels");
            } else {
                assert!(error <= channel_error / 127.5 + f32::EPSILON, "{name} normalized pixel error {error}");
            }
            let positions: Vec<[i64; 2]> = serde_json::from_value(case["position_ids"].clone()).unwrap();
            let indices: Vec<[i64; 4]> = serde_json::from_value(case["interpolation_indices"].clone()).unwrap();
            let weights: Vec<[f32; 4]> = serde_json::from_value(case["interpolation_weights"].clone()).unwrap();
            assert_eq!(processed.position_ids, positions, "{name} patch positions");
            assert_eq!(processed.interpolation_indices, indices, "{name} interpolation indices");
            assert_eq!(processed.interpolation_weights, weights, "{name} interpolation weights");
            let mut rotary = vec![];
            let mut start = 0;
            for geometry in &processed.geometry {
                rotary.push([0,0]);
                rotary.extend(processed.position_ids[start..start+geometry.patches]
                    .iter().map(|[r,c]| [r+1,c+1]));
                start += geometry.patches;
            }
            assert_eq!(serde_json::json!(rotary), case["rotary_positions"], "{name} class positions");
            let mut start = 0;
            let mut sequence = 0;
            for (image, geometry) in processed.geometry.iter().enumerate() {
                let staging = super::super::packing::prepare(&config,
                    crate::scheduler::RequestHandle {slot: 0, generation: 1}, &processed, image, 7).unwrap();
                let uploads = staging.uploads();
                let bytes = |name: &str| uploads.iter().find(|v| v.buffer == name).unwrap().bytes;
                let packed_positions: Vec<i64> = bytes("vision_position_ids").chunks_exact(8)
                    .map(|v| i64::from_le_bytes(v.try_into().unwrap())).collect();
                assert_eq!(packed_positions, rotary[sequence..sequence+geometry.patches+1]
                    .iter().flatten().copied().collect::<Vec<_>>(), "{name} uploaded class positions");
                for (key, expected) in [("vision_num_features", geometry.tokens as i32),
                    ("vision_num_patches", geometry.patches as i32), ("vision_feature_offset", 7)] {
                    assert_eq!(bytes(key), &expected.to_le_bytes(), "{name} {key}");
                }
                assert_eq!(bytes("vision_cu_seqlens"), [0i32, geometry.patches as i32 + 1]
                    .into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>(), "{name} uploaded sequence");
                assert_eq!(bytes("vision_pixel_values"), processed.pixel_values[start*FEATURES..(start+geometry.patches)*FEATURES]
                    .iter().copied().flat_map(f32::to_le_bytes).collect::<Vec<_>>(), "{name} uploaded pixels");
                start += geometry.patches;
                sequence += geometry.patches + 1;
            }
        }
    }

    #[test]
    fn transparency_and_orientation_follow_the_reference_decoder() {
        let root = fixture_root().join("inkling2_images-inputs");
        let processor = processor();
        for (file, alpha) in [("RGB.png", false), ("RGBA.png", true), ("LA.png", true), ("P.png", false),
            ("P-tRNS.png", true), ("RGB-tRNS.png", true), ("L-tRNS.png", true), ("RGBA-orientation-6.png", true)] {
            let image = processor.decode_png(&fs::read(root.join(file)).unwrap()).unwrap();
            assert_eq!(image.alpha.is_some(), alpha, "{file}");
            assert_eq!((image.width(), image.height()), (53, 47), "{file}");
        }
        let webp = processor.decode_webp(&fs::read(root.join("RGBA-lossless.webp")).unwrap()).unwrap();
        assert!(webp.alpha.is_some());
        let jpeg = processor.decode_jpeg(&fs::read(root.join("RGB-orientation-6.jpg")).unwrap()).unwrap();
        assert_eq!((jpeg.width(), jpeg.height()), (61, 29));
    }

    #[test]
    fn images_differing_only_in_alpha_have_distinct_identities() {
        let processor = processor();
        let mut image = RgbImage::new(9, 9, vec![80; 9 * 9 * 3]).unwrap();
        let opaque = processor.semantic_identity(&image);
        image.alpha = Some(vec![255; 81]);
        let alpha = processor.semantic_identity(&image);
        assert_ne!(opaque, alpha);
        image.alpha.as_mut().unwrap()[80] = 0;
        assert_ne!(alpha, processor.semantic_identity(&image));
        let transparent = processor.process(std::slice::from_ref(&image)).unwrap();
        image.alpha.as_mut().unwrap()[80] = 255;
        assert_ne!(transparent.pixel_values, processor.process(&[image]).unwrap().pixel_values);
    }

    #[test]
    fn size_is_bounded_by_decoded_pixels_and_prepared_tokens_without_resizing() {
        let processor = processor();
        let geometry = processor.geometry(4096, 4096).unwrap();
        assert_eq!((geometry.tokens, geometry.patches, geometry.grid_thw), (10609, 42436, [1, 206, 206]));
        let tall = processor.geometry(1, 41).unwrap();
        assert_eq!((tall.tokens, tall.width, tall.height), (2, 40, 80));
        assert_eq!(processor.geometry(8000, 1600).unwrap().tokens, 8000);
        let error = processor.geometry(4121, 4096).unwrap_err().to_string();
        assert!(error.contains("image needs 10712 image tokens") && error.contains("max_image_tokens_per_image is 10609"),
            "{error}");
        assert!(processor.geometry(0, 7).unwrap_err().to_string().contains("must be positive"));
        let small = ImageProcessor::load(ImageSettings::PaddedGrid {},
            ImageLimits { max_input_pixels: 1600, ..ImageLimits::default() }).unwrap();
        assert!(small.geometry(40, 40).is_ok());
        assert!(small.geometry(41, 40).unwrap_err().to_string().contains("max_image_pixels"));
        assert!(processor.resized_rgb(&RgbImage::new(1, 1, vec![0; 3]).unwrap()).is_err());
        let mut truncated = RgbImage::new(2, 1, vec![0; 6]).unwrap();
        truncated.alpha = Some(vec![0]);
        assert!(processor.process(&[truncated]).is_err());
    }
}
