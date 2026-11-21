//! GoldSrc Sprite (`.spr`) legacy adapter for FlatTexture.
//!
//! Decodes GoldSrc Sprite v2 files into direct-to-GPU `.flt` texture arrays.

use anyhow::{bail, Result};
use flatasset_texture::{FlatTextureBuilder, MipInput, VkFormat};

pub const SPRITE_MAGIC: [u8; 4] = *b"IDSP";
pub const SPRITE_VERSION_HL: u32 = 2;

#[allow(dead_code)]
pub const SPR_NORMAL: u32 = 0;
pub const SPR_ADDITIVE: u32 = 1;
pub const SPR_INDEXALPHA: u32 = 2;
pub const SPR_ALPHTEST: u32 = 3;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SpriteFrame {
    pub origin_x: i32,
    pub origin_y: i32,
    pub width: u32,
    pub height: u32,
    pub rgba_pixels: Vec<u8>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SpriteData {
    pub sprite_type: u32,
    pub tex_format: u32,
    pub bounding_radius: f32,
    pub max_width: u32,
    pub max_height: u32,
    pub frames: Vec<SpriteFrame>,
}

pub fn parse_spr(bytes: &[u8]) -> Result<SpriteData> {
    if bytes.len() < 40 {
        bail!("Buffer too small for SPR header: {} bytes", bytes.len());
    }

    if bytes[0..4] != SPRITE_MAGIC {
        bail!("Invalid SPR magic: {:?}", &bytes[0..4]);
    }

    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != SPRITE_VERSION_HL {
        bail!("Unsupported SPR version: {}", version);
    }

    let sprite_type = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let tex_format = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let bounding_radius = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let width = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
    let height = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
    let num_frames = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;

    let mut offset = 40usize;

    if offset + 2 > bytes.len() {
        bail!("Missing palette count in SPR");
    }
    let palette_count = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as usize;
    offset += 2;

    let palette_bytes_len = palette_count * 3;
    if offset + palette_bytes_len > bytes.len() {
        bail!("Palette bytes exceed buffer in SPR");
    }

    let palette = &bytes[offset..offset + palette_bytes_len];
    offset += palette_bytes_len;

    let mut frames = Vec::with_capacity(num_frames);

    for _ in 0..num_frames {
        if offset + 4 > bytes.len() {
            break;
        }
        let frame_type = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;

        if frame_type == 0 {
            // SINGLE FRAME
            if offset + 16 > bytes.len() {
                break;
            }
            let origin_x = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            let origin_y = i32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
            let w = u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap());
            let h = u32::from_le_bytes(bytes[offset + 12..offset + 16].try_into().unwrap());
            offset += 16;

            let pixel_count = (w as usize) * (h as usize);
            if offset + pixel_count > bytes.len() {
                break;
            }

            let index_data = &bytes[offset..offset + pixel_count];
            offset += pixel_count;

            let rgba_pixels = decode_pixels_to_rgba(index_data, palette, tex_format);
            frames.push(SpriteFrame {
                origin_x,
                origin_y,
                width: w,
                height: h,
                rgba_pixels,
            });
        } else {
            // GROUP FRAME
            if offset + 4 > bytes.len() {
                break;
            }
            let group_num =
                i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            // Skip interval floats
            offset += group_num * 4;

            for _ in 0..group_num {
                if offset + 16 > bytes.len() {
                    break;
                }
                let origin_x = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                let origin_y =
                    i32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
                let w = u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap());
                let h = u32::from_le_bytes(bytes[offset + 12..offset + 16].try_into().unwrap());
                offset += 16;

                let pixel_count = (w as usize) * (h as usize);
                if offset + pixel_count > bytes.len() {
                    break;
                }
                let index_data = &bytes[offset..offset + pixel_count];
                offset += pixel_count;

                let rgba_pixels = decode_pixels_to_rgba(index_data, palette, tex_format);
                frames.push(SpriteFrame {
                    origin_x,
                    origin_y,
                    width: w,
                    height: h,
                    rgba_pixels,
                });
            }
        }
    }

    Ok(SpriteData {
        sprite_type,
        tex_format,
        bounding_radius,
        max_width: width,
        max_height: height,
        frames,
    })
}

fn decode_pixels_to_rgba(indices: &[u8], palette: &[u8], tex_format: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(indices.len() * 4);
    let pal_colors = palette.len() / 3;

    let last_pal_idx = pal_colors.saturating_sub(1) * 3;
    let base_rgb = if last_pal_idx + 3 <= palette.len() {
        [
            palette[last_pal_idx],
            palette[last_pal_idx + 1],
            palette[last_pal_idx + 2],
        ]
    } else {
        [255, 255, 255]
    };

    for &idx in indices {
        let p_idx = (idx as usize) * 3;
        match tex_format {
            SPR_INDEXALPHA => {
                out.extend_from_slice(&[base_rgb[0], base_rgb[1], base_rgb[2], idx]);
            }
            SPR_ALPHTEST => {
                if idx == 255 {
                    out.extend_from_slice(&[0, 0, 0, 0]);
                } else if p_idx + 3 <= palette.len() {
                    out.extend_from_slice(&[
                        palette[p_idx],
                        palette[p_idx + 1],
                        palette[p_idx + 2],
                        255,
                    ]);
                } else {
                    out.extend_from_slice(&[0, 0, 0, 0]);
                }
            }
            SPR_ADDITIVE => {
                if p_idx + 3 <= palette.len() {
                    let r = palette[p_idx];
                    let g = palette[p_idx + 1];
                    let b = palette[p_idx + 2];
                    let a = (u16::from(r) + u16::from(g) + u16::from(b)) / 3;
                    out.extend_from_slice(&[r, g, b, a as u8]);
                } else {
                    out.extend_from_slice(&[0, 0, 0, 0]);
                }
            }
            _ => {
                if p_idx + 3 <= palette.len() {
                    out.extend_from_slice(&[
                        palette[p_idx],
                        palette[p_idx + 1],
                        palette[p_idx + 2],
                        255,
                    ]);
                } else {
                    out.extend_from_slice(&[0, 0, 0, 255]);
                }
            }
        }
    }

    out
}

/// Convert parsed Sprite frames into a Direct-to-GPU `.flt` texture package.
pub fn convert_spr_to_flt(bytes: &[u8]) -> Result<Vec<u8>> {
    let spr = parse_spr(bytes)?;
    if spr.frames.is_empty() {
        bail!("Sprite contains 0 frames");
    }

    // Use primary frame for the 2D texture representation
    let frame = &spr.frames[0];
    let (bc1_bytes, row_pitch, rows) =
        flatasset_texture::encode_rgba_to_bc1_dma(frame.width, frame.height, &frame.rgba_pixels);

    let mut builder = FlatTextureBuilder::new_2d(
        frame.width as u16,
        frame.height as u16,
        VkFormat::Bc1RgbUnormBlock,
    );
    builder.add_mip(MipInput {
        width: frame.width,
        height: frame.height,
        dma_bytes: bc1_bytes,
        aligned_row_pitch: row_pitch,
        rows_per_image: rows,
    });

    builder
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build FlatTexture from sprite: {:?}", e))
}
