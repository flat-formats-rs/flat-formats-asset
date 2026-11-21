//! High-performance algorithmic BC1 (DXT1) Block Encoder.
//!
//! Encodes 4x4 texel blocks of RGBA8 into standard 8-byte BC1 blocks:
//! - 2 bytes: Color 0 (RGB565)
//! - 2 bytes: Color 1 (RGB565)
//! - 4 bytes: 16x 2-bit color index map (32 bits)
//!
//! Zero-allocation, hardware-friendly, deterministic encoding.

#[inline(always)]
fn rgb8_to_565(r: u8, g: u8, b: u8) -> u16 {
    let r5 = (r as u16 >> 3) & 0x1F;
    let g6 = (g as u16 >> 2) & 0x3F;
    let b5 = (b as u16 >> 3) & 0x1F;
    (r5 << 11) | (g6 << 5) | b5
}

#[inline(always)]
fn rgb565_to_888(c: u16) -> (u8, u8, u8) {
    let r5 = ((c >> 11) & 0x1F) as u8;
    let g6 = ((c >> 5) & 0x3F) as u8;
    let b5 = (c & 0x1F) as u8;
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g6 << 2) | (g6 >> 4);
    let b = (b5 << 3) | (b5 >> 2);
    (r, g, b)
}

#[inline(always)]
fn color_dist_sq(r1: i32, g1: i32, b1: i32, r2: i32, g2: i32, b2: i32) -> i32 {
    let dr = r1 - r2;
    let dg = g1 - g2;
    let db = b1 - b2;
    // Perceptual weighting approximation: 2*dr^2 + 4*dg^2 + db^2
    2 * dr * dr + 4 * dg * dg + db * db
}

/// Encodes a single 4x4 RGBA8 block into an 8-byte BC1 block.
pub fn encode_bc1_block(block_rgba: &[[u8; 4]; 16]) -> [u8; 8] {
    // 1. Find min and max colors across the 16 texels (bounding box in RGB space)
    let mut min_r = 255i32;
    let mut min_g = 255i32;
    let mut min_b = 255i32;
    let mut max_r = 0i32;
    let mut max_g = 0i32;
    let mut max_b = 0i32;

    for p in block_rgba {
        let r = p[0] as i32;
        let g = p[1] as i32;
        let b = p[2] as i32;
        min_r = min_r.min(r);
        min_g = min_g.min(g);
        min_b = min_b.min(b);
        max_r = max_r.max(r);
        max_g = max_g.max(g);
        max_b = max_b.max(b);
    }

    let mut c0_565 = rgb8_to_565(max_r as u8, max_g as u8, max_b as u8);
    let mut c1_565 = rgb8_to_565(min_r as u8, min_g as u8, min_b as u8);

    // In BC1, c0 > c1 selects the 4-color opaque mode (c0, c1, 2/3*c0 + 1/3*c1, 1/3*c0 + 2/3*c1)
    if c0_565 <= c1_565 {
        if c0_565 == c1_565 {
            if c0_565 < 0xFFFF {
                c0_565 += 1;
            } else {
                c1_565 = c1_565.saturating_sub(1);
            }
        } else {
            core::mem::swap(&mut c0_565, &mut c1_565);
        }
    }

    // Reconstruct exact palette RGB888 from the 16-bit 565 endpoints
    let (r0, g0, b0) = rgb565_to_888(c0_565);
    let (r1, g1, b1) = rgb565_to_888(c1_565);

    let palette: [(i32, i32, i32); 4] = [
        (r0 as i32, g0 as i32, b0 as i32),
        (r1 as i32, g1 as i32, b1 as i32),
        (
            (2 * r0 as i32 + r1 as i32) / 3,
            (2 * g0 as i32 + g1 as i32) / 3,
            (2 * b0 as i32 + b1 as i32) / 3,
        ),
        (
            (r0 as i32 + 2 * r1 as i32) / 3,
            (g0 as i32 + 2 * g1 as i32) / 3,
            (b0 as i32 + 2 * b1 as i32) / 3,
        ),
    ];

    // 2. Select best matching index for each of the 16 pixels
    let mut indices_u32 = 0u32;
    for (i, p) in block_rgba.iter().enumerate() {
        let pr = p[0] as i32;
        let pg = p[1] as i32;
        let pb = p[2] as i32;

        let mut best_dist = i32::MAX;
        let mut best_idx = 0u32;

        for (idx, pal) in palette.iter().enumerate() {
            let dist = color_dist_sq(pr, pg, pb, pal.0, pal.1, pal.2);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx as u32;
            }
        }

        indices_u32 |= (best_idx & 0x03) << (i * 2);
    }

    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&c0_565.to_le_bytes());
    out[2..4].copy_from_slice(&c1_565.to_le_bytes());
    out[4..8].copy_from_slice(&indices_u32.to_le_bytes());
    out
}

/// Encodes an uncompressed RGBA8 2D image into 256-byte aligned BC1 rows.
pub fn encode_rgba_to_bc1_dma(
    width: u32,
    height: u32,
    rgba_pixels: &[u8],
) -> (alloc::vec::Vec<u8>, u32, u16) {
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);

    let logical_pitch = blocks_x * 8; // 8 bytes per 4x4 block
    let aligned_pitch = (logical_pitch + 255) & !255; // 256-byte row pitch requirement
    let total_bytes = (aligned_pitch * blocks_y) as usize;

    let mut out_buffer = alloc::vec![0u8; total_bytes];

    for by in 0..blocks_y {
        let row_start = (by * aligned_pitch) as usize;
        for bx in 0..blocks_x {
            let mut block = [[0u8; 4]; 16];
            for py in 0..4 {
                let y = (by * 4 + py).min(height - 1);
                for px in 0..4 {
                    let x = (bx * 4 + px).min(width - 1);
                    let pixel_idx = ((y * width + x) * 4) as usize;
                    if pixel_idx + 4 <= rgba_pixels.len() {
                        block[(py * 4 + px) as usize] = [
                            rgba_pixels[pixel_idx],
                            rgba_pixels[pixel_idx + 1],
                            rgba_pixels[pixel_idx + 2],
                            rgba_pixels[pixel_idx + 3],
                        ];
                    }
                }
            }
            let encoded_block = encode_bc1_block(&block);
            let block_offset = row_start + (bx * 8) as usize;
            out_buffer[block_offset..block_offset + 8].copy_from_slice(&encoded_block);
        }
    }

    (out_buffer, aligned_pitch, blocks_y as u16)
}
