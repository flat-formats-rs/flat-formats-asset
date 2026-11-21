//! FlatLightmaps: High-performance, GPU-ready Lightmap Atlas for FlatWorld.
//!
//! Provides zero-copy representations, 256-byte Vulkan/WGPU Direct-DMA row pitch alignment,
//! 32-byte cache-line friendly face descriptors, and deterministic shelf packing with
//! 1-pixel gutter replication to eliminate bilinear sampling bleeding.

use core::mem::size_of;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// File format signature for Lightmap lump payload: b"LMAP"
pub const LIGHTMAP_MAGIC: [u8; 4] = *b"LMAP";
/// Supported format version.
pub const LIGHTMAP_VERSION: u32 = 1;
/// Default atlas width matching modern GPU raster requirements.
pub const DEFAULT_ATLAS_WIDTH: u32 = 2048;
/// Standard Vulkan/WGPU buffer row alignment for Direct-DMA uploads (256 bytes).
pub const GPU_ROW_PITCH_ALIGNMENT: u32 = 256;

/// Canonical GoldSrc standard display gamma constant (`v_lightgamma 2.5`).
pub const DEFAULT_LIGHTGAMMA: f32 = 2.5;
/// Canonical GoldSrc lightscale factor.
pub const DEFAULT_LIGHTSCALE: f32 = 2.0;

/// Pixel format of the lightmap atlas buffer.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightmapFormat {
    /// 32-bit RGBA (8 bits per channel), sRGB or linear UNORM.
    Rgba8Unorm = 0,
    /// Block-compressed 4bpp BC1/DXT1 format.
    Bc1Unorm = 1,
    /// Single-channel 8-bit luxels (monochrome lighting).
    R8Unorm = 2,
    /// 64-bit RGBA (16-bit half-float per channel) for HDR lighting.
    Rgba16Float = 3,
}

impl From<u32> for LightmapFormat {
    #[inline]
    fn from(val: u32) -> Self {
        match val {
            1 => Self::Bc1Unorm,
            2 => Self::R8Unorm,
            3 => Self::Rgba16Float,
            _ => Self::Rgba8Unorm,
        }
    }
}

/// 64-byte cache-line sized Lightmap Atlas Header with zero padding holes.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, PartialEq)]
pub struct LightmapAtlasHeader {
    /// File magic: b"LMAP" (offset 0..4)
    pub magic: [u8; 4],
    /// Atlas format version (offset 4..8)
    pub version: u32,
    /// Pixel format (`LightmapFormat`) (offset 8..12)
    pub format: u32,
    /// Feature and compression flags (offset 12..16)
    pub flags: u32,
    /// Atlas width in luxels (offset 16..20)
    pub width: u32,
    /// Atlas height in luxels (offset 20..24)
    pub height: u32,
    /// Total number of mapped faces (offset 24..28)
    pub face_count: u32,
    /// Byte pitch per row in memory, aligned to 256 bytes for Direct-DMA (offset 28..32)
    pub row_pitch_bytes: u32,
    /// Absolute byte offset from lump start to `[FaceLightmapDesc]` array (offset 32..40)
    pub descriptors_offset: u64,
    /// Absolute byte offset from lump start to raw atlas texel buffer (offset 40..48)
    pub data_offset: u64,
    /// Total byte size of the raw texel buffer (offset 48..56)
    pub data_size: u64,
    /// Reserved space for future extensions (offset 56..64)
    pub _reserved: [u8; 8],
}

const _: () = assert!(size_of::<LightmapAtlasHeader>() == 64);

/// 32-byte Face Lightmap Descriptor (fits exactly 2 entries per 64-byte cache line).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, PartialEq)]
pub struct FaceLightmapDesc {
    /// Surface projection minimum on S axis in world space (offset 0..4)
    pub s_min: f32,
    /// Surface projection minimum on T axis in world space (offset 4..8)
    pub t_min: f32,
    /// Horizontal offset in luxels within atlas (offset 8..10)
    pub atlas_x: u16,
    /// Vertical offset in luxels within atlas (offset 10..12)
    pub atlas_y: u16,
    /// Face lightmap width in luxels (offset 12..14)
    pub width: u16,
    /// Face lightmap height in luxels (offset 14..16)
    pub height: u16,
    /// Normalized UV minimum [U, V] within [0..1] range for pixel shaders (offset 16..24)
    pub uv_min: [f32; 2],
    /// Normalized UV maximum [U, V] within [0..1] range for pixel shaders (offset 24..32)
    pub uv_max: [f32; 2],
}

const _: () = assert!(size_of::<FaceLightmapDesc>() == 32);

/// Error types occurring during lightmap parsing and validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightmapError {
    /// Buffer is smaller than the 64-byte header.
    BufferTooSmall,
    /// Header magic does not match `b"LMAP"`.
    InvalidMagic,
    /// Format version is unsupported.
    UnsupportedVersion,
    /// Face descriptors lump bounds are out of range or misaligned.
    InvalidDescriptorsBounds,
    /// Texel data buffer bounds are out of range.
    InvalidDataBounds,
    /// Pitch does not satisfy GPU alignment constraints.
    MisalignedPitch,
}

impl core::fmt::Display for LightmapError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BufferTooSmall => write!(f, "lightmap buffer too small for header"),
            Self::InvalidMagic => write!(f, "invalid lightmap magic signature, expected LMAP"),
            Self::UnsupportedVersion => write!(f, "unsupported lightmap version"),
            Self::InvalidDescriptorsBounds => write!(f, "face descriptors table out of bounds"),
            Self::InvalidDataBounds => write!(f, "texel data buffer out of bounds"),
            Self::MisalignedPitch => write!(f, "texel row pitch violates 256-byte alignment"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LightmapError {}

/// Zero-copy view into a FlatLightmap atlas lump.
#[derive(Debug, Clone, Copy)]
pub struct LightmapAtlasView<'a> {
    header: &'a LightmapAtlasHeader,
    face_descs: &'a [FaceLightmapDesc],
    atlas_bytes: &'a [u8],
}

impl<'a> LightmapAtlasView<'a> {
    /// Validates and parses raw lump slice into a zero-copy `LightmapAtlasView`.
    pub fn new(slice: &'a [u8]) -> Result<Self, LightmapError> {
        if slice.len() < size_of::<LightmapAtlasHeader>() {
            return Err(LightmapError::BufferTooSmall);
        }

        let (header_ref, _) = LightmapAtlasHeader::ref_from_prefix(slice)
            .map_err(|_| LightmapError::BufferTooSmall)?;

        if header_ref.magic != LIGHTMAP_MAGIC {
            return Err(LightmapError::InvalidMagic);
        }
        if header_ref.version != LIGHTMAP_VERSION {
            return Err(LightmapError::UnsupportedVersion);
        }
        if header_ref.row_pitch_bytes % GPU_ROW_PITCH_ALIGNMENT != 0 {
            return Err(LightmapError::MisalignedPitch);
        }

        let desc_start = header_ref.descriptors_offset as usize;
        let desc_count = header_ref.face_count as usize;
        let desc_len = desc_count
            .checked_mul(size_of::<FaceLightmapDesc>())
            .ok_or(LightmapError::InvalidDescriptorsBounds)?;
        let desc_end = desc_start
            .checked_add(desc_len)
            .ok_or(LightmapError::InvalidDescriptorsBounds)?;

        if desc_end > slice.len() {
            return Err(LightmapError::InvalidDescriptorsBounds);
        }

        let desc_slice = &slice[desc_start..desc_end];
        let face_descs = <[FaceLightmapDesc]>::ref_from_bytes(desc_slice)
            .map_err(|_| LightmapError::InvalidDescriptorsBounds)?;

        let data_start = header_ref.data_offset as usize;
        let data_len = header_ref.data_size as usize;
        let data_end = data_start
            .checked_add(data_len)
            .ok_or(LightmapError::InvalidDataBounds)?;

        if data_end > slice.len() {
            return Err(LightmapError::InvalidDataBounds);
        }

        let atlas_bytes = &slice[data_start..data_end];

        Ok(Self {
            header: header_ref,
            face_descs,
            atlas_bytes,
        })
    }

    /// Access the 64-byte atlas header.
    #[inline]
    pub fn header(&self) -> &LightmapAtlasHeader {
        self.header
    }

    /// Access the table of face lightmap descriptors.
    #[inline]
    pub fn face_descs(&self) -> &[FaceLightmapDesc] {
        self.face_descs
    }

    /// Retrieve a specific face descriptor by index.
    #[inline]
    pub fn face_desc(&self, face_idx: usize) -> Option<&FaceLightmapDesc> {
        self.face_descs.get(face_idx)
    }

    /// Access the raw Direct-DMA aligned texel buffer ready for GPU upload.
    #[inline]
    pub fn atlas_bytes(&self) -> &'a [u8] {
        self.atlas_bytes
    }

    /// Compute exact normalized atlas UV coordinates for a 3D world vertex on a given face.
    ///
    /// `s_vec` and `t_vec` are the [X, Y, Z, Offset] projection vectors from the face's TexInfo.
    #[inline]
    pub fn compute_luxel_uv(
        &self,
        face_idx: usize,
        world_pos: [f32; 3],
        s_vec: [f32; 4],
        t_vec: [f32; 4],
    ) -> Option<[f32; 2]> {
        let desc = self.face_descs.get(face_idx)?;

        let s =
            world_pos[0] * s_vec[0] + world_pos[1] * s_vec[1] + world_pos[2] * s_vec[2] + s_vec[3];
        let t =
            world_pos[0] * t_vec[0] + world_pos[1] * t_vec[1] + world_pos[2] * t_vec[2] + t_vec[3];

        let atlas_w = self.header.width as f32;
        let atlas_h = self.header.height as f32;

        let u = (f32::from(desc.atlas_x) + ((s - desc.s_min) / 16.0) + 0.5) / atlas_w;
        let v = (f32::from(desc.atlas_y) + ((t - desc.t_min) / 16.0) + 0.5) / atlas_h;

        Some([u, v])
    }
}

/// Metadata and raw luxels for an input face before atlas packing.
#[derive(Debug, Clone)]
pub struct FaceLightmapInput<'a> {
    /// Original face index.
    pub face_idx: usize,
    /// Projection minimum S.
    pub s_min: f32,
    /// Projection minimum T.
    pub t_min: f32,
    /// Lightmap width in luxels.
    pub width: u16,
    /// Lightmap height in luxels.
    pub height: u16,
    /// Raw RGB24 luxel slice (size = width * height * 3).
    pub rgb_luxels: &'a [u8],
}

/// Precomputed gamma curve matching original GoldSrc CRT color reproduction.
pub struct GammaTable {
    table: [u8; 1024],
}

impl GammaTable {
    /// Builds a gamma decoding table for a given gamma power (`2.5` default).
    #[must_use]
    pub fn new(gamma: f32) -> Self {
        let mut table = [0u8; 1024];
        let inv_gamma = 1.0 / gamma;
        for (i, entry) in table.iter_mut().enumerate() {
            let f = (i as f32) / 1023.0;
            let corrected = f.powf(inv_gamma);
            let val = (corrected * 255.0).round().clamp(0.0, 255.0) as u8;
            *entry = val;
        }
        Self { table }
    }

    /// Decodes a raw 8-bit luxel byte using lightscale and gamma table.
    #[inline]
    pub fn decode(&self, raw: u8, lightscale: f32) -> u8 {
        let scaled = ((f32::from(raw) * lightscale) as usize).min(1023);
        self.table[scaled]
    }
}

/// High-performance 2D Decreasing-Height Shelf Packer for lightmap blocks with gutter replication.
pub struct LightmapPacker {
    atlas_width: u32,
    gamma: GammaTable,
    lightscale: f32,
    gutter: u16,
}

impl Default for LightmapPacker {
    fn default() -> Self {
        Self::new(DEFAULT_ATLAS_WIDTH, DEFAULT_LIGHTGAMMA, DEFAULT_LIGHTSCALE)
    }
}

impl LightmapPacker {
    /// Creates a packer with target atlas width and gamma parameters.
    pub fn new(atlas_width: u32, gamma: f32, lightscale: f32) -> Self {
        Self {
            atlas_width,
            gamma: GammaTable::new(gamma),
            lightscale,
            gutter: 1, // 1-pixel border replication to eliminate bilinear seams
        }
    }

    /// Packs an array of face lightmaps into a Direct-DMA aligned RGBA8 Lightmap Atlas container.
    ///
    /// Produces a contiguous byte buffer containing:
    /// 1. `LightmapAtlasHeader` (64 bytes).
    /// 2. Padded alignment to 64 bytes.
    /// 3. `[FaceLightmapDesc]` table (32 bytes per face).
    /// 4. Padded alignment to 256 bytes.
    /// 5. Raw Direct-DMA texel buffer with rows aligned to 256 bytes.
    pub fn pack<'a>(
        &self,
        faces: &[FaceLightmapInput<'a>],
    ) -> (LightmapAtlasHeader, Vec<FaceLightmapDesc>, Vec<u8>) {
        if faces.is_empty() {
            let header = LightmapAtlasHeader {
                magic: LIGHTMAP_MAGIC,
                version: LIGHTMAP_VERSION,
                format: LightmapFormat::Rgba8Unorm as u32,
                flags: 0,
                width: self.atlas_width,
                height: 0,
                face_count: 0,
                row_pitch_bytes: self.atlas_width * 4,
                descriptors_offset: 64,
                data_offset: 64,
                data_size: 0,
                _reserved: [0; 8],
            };
            return (header, Vec::new(), Vec::new());
        }

        // Shelf packing allocation with gutter padding:
        // Each block takes (w + gutter * 2) by (h + gutter * 2)
        let gutter = self.gutter as u32;
        let mut placements: Vec<(usize, u16, u16)> = Vec::with_capacity(faces.len());
        let mut cur_x = 0u32;
        let mut cur_y = 0u32;
        let mut shelf_h = 0u32;

        for (i, face) in faces.iter().enumerate() {
            let alloc_w = u32::from(face.width) + gutter * 2;
            let alloc_h = u32::from(face.height) + gutter * 2;

            if cur_x + alloc_w > self.atlas_width {
                // Advance to next shelf
                cur_x = 0;
                cur_y += shelf_h;
                shelf_h = 0;
            }

            // Luxel content is placed at (cur_x + gutter, cur_y + gutter)
            let luxel_x = (cur_x + gutter) as u16;
            let luxel_y = (cur_y + gutter) as u16;
            placements.push((i, luxel_x, luxel_y));

            shelf_h = shelf_h.max(alloc_h);
            cur_x += alloc_w;
        }

        let total_h = (cur_y + shelf_h).max(1).next_power_of_two();
        let row_pitch =
            (self.atlas_width * 4).div_ceil(GPU_ROW_PITCH_ALIGNMENT) * GPU_ROW_PITCH_ALIGNMENT;

        let total_texels_size = (row_pitch as usize) * (total_h as usize);
        let mut atlas_bytes = vec![0u8; total_texels_size];

        let mut face_descs: Vec<FaceLightmapDesc> = Vec::with_capacity(faces.len());
        let atlas_w_f32 = self.atlas_width as f32;
        let atlas_h_f32 = total_h as f32;

        for &(face_idx, luxel_x, luxel_y) in &placements {
            let face = &faces[face_idx];
            let w = face.width as usize;
            let h = face.height as usize;

            let uv_min = [
                f32::from(luxel_x) / atlas_w_f32,
                f32::from(luxel_y) / atlas_h_f32,
            ];
            let uv_max = [
                f32::from(luxel_x + face.width) / atlas_w_f32,
                f32::from(luxel_y + face.height) / atlas_h_f32,
            ];

            face_descs.push(FaceLightmapDesc {
                s_min: face.s_min,
                t_min: face.t_min,
                atlas_x: luxel_x,
                atlas_y: luxel_y,
                width: face.width,
                height: face.height,
                uv_min,
                uv_max,
            });

            // Decode RGB24 -> RGBA8 into the atlas with gamma & border gutter replication
            for ly in 0..h {
                for lx in 0..w {
                    let src_idx = (ly * w + lx) * 3;
                    if src_idx + 2 < face.rgb_luxels.len() {
                        let r = self.gamma.decode(face.rgb_luxels[src_idx], self.lightscale);
                        let g = self
                            .gamma
                            .decode(face.rgb_luxels[src_idx + 1], self.lightscale);
                        let b = self
                            .gamma
                            .decode(face.rgb_luxels[src_idx + 2], self.lightscale);

                        let px = (luxel_x as usize) + lx;
                        let py = (luxel_y as usize) + ly;
                        let dst_idx = py * (row_pitch as usize) + px * 4;

                        atlas_bytes[dst_idx] = r;
                        atlas_bytes[dst_idx + 1] = g;
                        atlas_bytes[dst_idx + 2] = b;
                        atlas_bytes[dst_idx + 3] = 255;

                        // Border replication (1-pixel gutter clamping):
                        // Left gutter
                        if lx == 0 {
                            let g_dst = py * (row_pitch as usize) + (px - 1) * 4;
                            atlas_bytes[g_dst..g_dst + 4].copy_from_slice(&[r, g, b, 255]);
                        }
                        // Right gutter
                        if lx == w - 1 {
                            let g_dst = py * (row_pitch as usize) + (px + 1) * 4;
                            atlas_bytes[g_dst..g_dst + 4].copy_from_slice(&[r, g, b, 255]);
                        }
                        // Top gutter
                        if ly == 0 {
                            let g_dst = (py - 1) * (row_pitch as usize) + px * 4;
                            atlas_bytes[g_dst..g_dst + 4].copy_from_slice(&[r, g, b, 255]);
                        }
                        // Bottom gutter
                        if ly == h - 1 {
                            let g_dst = (py + 1) * (row_pitch as usize) + px * 4;
                            atlas_bytes[g_dst..g_dst + 4].copy_from_slice(&[r, g, b, 255]);
                        }
                    }
                }
            }

            // Gutter corner replication:
            let corners = [
                (luxel_x as usize - 1, luxel_y as usize - 1, 0, 0),
                (luxel_x as usize + w, luxel_y as usize - 1, w - 1, 0),
                (luxel_x as usize - 1, luxel_y as usize + h, 0, h - 1),
                (luxel_x as usize + w, luxel_y as usize + h, w - 1, h - 1),
            ];
            for (cx, cy, lx, ly) in corners {
                let src_idx = (ly * w + lx) * 3;
                if src_idx + 2 < face.rgb_luxels.len() {
                    let r = self.gamma.decode(face.rgb_luxels[src_idx], self.lightscale);
                    let g = self
                        .gamma
                        .decode(face.rgb_luxels[src_idx + 1], self.lightscale);
                    let b = self
                        .gamma
                        .decode(face.rgb_luxels[src_idx + 2], self.lightscale);
                    let dst_idx = cy * (row_pitch as usize) + cx * 4;
                    if dst_idx + 3 < atlas_bytes.len() {
                        atlas_bytes[dst_idx..dst_idx + 4].copy_from_slice(&[r, g, b, 255]);
                    }
                }
            }
        }

        // Layout container offsets:
        let desc_offset = 64u64;
        let desc_bytes_len = (face_descs.len() * size_of::<FaceLightmapDesc>()) as u64;
        // Align data offset to 256 bytes for Direct-DMA GPU staging
        let data_offset = (desc_offset + desc_bytes_len).div_ceil(GPU_ROW_PITCH_ALIGNMENT as u64)
            * (GPU_ROW_PITCH_ALIGNMENT as u64);

        let header = LightmapAtlasHeader {
            magic: LIGHTMAP_MAGIC,
            version: LIGHTMAP_VERSION,
            format: LightmapFormat::Rgba8Unorm as u32,
            flags: 0,
            width: self.atlas_width,
            height: total_h,
            face_count: face_descs.len() as u32,
            row_pitch_bytes: row_pitch,
            descriptors_offset: desc_offset,
            data_offset,
            data_size: atlas_bytes.len() as u64,
            _reserved: [0; 8],
        };

        (header, face_descs, atlas_bytes)
    }

    /// Serializes packed atlas header, descriptors, and pixel data into a single lump payload.
    pub fn build_lump_payload(&self, faces: &[FaceLightmapInput<'_>]) -> Vec<u8> {
        let (header, descs, pixels) = self.pack(faces);

        let total_size = header.data_offset as usize + pixels.len();
        let mut buffer = vec![0u8; total_size];

        // 1. Copy Header (0..64)
        buffer[..size_of::<LightmapAtlasHeader>()].copy_from_slice(header.as_bytes());

        // 2. Copy Descriptors
        let desc_start = header.descriptors_offset as usize;
        let desc_bytes = descs.as_bytes();
        buffer[desc_start..desc_start + desc_bytes.len()].copy_from_slice(desc_bytes);

        // 3. Copy Direct-DMA Texel Buffer
        let data_start = header.data_offset as usize;
        buffer[data_start..data_start + pixels.len()].copy_from_slice(&pixels);

        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lightmap_layouts_and_sizes() {
        assert_eq!(size_of::<LightmapAtlasHeader>(), 64);
        assert_eq!(size_of::<FaceLightmapDesc>(), 32);
    }

    #[test]
    fn test_packer_empty() {
        let packer = LightmapPacker::default();
        let payload = packer.build_lump_payload(&[]);
        assert!(payload.len() >= 64);

        let view = LightmapAtlasView::new(&payload).expect("empty view valid");
        assert_eq!(view.header().face_count, 0);
        assert_eq!(view.face_descs().len(), 0);
    }

    #[test]
    fn test_packer_and_zero_copy_roundtrip() {
        let packer = LightmapPacker::new(512, DEFAULT_LIGHTGAMMA, DEFAULT_LIGHTSCALE);

        let dummy_luxels_1 = vec![200u8; 16 * 16 * 3]; // 16x16 face
        let dummy_luxels_2 = vec![100u8; 32 * 16 * 3]; // 32x16 face

        let faces = [
            FaceLightmapInput {
                face_idx: 0,
                s_min: 0.0,
                t_min: 0.0,
                width: 16,
                height: 16,
                rgb_luxels: &dummy_luxels_1,
            },
            FaceLightmapInput {
                face_idx: 1,
                s_min: 128.0,
                t_min: 64.0,
                width: 32,
                height: 16,
                rgb_luxels: &dummy_luxels_2,
            },
        ];

        let payload = packer.build_lump_payload(&faces);
        let view = LightmapAtlasView::new(&payload).expect("parse payload");

        assert_eq!(view.header().magic, LIGHTMAP_MAGIC);
        assert_eq!(view.header().width, 512);
        assert_eq!(view.header().face_count, 2);
        assert_eq!(view.face_descs().len(), 2);

        // Check Direct-DMA 256-byte pitch alignment:
        assert_eq!(view.header().row_pitch_bytes % 256, 0);
        assert_eq!(view.header().data_offset % 256, 0);

        // Verify face descriptors:
        let desc0 = view.face_desc(0).unwrap();
        assert_eq!(desc0.width, 16);
        assert_eq!(desc0.height, 16);
        assert!(desc0.atlas_x >= 1); // 1-pixel gutter
        assert!(desc0.atlas_y >= 1);

        let desc1 = view.face_desc(1).unwrap();
        assert_eq!(desc1.width, 32);
        assert_eq!(desc1.height, 16);

        // Verify UV computation:
        let uv = view
            .compute_luxel_uv(
                0,
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            )
            .unwrap();
        assert!(uv[0] > 0.0 && uv[0] < 1.0);
        assert!(uv[1] > 0.0 && uv[1] < 1.0);
    }
}
