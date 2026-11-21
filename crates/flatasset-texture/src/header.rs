//! Container header and packed format signature for FlatTexture.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// 8-byte packed format signature (Magic 48-bit + Major 8-bit + Minor 8-bit).
#[derive(Copy, Clone, Debug, PartialEq, Eq, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(transparent)]
pub struct FormatSignature(pub u64);

impl FormatSignature {
    /// Canonical v1.0 signature for FlatTexture (`b"FLTTEX"` + major 1 + minor 0).
    pub const FLATTEXTURE_V1_0: Self = Self::pack(*b"FLTTEX", 1, 0);

    pub const fn pack(magic: [u8; 6], major: u8, minor: u8) -> Self {
        let m = u64::from_le_bytes([
            magic[0], magic[1], magic[2], magic[3], magic[4], magic[5], 0, 0,
        ]);
        Self(m | ((major as u64) << 48) | ((minor as u64) << 56))
    }

    #[inline(always)]
    pub fn is_valid(&self, expected: Self) -> bool {
        self.0 == expected.0
    }

    #[inline(always)]
    pub fn version(&self) -> (u8, u8) {
        ((self.0 >> 48) as u8, (self.0 >> 56) as u8)
    }
}
const _: () = assert!(core::mem::size_of::<FormatSignature>() == 8);
const _: () = assert!(core::mem::align_of::<FormatSignature>() == 8);

/// 32-byte ultra-compact texture header.
///
/// Fits into half of a 64-byte L1 CPU cache line.
/// Zero padding holes with natural field alignment.
#[derive(Copy, Clone, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C, align(32))]
pub struct FlatTextureHeader {
    /// 8 bytes (offset 0..8): Packed Magic + Major/Minor version.
    pub signature: FormatSignature,
    /// 4 bytes (offset 8..12): Total payload byte length in kilobytes.
    pub total_payload_kb: u32,
    /// 2 bytes (offset 12..14): Standard Vulkan/WGPU VkFormat enum value.
    pub vk_format: u16,
    /// 2 bytes (offset 14..16): Width in texels (up to 65535).
    pub width: u16,
    /// 2 bytes (offset 16..18): Height in texels (up to 65535).
    pub height: u16,
    /// 2 bytes (offset 18..20): Depth (for 3D textures) or array layer count (for cubemaps/arrays).
    pub depth_or_layers: u16,
    /// 2 bytes (offset 20..22): Offset to mip descriptor directory / 32.
    pub mip_table_offset_div32: u16,
    /// 2 bytes (offset 22..24): Offset to Direct DMA payload / 256.
    pub payload_offset_div256: u16,
    /// 1 byte (offset 24..25): Number of mip levels (1..16).
    pub mip_levels: u8,
    /// 1 byte (offset 25..26): Bit 0: sRGB, Bit 1: Normal map, Bit 2: Has palette.
    pub flags: u8,
    /// 6 bytes (offset 26..32): Reserved for future extensions.
    pub _reserved: [u8; 6],
}

const _: () = assert!(core::mem::size_of::<FlatTextureHeader>() == 32);
const _: () = assert!(core::mem::align_of::<FlatTextureHeader>() == 32);

impl FlatTextureHeader {
    pub const FLAG_SRGB: u8 = 1 << 0;
    pub const FLAG_NORMAL_MAP: u8 = 1 << 1;
    pub const FLAG_HAS_PALETTE: u8 = 1 << 2;

    #[inline(always)]
    pub fn is_srgb(&self) -> bool {
        (self.flags & Self::FLAG_SRGB) != 0
    }

    #[inline(always)]
    pub fn is_normal_map(&self) -> bool {
        (self.flags & Self::FLAG_NORMAL_MAP) != 0
    }

    #[inline(always)]
    pub fn has_palette(&self) -> bool {
        (self.flags & Self::FLAG_HAS_PALETTE) != 0
    }

    #[inline(always)]
    pub fn mip_table_offset(&self) -> usize {
        (self.mip_table_offset_div32 as usize) * 32
    }

    #[inline(always)]
    pub fn payload_offset(&self) -> usize {
        (self.payload_offset_div256 as usize) * 256
    }
}
