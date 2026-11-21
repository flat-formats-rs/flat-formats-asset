//! Standard Vulkan/WGPU GPU texture format identifiers.

/// 16-bit packed representation of standard GPU block-compressed and uncompressed formats.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum VkFormat {
    R8G8B8A8Unorm = 37,
    R8G8B8A8Srgb = 43,
    Bc1RgbUnormBlock = 131,
    Bc1RgbSrgbBlock = 132,
    Bc2UnormBlock = 135,
    Bc2SrgbBlock = 136,
    Bc3UnormBlock = 137,
    Bc3SrgbBlock = 138,
    Bc4UnormBlock = 139,
    Bc5UnormBlock = 141,
    Bc7UnormBlock = 145,
    Bc7SrgbBlock = 146,
    Astc4x4UnormBlock = 157,
    Astc4x4SrgbBlock = 158,
    /// Custom GoldSrc 8-bit paletted texture (requires compute shader resolve).
    Paletted8 = 0xFFFF,
}

impl VkFormat {
    /// Return block dimension in texels `(width, height)`.
    #[inline(always)]
    pub const fn block_dimensions(&self) -> (u32, u32) {
        match self {
            Self::Bc1RgbUnormBlock
            | Self::Bc1RgbSrgbBlock
            | Self::Bc2UnormBlock
            | Self::Bc2SrgbBlock
            | Self::Bc3UnormBlock
            | Self::Bc3SrgbBlock
            | Self::Bc4UnormBlock
            | Self::Bc5UnormBlock
            | Self::Bc7UnormBlock
            | Self::Bc7SrgbBlock
            | Self::Astc4x4UnormBlock
            | Self::Astc4x4SrgbBlock => (4, 4),
            Self::R8G8B8A8Unorm | Self::R8G8B8A8Srgb | Self::Paletted8 => (1, 1),
        }
    }

    /// Return bytes per block.
    #[inline(always)]
    pub const fn bytes_per_block(&self) -> u32 {
        match self {
            Self::Bc1RgbUnormBlock | Self::Bc1RgbSrgbBlock | Self::Bc4UnormBlock => 8,
            Self::Bc2UnormBlock
            | Self::Bc2SrgbBlock
            | Self::Bc3UnormBlock
            | Self::Bc3SrgbBlock
            | Self::Bc5UnormBlock
            | Self::Bc7UnormBlock
            | Self::Bc7SrgbBlock
            | Self::Astc4x4UnormBlock
            | Self::Astc4x4SrgbBlock => 16,
            Self::R8G8B8A8Unorm | Self::R8G8B8A8Srgb => 4,
            Self::Paletted8 => 1,
        }
    }
}
