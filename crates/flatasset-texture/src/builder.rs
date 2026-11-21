//! Direct-DMA FlatTexture Container Builder.
//!
//! Synthesizes zero-copy `.flt` texture files from raw or block-compressed image streams
//! with strict 256-byte row-pitch alignment and 512-byte subresource alignment.

use crate::error::{FlatTextureError, Result};
use crate::format::VkFormat;
use crate::header::{FlatTextureHeader, FormatSignature};
use crate::mip::FlatMipDesc;
use crate::streaming::CONTAINER_PAYLOAD_OFFSET_4KB;
use alloc::vec::Vec;
use zerocopy::IntoBytes;

/// Subresource input buffer for builder.
pub struct MipInput {
    pub width: u32,
    pub height: u32,
    /// Pre-aligned direct DMA bytes conforming to 256-byte row pitch.
    pub dma_bytes: Vec<u8>,
    pub aligned_row_pitch: u32,
    pub rows_per_image: u16,
}

/// Builder for constructing valid Direct-DMA FlatTexture binary packages.
pub struct FlatTextureBuilder {
    width: u16,
    height: u16,
    depth_or_layers: u16,
    vk_format: VkFormat,
    flags: u8,
    mips: Vec<MipInput>,
}

impl FlatTextureBuilder {
    /// Creates a new builder for a 2D texture.
    pub fn new_2d(width: u16, height: u16, vk_format: VkFormat) -> Self {
        Self {
            width,
            height,
            depth_or_layers: 1,
            vk_format,
            flags: 0,
            mips: Vec::new(),
        }
    }

    /// Creates a new builder for a 6-face Cubemap texture (depth_or_layers = 6).
    pub fn new_cubemap(width: u16, height: u16, vk_format: VkFormat) -> Self {
        Self {
            width,
            height,
            depth_or_layers: 6,
            vk_format,
            flags: 0,
            mips: Vec::new(),
        }
    }

    /// Sets the sRGB color space flag.
    pub fn with_srgb(mut self, srgb: bool) -> Self {
        if srgb {
            self.flags |= FlatTextureHeader::FLAG_SRGB;
        } else {
            self.flags &= !FlatTextureHeader::FLAG_SRGB;
        }
        self
    }

    /// Adds a pre-aligned mip subresource.
    pub fn add_mip(&mut self, mip: MipInput) {
        self.mips.push(mip);
    }

    /// Serializes the complete `.flt` container.
    pub fn build(self) -> Result<Vec<u8>> {
        if self.mips.is_empty() {
            return Err(FlatTextureError::InvalidLayout);
        }

        let mip_levels = self.mips.len() as u8;
        let mip_table_offset = 32usize; // right after 32B header
        let mip_table_size = (mip_levels as usize) * core::mem::size_of::<FlatMipDesc>();

        if mip_table_offset + mip_table_size > CONTAINER_PAYLOAD_OFFSET_4KB {
            return Err(FlatTextureError::InvalidLayout);
        }

        let mut payload_data = Vec::new();
        let mut mip_descs = Vec::with_capacity(mip_levels as usize);

        for mip in self.mips {
            // Align start of each subresource to 512 bytes
            let current_pos = CONTAINER_PAYLOAD_OFFSET_4KB + payload_data.len();
            let aligned_pos = (current_pos + 511) & !511;
            let pad_len = aligned_pos - current_pos;
            payload_data.resize(payload_data.len() + pad_len, 0);

            let subresource_offset = (CONTAINER_PAYLOAD_OFFSET_4KB + payload_data.len()) as u64;
            let byte_size = mip.dma_bytes.len() as u32;

            mip_descs.push(FlatMipDesc {
                offset: subresource_offset,
                byte_size,
                aligned_row_pitch_div256: (mip.aligned_row_pitch / 256) as u16,
                rows_per_image: mip.rows_per_image,
            });

            payload_data.extend_from_slice(&mip.dma_bytes);
        }

        // 4KB alignment for whole container payload
        let total_size = CONTAINER_PAYLOAD_OFFSET_4KB + payload_data.len();
        let mut container = alloc::vec![0u8; total_size];

        let header = FlatTextureHeader {
            signature: FormatSignature::FLATTEXTURE_V1_0,
            total_payload_kb: (payload_data.len() / 1024) as u32,
            vk_format: self.vk_format as u16,
            width: self.width,
            height: self.height,
            depth_or_layers: self.depth_or_layers,
            mip_table_offset_div32: (mip_table_offset / 32) as u16,
            payload_offset_div256: (CONTAINER_PAYLOAD_OFFSET_4KB / 256) as u16,
            mip_levels,
            flags: self.flags,
            _reserved: [0; 6],
        };

        // Write header
        container[..32].copy_from_slice(header.as_bytes());

        // Write mip table
        let mut m_offset = mip_table_offset;
        for desc in &mip_descs {
            container[m_offset..m_offset + 16].copy_from_slice(desc.as_bytes());
            m_offset += 16;
        }

        // Write payload data at 4KB offset
        container[CONTAINER_PAYLOAD_OFFSET_4KB..CONTAINER_PAYLOAD_OFFSET_4KB + payload_data.len()]
            .copy_from_slice(&payload_data);

        Ok(container)
    }
}
