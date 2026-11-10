//! FlatWorld container builder and serializer.

use std::vec::Vec;
use zerocopy::IntoBytes;

use crate::header::{
    CompressionType, FlatWorldHeader, FlatWorldLumpDesc, LumpType, CONTAINER_PAYLOAD_OFFSET,
    FLATWORLD_MAGIC, FLATWORLD_VERSION,
};

/// Builder for constructing valid FlatWorld (.flw) containers.
#[derive(Debug, Default)]
pub struct FlatWorldBuilder {
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    flags: u32,
    lumps: Vec<(FlatWorldLumpDesc, Vec<u8>)>,
}

impl FlatWorldBuilder {
    /// Creates a new builder with world bounds.
    pub fn new(bounds_min: [f32; 3], bounds_max: [f32; 3]) -> Self {
        Self {
            bounds_min,
            bounds_max,
            flags: 0,
            lumps: Vec::new(),
        }
    }

    /// Sets container flags.
    pub fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }

    /// Adds a typed lump with raw uncompressed payload.
    pub fn add_lump(
        self,
        lump_type: LumpType,
        element_count: u64,
        element_stride: u32,
        payload: Vec<u8>,
    ) -> Self {
        self.add_lump_with_compression(
            lump_type,
            CompressionType::Uncompressed,
            element_count,
            element_stride,
            payload,
        )
    }

    /// Adds a typed lump with optional in-container compression.
    pub fn add_lump_with_compression(
        mut self,
        lump_type: LumpType,
        compression_type: CompressionType,
        element_count: u64,
        element_stride: u32,
        uncompressed_payload: Vec<u8>,
    ) -> Self {
        let uncompressed_size = uncompressed_payload.len() as u64;

        let (actual_compression, payload) = match compression_type {
            #[cfg(feature = "zstd")]
            CompressionType::Zstd => {
                let compressed = zstd::encode_all(&uncompressed_payload[..], 3)
                    .unwrap_or_else(|_| uncompressed_payload.clone());
                (CompressionType::Zstd, compressed)
            }
            _ => (CompressionType::Uncompressed, uncompressed_payload),
        };

        let checksum = {
            #[cfg(feature = "blake3")]
            {
                let hash = blake3::hash(&payload);
                let mut truncated = [0u8; 16];
                truncated.copy_from_slice(&hash.as_bytes()[..16]);
                truncated
            }
            #[cfg(not(feature = "blake3"))]
            {
                [0u8; 16]
            }
        };

        let desc = FlatWorldLumpDesc {
            lump_type: lump_type as u32,
            compression_type: actual_compression as u32,
            file_offset: 0, // Computed during build
            compressed_size: payload.len() as u64,
            uncompressed_size,
            element_count,
            element_stride,
            _pad_align: [0; 4],
            checksum_128: checksum,
        };

        self.lumps.push((desc, payload));
        self
    }

    /// Compiles and serializes the complete container to bytes.
    pub fn build(mut self) -> Vec<u8> {
        let lump_count = self.lumps.len() as u32;
        let header_and_table_size =
            core::mem::size_of::<FlatWorldHeader>() + (lump_count as usize) * 64;

        // Payloads begin at 4KB offset or aligned boundary
        let mut current_offset = (header_and_table_size as u64).max(CONTAINER_PAYLOAD_OFFSET);
        // Align to 4096
        if current_offset % CONTAINER_PAYLOAD_OFFSET != 0 {
            current_offset +=
                CONTAINER_PAYLOAD_OFFSET - (current_offset % CONTAINER_PAYLOAD_OFFSET);
        }

        // Assign file offsets and 4KB align each subsequent lump
        for (desc, payload) in &mut self.lumps {
            desc.file_offset = current_offset;
            let len = payload.len() as u64;
            current_offset += len;
            // Pad to 64 bytes for cache line alignment
            if current_offset % 64 != 0 {
                current_offset += 64 - (current_offset % 64);
            }
        }

        let content_hash_128 = {
            #[cfg(feature = "blake3")]
            {
                let mut hasher = blake3::Hasher::new();
                for (_, payload) in &self.lumps {
                    hasher.update(payload);
                }
                let mut truncated = [0u8; 16];
                truncated.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
                truncated
            }
            #[cfg(not(feature = "blake3"))]
            {
                [0u8; 16]
            }
        };

        let header = FlatWorldHeader {
            magic: FLATWORLD_MAGIC,
            version: FLATWORLD_VERSION,
            flags: self.flags,
            lump_count,
            bounds_min: self.bounds_min,
            bounds_max: self.bounds_max,
            content_hash_128,
            _reserved: [0; 8],
        };

        let mut out = Vec::with_capacity(current_offset as usize);
        out.extend_from_slice(header.as_bytes());

        for (desc, _) in &self.lumps {
            out.extend_from_slice(desc.as_bytes());
        }

        for (desc, payload) in self.lumps {
            let padding_needed = (desc.file_offset as usize).saturating_sub(out.len());
            out.resize(out.len() + padding_needed, 0);
            out.extend_from_slice(&payload);
            let align_pad = out.len() % 64;
            if align_pad != 0 {
                out.resize(out.len() + (64 - align_pad), 0);
            }
        }

        out
    }
}
