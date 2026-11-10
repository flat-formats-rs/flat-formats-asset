//! # flattex-core
//!
//! Direct-to-GPU DMA 256-byte row-pitch pre-aligned texture container format.
//!
//! Eliminates host-side CPU repacking and re-allocations during texture uploads.
//! All subresources conform to WGPU (`COPY_BYTES_PER_ROW_ALIGNMENT = 256`),
//! DirectX 12 (`D3D12_TEXTURE_DATA_PITCH_ALIGNMENT = 256`), and Vulkan optimal placement.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod error;
pub mod format;
pub mod header;
pub mod mip;
pub mod reader;
pub mod streaming;

pub use error::{FlatTexError, Result};
pub use format::VkFormat;
pub use header::{FlatTexHeader, FormatSignature};
pub use mip::FlatMipDesc;
pub use reader::FlatTexView;
pub use streaming::{stream_copy, CONTAINER_PAYLOAD_OFFSET_4KB, DIRECT_STORAGE_SECTOR_ALIGNMENT};
