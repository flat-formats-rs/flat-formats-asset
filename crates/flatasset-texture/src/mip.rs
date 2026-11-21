//! Subresource Mip Descriptors and 256-Byte Row Pitch Guarantees.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
pub const D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT: u64 = 512;

/// 16-byte Subresource Mip Descriptor.
///
/// Packs exactly 4 subresource descriptors per 64-byte L1 cache line.
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, FromBytes, Immutable, IntoBytes, KnownLayout,
)]
#[repr(C, align(16))]
pub struct FlatMipDesc {
    /// 8 bytes: Byte offset of the mip level from container start (512-byte aligned).
    pub offset: u64,
    /// 4 bytes: Total bytes of this subresource including 256-byte row padding.
    pub byte_size: u32,
    /// 2 bytes: Pre-padded byte stride of each block row divided by 256.
    pub aligned_row_pitch_div256: u16,
    /// 2 bytes: Number of block rows in this subresource.
    pub rows_per_image: u16,
}

const _: () = assert!(core::mem::size_of::<FlatMipDesc>() == 16);
const _: () = assert!(core::mem::align_of::<FlatMipDesc>() == 16);

impl FlatMipDesc {
    #[inline(always)]
    pub fn aligned_row_pitch(&self) -> u32 {
        (self.aligned_row_pitch_div256 as u32) * COPY_BYTES_PER_ROW_ALIGNMENT
    }
}
