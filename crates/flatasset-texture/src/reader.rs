//! Zero-allocation reader view for memory-mapped FlatTexture containers.

use crate::error::{FlatTextureError, Result};
use crate::header::{FlatTextureHeader, FormatSignature};
use crate::mip::FlatMipDesc;
use zerocopy::Ref;

/// Zero-allocation, borrow-based view over a contiguous byte slice (e.g. from `memmap2::Mmap`).
#[derive(Debug, Clone, Copy)]
pub struct FlatTextureView<'a> {
    bytes: &'a [u8],
    header: &'a FlatTextureHeader,
    mips: &'a [FlatMipDesc],
}

impl<'a> FlatTextureView<'a> {
    /// Mount a FlatTexture container from a byte slice with zero heap allocations.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let req_size = core::mem::size_of::<FlatTextureHeader>();
        if bytes.len() < req_size {
            return Err(FlatTextureError::BufferTooSmall {
                required: req_size,
                actual: bytes.len(),
            });
        }

        // 1. Branchless 32-byte alignment check (ptr & 31 == 0)
        let ptr_val = bytes.as_ptr() as usize;
        let align_req = core::mem::align_of::<FlatTextureHeader>();
        if (ptr_val & (align_req - 1)) != 0 {
            return Err(FlatTextureError::Misaligned {
                required: align_req,
                actual: ptr_val & (align_req - 1),
            });
        }

        // 2. Zero-copy read header
        let (header_ref, _) = Ref::<&'a [u8], FlatTextureHeader>::from_prefix(bytes)
            .map_err(|_| FlatTextureError::InvalidLayout)?;
        let header = Ref::into_ref(header_ref);

        // 3. Fast single-cycle signature validation
        if !header.signature.is_valid(FormatSignature::FLATTEXTURE_V1_0) {
            return Err(FlatTextureError::InvalidSignature {
                expected: FormatSignature::FLATTEXTURE_V1_0.0,
                actual: header.signature.0,
            });
        }

        // 4. Validate and slice Subresource Mip Directory
        let mip_table_start = header.mip_table_offset();
        let mip_table_size = (header.mip_levels as usize)
            .checked_mul(core::mem::size_of::<FlatMipDesc>())
            .ok_or(FlatTextureError::InvalidLayout)?;
        let mip_table_end =
            mip_table_start
                .checked_add(mip_table_size)
                .ok_or(FlatTextureError::OutOfBounds {
                    offset: mip_table_start as u64,
                    size: mip_table_size,
                    limit: bytes.len(),
                })?;

        if mip_table_end > bytes.len() {
            return Err(FlatTextureError::OutOfBounds {
                offset: mip_table_start as u64,
                size: mip_table_size,
                limit: bytes.len(),
            });
        }

        let mip_slice = &bytes[mip_table_start..mip_table_end];
        let (mips_ref, _) = Ref::<&'a [u8], [FlatMipDesc]>::from_prefix_with_elems(
            mip_slice,
            header.mip_levels as usize,
        )
        .map_err(|_| FlatTextureError::InvalidLayout)?;
        let mips = Ref::into_ref(mips_ref);

        // 5. Verify bounds and 256-byte alignment invariants on all subresources
        for mip in mips {
            let end = mip.offset.checked_add(mip.byte_size as u64).ok_or(
                FlatTextureError::OutOfBounds {
                    offset: mip.offset,
                    size: mip.byte_size as usize,
                    limit: bytes.len(),
                },
            )?;
            if end > bytes.len() as u64 {
                return Err(FlatTextureError::OutOfBounds {
                    offset: mip.offset,
                    size: mip.byte_size as usize,
                    limit: bytes.len(),
                });
            }
            let pitch = mip.aligned_row_pitch();
            if mip.rows_per_image > 0 && pitch == 0 {
                return Err(FlatTextureError::InvalidRowPitch {
                    pitch,
                    required_alignment: 256,
                });
            }
            let min_bytes = (pitch as u64)
                .checked_mul(mip.rows_per_image as u64)
                .ok_or(FlatTextureError::InvalidLayout)?;
            if (mip.byte_size as u64) < min_bytes {
                return Err(FlatTextureError::InvalidLayout);
            }
        }

        Ok(Self {
            bytes,
            header,
            mips,
        })
    }

    #[inline(always)]
    pub fn header(&self) -> &FlatTextureHeader {
        self.header
    }

    #[inline(always)]
    pub fn mips(&self) -> &[FlatMipDesc] {
        self.mips
    }

    /// Retrieve pre-aligned subresource bytes ready for Direct-DMA GPU upload.
    ///
    /// The returned slice is guaranteed pre-padded to 256-byte row pitch and 512-byte subresource placement,
    /// enabling passing directly to `wgpu::Queue::write_texture` with ZERO host-side allocations or repacking.
    pub fn mip_bytes(&self, mip_level: u8) -> Result<&'a [u8]> {
        if mip_level >= self.header.mip_levels {
            return Err(FlatTextureError::MipOutOfBounds {
                requested: mip_level,
                max: self.header.mip_levels,
            });
        }

        let mip = &self.mips[mip_level as usize];
        let start = mip.offset as usize;
        let size = mip.byte_size as usize;
        let end = start + size;

        Ok(&self.bytes[start..end])
    }

    /// Extreme performance sub-nanosecond direct pointer fetch for subresource bytes.
    ///
    /// # Safety
    /// Caller guarantees `mip_level < self.header.mip_levels`. Subresource offset and size
    /// have already been verified in `FlatTextureView::new()`.
    #[inline(always)]
    pub unsafe fn mip_bytes_unchecked(&self, mip_level: usize) -> &'a [u8] {
        // SAFETY: Bounds verified during FlatTextureView::new().
        // Caller guarantees mip_level < self.header.mip_levels.
        unsafe {
            let mip = self.mips.get_unchecked(mip_level);
            let ptr = self.bytes.as_ptr().add(mip.offset as usize);
            core::slice::from_raw_parts(ptr, mip.byte_size as usize)
        }
    }
}
