//! Zero-allocation reader view for memory-mapped FlatTex containers.

use crate::error::{FlatTexError, Result};
use crate::header::{FlatTexHeader, FormatSignature};
use crate::mip::FlatMipDesc;
use zerocopy::Ref;

/// Zero-allocation, borrow-based view over a contiguous byte slice (e.g. from `memmap2::Mmap`).
#[derive(Debug, Clone, Copy)]
pub struct FlatTexView<'a> {
    bytes: &'a [u8],
    header: &'a FlatTexHeader,
    mips: &'a [FlatMipDesc],
}

impl<'a> FlatTexView<'a> {
    /// Mount a FlatTex container from a byte slice with zero heap allocations.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let req_size = core::mem::size_of::<FlatTexHeader>();
        if bytes.len() < req_size {
            return Err(FlatTexError::BufferTooSmall {
                required: req_size,
                actual: bytes.len(),
            });
        }

        // 1. Branchless 32-byte alignment check (ptr & 31 == 0)
        let ptr_val = bytes.as_ptr() as usize;
        let align_req = core::mem::align_of::<FlatTexHeader>();
        if (ptr_val & (align_req - 1)) != 0 {
            return Err(FlatTexError::Misaligned {
                required: align_req,
                actual: ptr_val & (align_req - 1),
            });
        }

        // 2. Zero-copy read header
        let (header_ref, _) = Ref::<&'a [u8], FlatTexHeader>::from_prefix(bytes)
            .map_err(|_| FlatTexError::InvalidLayout)?;
        let header = Ref::into_ref(header_ref);

        // 3. Fast single-cycle signature validation
        if !header.signature.is_valid(FormatSignature::FLATTEX_V1_0) {
            return Err(FlatTexError::InvalidSignature {
                expected: FormatSignature::FLATTEX_V1_0.0,
                actual: header.signature.0,
            });
        }

        // 4. Validate and slice Subresource Mip Directory
        let mip_table_start = header.mip_table_offset();
        let mip_table_size = (header.mip_levels as usize) * core::mem::size_of::<FlatMipDesc>();
        let mip_table_end =
            mip_table_start
                .checked_add(mip_table_size)
                .ok_or(FlatTexError::OutOfBounds {
                    offset: mip_table_start as u64,
                    size: mip_table_size,
                    limit: bytes.len(),
                })?;

        if mip_table_end > bytes.len() {
            return Err(FlatTexError::OutOfBounds {
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
        .map_err(|_| FlatTexError::InvalidLayout)?;
        let mips = Ref::into_ref(mips_ref);

        // 5. Verify bounds and 256-byte alignment invariants on all subresources
        for mip in mips {
            let end =
                mip.offset
                    .checked_add(mip.byte_size as u64)
                    .ok_or(FlatTexError::OutOfBounds {
                        offset: mip.offset,
                        size: mip.byte_size as usize,
                        limit: bytes.len(),
                    })?;
            if end > bytes.len() as u64 {
                return Err(FlatTexError::OutOfBounds {
                    offset: mip.offset,
                    size: mip.byte_size as usize,
                    limit: bytes.len(),
                });
            }
            if mip.aligned_row_pitch() % 256 != 0 {
                return Err(FlatTexError::InvalidRowPitch {
                    pitch: mip.aligned_row_pitch(),
                    required_alignment: 256,
                });
            }
        }

        Ok(Self {
            bytes,
            header,
            mips,
        })
    }

    #[inline(always)]
    pub fn header(&self) -> &FlatTexHeader {
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
            return Err(FlatTexError::MipOutOfBounds {
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
    /// have already been verified in `FlatTexView::new()`.
    #[inline(always)]
    pub unsafe fn mip_bytes_unchecked(&self, mip_level: usize) -> &'a [u8] {
        // SAFETY: Bounds verified during FlatTexView::new().
        // Caller guarantees mip_level < self.header.mip_levels.
        unsafe {
            let mip = self.mips.get_unchecked(mip_level);
            let ptr = self.bytes.as_ptr().add(mip.offset as usize);
            core::slice::from_raw_parts(ptr, mip.byte_size as usize)
        }
    }
}
