//! Zero-allocation reader view for memory-mapped FlatMesh containers.

use crate::error::{FlatMeshError, Result};
use crate::header::{FlatMeshHeader, FormatSignature};
use crate::lump::{FlatMeshLumpEntry, LumpId};
use crate::meshlet::{FlatDrawIndexedIndirect, FlatMeshletDesc};
use crate::vertex::{FlatAttributeVertex, FlatPositionVertex, FlatSkinVertex};
use zerocopy::Ref;

/// Zero-allocation, borrow-based view over a contiguous byte slice (e.g. from `memmap2::Mmap`).
#[derive(Debug, Clone, Copy)]
pub struct FlatMeshView<'a> {
    bytes: &'a [u8],
    header: &'a FlatMeshHeader,
    lumps: &'a [FlatMeshLumpEntry],
}

impl<'a> FlatMeshView<'a> {
    /// Mount a FlatMesh container from a byte slice with zero heap allocations.
    ///
    /// Validates magic signature, 64-byte cache line alignment, and bounds for the lump table.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let req_size = core::mem::size_of::<FlatMeshHeader>();
        if bytes.len() < req_size {
            return Err(FlatMeshError::BufferTooSmall {
                required: req_size,
                actual: bytes.len(),
            });
        }

        // 1. Branchless 64-byte cache line alignment check (ptr & 63 == 0)
        let ptr_val = bytes.as_ptr() as usize;
        let align_req = core::mem::align_of::<FlatMeshHeader>();
        if (ptr_val & (align_req - 1)) != 0 {
            return Err(FlatMeshError::Misaligned {
                required: align_req,
                actual: ptr_val & (align_req - 1),
            });
        }

        // 2. Zero-copy read 64-byte header
        let (header_ref, _) = Ref::<&'a [u8], FlatMeshHeader>::from_prefix(bytes)
            .map_err(|_| FlatMeshError::InvalidLayout)?;
        let header = Ref::into_ref(header_ref);

        // 3. Fast single-cycle signature validation
        if !header.signature.is_valid(FormatSignature::FLATMESH_V1_0) {
            return Err(FlatMeshError::InvalidSignature {
                expected: FormatSignature::FLATMESH_V1_0.0,
                actual: header.signature.0,
            });
        }

        // 4. Validate and slice Dynamic Lump Table
        let lump_table_start = header.lump_table_offset();
        let lump_table_size =
            (header.lump_count as usize) * core::mem::size_of::<FlatMeshLumpEntry>();
        let lump_table_end =
            lump_table_start
                .checked_add(lump_table_size)
                .ok_or(FlatMeshError::OutOfBounds {
                    offset: lump_table_start as u64,
                    size: lump_table_size as u64,
                    limit: bytes.len(),
                })?;

        if lump_table_end > bytes.len() {
            return Err(FlatMeshError::OutOfBounds {
                offset: lump_table_start as u64,
                size: lump_table_size as u64,
                limit: bytes.len(),
            });
        }

        let lump_slice = &bytes[lump_table_start..lump_table_end];
        let (lumps_ref, _) = Ref::<&'a [u8], [FlatMeshLumpEntry]>::from_prefix_with_elems(
            lump_slice,
            header.lump_count as usize,
        )
        .map_err(|_| FlatMeshError::InvalidLayout)?;
        let lumps = Ref::into_ref(lumps_ref);

        // 5. Verify presence of all required lumps
        for lump in lumps {
            if lump.is_required() {
                let end = lump.offset.checked_add(lump.uncompressed_size).ok_or(
                    FlatMeshError::OutOfBounds {
                        offset: lump.offset,
                        size: lump.uncompressed_size,
                        limit: bytes.len(),
                    },
                )?;
                if end > bytes.len() as u64 {
                    return Err(FlatMeshError::OutOfBounds {
                        offset: lump.offset,
                        size: lump.uncompressed_size,
                        limit: bytes.len(),
                    });
                }
            }
        }

        Ok(Self {
            bytes,
            header,
            lumps,
        })
    }

    #[inline(always)]
    pub fn header(&self) -> &FlatMeshHeader {
        self.header
    }

    #[inline(always)]
    pub fn lumps(&self) -> &[FlatMeshLumpEntry] {
        self.lumps
    }

    /// O(1) branchless rank lookup via header.lump_mask (BMI2 bzhi + popcnt).
    ///
    /// Fallback to linear scan if lump_mask is unpopulated.
    #[inline(always)]
    pub fn find_lump(&self, lump_id: LumpId) -> Option<&'a FlatMeshLumpEntry> {
        let id = lump_id as u32;
        let mask = self.header.lump_mask;
        if mask != 0 {
            if (mask & (1u64 << id)) == 0 {
                return None;
            }
            let slot = silicon_kit::bits::rank64(mask, id);
            self.lumps.get(slot)
        } else {
            let target_id = lump_id as u16;
            self.lumps.iter().find(|l| l.lump_id == target_id)
        }
    }

    /// Access raw lump bytes with alignment and bounds verification.
    pub fn lump_bytes(&self, lump: &FlatMeshLumpEntry) -> Result<&'a [u8]> {
        let start = lump.offset as usize;
        let size = lump.uncompressed_size as usize;
        let end = start.checked_add(size).ok_or(FlatMeshError::OutOfBounds {
            offset: lump.offset,
            size: lump.uncompressed_size,
            limit: self.bytes.len(),
        })?;

        if end > self.bytes.len() {
            return Err(FlatMeshError::OutOfBounds {
                offset: lump.offset,
                size: lump.uncompressed_size,
                limit: self.bytes.len(),
            });
        }

        let slice = &self.bytes[start..end];
        let req_align = lump.alignment();
        if (slice.as_ptr() as usize) % req_align != 0 {
            return Err(FlatMeshError::Misaligned {
                required: req_align,
                actual: (slice.as_ptr() as usize) % req_align,
            });
        }

        Ok(slice)
    }

    /// Zero-copy access to Stream 0 (Position Stream).
    pub fn positions(&self) -> Result<&'a [FlatPositionVertex]> {
        let lump =
            self.find_lump(LumpId::PositionStream)
                .ok_or(FlatMeshError::MissingRequiredLump(
                    LumpId::PositionStream as u16,
                ))?;
        let bytes = self.lump_bytes(lump)?;
        let (slice_ref, _) = Ref::<&'a [u8], [FlatPositionVertex]>::from_prefix_with_elems(
            bytes,
            lump.element_count as usize,
        )
        .map_err(|_| FlatMeshError::InvalidLayout)?;
        Ok(Ref::into_ref(slice_ref))
    }

    /// Extreme performance unchecked slice fetch for Stream 0 positions.
    ///
    /// # Safety
    /// Caller guarantees that `self.find_lump(LumpId::PositionStream)` exists and points to
    /// valid aligned memory verified during `FlatMeshView::new()`.
    #[inline(always)]
    pub unsafe fn positions_unchecked(&self) -> &'a [FlatPositionVertex] {
        // SAFETY: Bounds and alignment invariants verified during FlatMeshView::new().
        // Caller guarantees lump exists.
        unsafe {
            let lump = self
                .find_lump(LumpId::PositionStream)
                .unwrap_unchecked();
            let ptr = self.bytes.as_ptr().add(lump.offset as usize) as *const FlatPositionVertex;
            core::slice::from_raw_parts(ptr, lump.element_count as usize)
        }
    }

    /// Zero-copy access to Stream 1 (Surface Attributes).
    pub fn attributes(&self) -> Result<&'a [FlatAttributeVertex]> {
        let lump =
            self.find_lump(LumpId::AttributeStream)
                .ok_or(FlatMeshError::MissingRequiredLump(
                    LumpId::AttributeStream as u16,
                ))?;
        let bytes = self.lump_bytes(lump)?;
        let (slice_ref, _) = Ref::<&'a [u8], [FlatAttributeVertex]>::from_prefix_with_elems(
            bytes,
            lump.element_count as usize,
        )
        .map_err(|_| FlatMeshError::InvalidLayout)?;
        Ok(Ref::into_ref(slice_ref))
    }

    /// Extreme performance unchecked slice fetch for Stream 1 attributes.
    ///
    /// # Safety
    /// Caller guarantees that `self.find_lump(LumpId::AttributeStream)` exists and points to
    /// valid aligned memory verified during `FlatMeshView::new()`.
    #[inline(always)]
    pub unsafe fn attributes_unchecked(&self) -> &'a [FlatAttributeVertex] {
        // SAFETY: Bounds and alignment invariants verified during FlatMeshView::new().
        // Caller guarantees lump exists.
        unsafe {
            let lump = self
                .find_lump(LumpId::AttributeStream)
                .unwrap_unchecked();
            let ptr = self.bytes.as_ptr().add(lump.offset as usize) as *const FlatAttributeVertex;
            core::slice::from_raw_parts(ptr, lump.element_count as usize)
        }
    }

    /// Zero-copy access to Stream 2 (Skeletal Rig).
    pub fn skin_weights(&self) -> Result<Option<&'a [FlatSkinVertex]>> {
        let Some(lump) = self.find_lump(LumpId::SkinRigStream) else {
            return Ok(None);
        };
        let bytes = self.lump_bytes(lump)?;
        let (slice_ref, _) = Ref::<&'a [u8], [FlatSkinVertex]>::from_prefix_with_elems(
            bytes,
            lump.element_count as usize,
        )
        .map_err(|_| FlatMeshError::InvalidLayout)?;
        Ok(Some(Ref::into_ref(slice_ref)))
    }

    /// Zero-copy access to 16-bit Triangle Index Buffer.
    pub fn indices_u16(&self) -> Result<&'a [u16]> {
        let lump =
            self.find_lump(LumpId::IndexBuffer)
                .ok_or(FlatMeshError::MissingRequiredLump(
                    LumpId::IndexBuffer as u16,
                ))?;
        let bytes = self.lump_bytes(lump)?;
        let (slice_ref, _) =
            Ref::<&'a [u8], [u16]>::from_prefix_with_elems(bytes, lump.element_count as usize)
                .map_err(|_| FlatMeshError::InvalidLayout)?;
        Ok(Ref::into_ref(slice_ref))
    }

    /// Zero-copy access to GPU Meshlet Cluster Descriptors.
    pub fn meshlets(&self) -> Result<Option<&'a [FlatMeshletDesc]>> {
        let Some(lump) = self.find_lump(LumpId::MeshletClusters) else {
            return Ok(None);
        };
        let bytes = self.lump_bytes(lump)?;
        let (slice_ref, _) = Ref::<&'a [u8], [FlatMeshletDesc]>::from_prefix_with_elems(
            bytes,
            lump.element_count as usize,
        )
        .map_err(|_| FlatMeshError::InvalidLayout)?;
        Ok(Some(Ref::into_ref(slice_ref)))
    }

    /// Zero-copy access to GPU Meshlet 8-bit Micro-Indices.
    pub fn meshlet_indices(&self) -> Result<Option<&'a [u8]>> {
        let Some(lump) = self.find_lump(LumpId::MeshletIndices) else {
            return Ok(None);
        };
        let bytes = self.lump_bytes(lump)?;
        let (slice_ref, _) = Ref::<&'a [u8], [u8]>::from_prefix_with_elems(
            bytes,
            lump.element_count as usize,
        )
        .map_err(|_| FlatMeshError::InvalidLayout)?;
        Ok(Some(Ref::into_ref(slice_ref)))
    }

    /// Zero-copy access to GPU Indirect Draw Commands.
    pub fn indirect_draws(&self) -> Result<Option<&'a [FlatDrawIndexedIndirect]>> {
        let Some(lump) = self.find_lump(LumpId::IndirectDrawBuffer) else {
            return Ok(None);
        };
        let bytes = self.lump_bytes(lump)?;
        let (slice_ref, _) = Ref::<&'a [u8], [FlatDrawIndexedIndirect]>::from_prefix_with_elems(
            bytes,
            lump.element_count as usize,
        )
        .map_err(|_| FlatMeshError::InvalidLayout)?;
        Ok(Some(Ref::into_ref(slice_ref)))
    }
}
