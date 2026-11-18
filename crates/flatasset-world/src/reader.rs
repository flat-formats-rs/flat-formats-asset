//! Zero-allocation FlatWorld memory-mapped container viewer.

use zerocopy::FromBytes;

use crate::{
    header::{FlatWorldHeader, FlatWorldLumpDesc, LumpType, FLATWORLD_MAGIC, FLATWORLD_VERSION},
    meshlet::FlatWorldMeshlet,
    pvs::PvsView,
    qbvh::QBvh4Node,
};

/// Errors encountered when validating or reading a FlatWorld container.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FlatWorldError {
    #[error("Buffer too small: expected at least {expected} bytes, got {actual}")]
    BufferTooSmall { expected: usize, actual: usize },
    #[error("Invalid magic: expected b\"FLW\\0\", found {0:?}")]
    InvalidMagic([u8; 4]),
    #[error("Unsupported format version: {0} (supported: {FLATWORLD_VERSION})")]
    UnsupportedVersion(u32),
    #[error("Lump {lump_id} out of bounds: offset {offset} + size {size} > total {total}")]
    LumpOutOfBounds {
        lump_id: u32,
        offset: u64,
        size: u64,
        total: usize,
    },
    #[error("Lump {lump_id} alignment error: offset {offset} is not aligned to {required}")]
    MisalignedLump {
        lump_id: u32,
        offset: u64,
        required: usize,
    },
    #[error("Lump {0:?} was not found in container")]
    LumpNotFound(LumpType),
    #[error("Lump {0:?} is compressed (type {1}); use decompressed_lump_bytes")]
    LumpCompressed(LumpType, u32),
    #[error("Failed to decompress lump payload")]
    DecompressionFailed,
    #[error("Unsupported compression type: {0}")]
    UnsupportedCompression(u32),
    #[error("Invalid UTF-8 payload in entity lump")]
    InvalidUtf8,
}

/// Zero-copy, zero-allocation memory-mapped viewer of a FlatWorld container.
#[derive(Debug, Clone, Copy)]
pub struct FlatWorldView<'a> {
    raw_bytes: &'a [u8],
    header: &'a FlatWorldHeader,
    lumps: &'a [FlatWorldLumpDesc],
}

impl<'a> FlatWorldView<'a> {
    /// Validates and constructs a `FlatWorldView` over an in-memory slice or memory-mapped file.
    pub fn new(bytes: &'a [u8]) -> Result<Self, FlatWorldError> {
        if bytes.len() < core::mem::size_of::<FlatWorldHeader>() {
            return Err(FlatWorldError::BufferTooSmall {
                expected: core::mem::size_of::<FlatWorldHeader>(),
                actual: bytes.len(),
            });
        }

        let (header_ref, rest) = FlatWorldHeader::ref_from_prefix(bytes).map_err(|_| {
            FlatWorldError::BufferTooSmall {
                expected: core::mem::size_of::<FlatWorldHeader>(),
                actual: bytes.len(),
            }
        })?;

        if header_ref.magic != FLATWORLD_MAGIC {
            return Err(FlatWorldError::InvalidMagic(header_ref.magic));
        }
        if header_ref.version != FLATWORLD_VERSION {
            return Err(FlatWorldError::UnsupportedVersion(header_ref.version));
        }

        let lump_count = header_ref.lump_count as usize;
        let (lumps_ref, _) = <[FlatWorldLumpDesc]>::ref_from_prefix_with_elems(rest, lump_count)
            .map_err(|_| FlatWorldError::BufferTooSmall {
                expected: core::mem::size_of::<FlatWorldHeader>()
                    + lump_count * core::mem::size_of::<FlatWorldLumpDesc>(),
                actual: bytes.len(),
            })?;

        // Validate each lump offset, bounds, and uncompressed stride consistency
        for lump in lumps_ref {
            let end_offset = lump.file_offset.checked_add(lump.compressed_size).ok_or(
                FlatWorldError::LumpOutOfBounds {
                    lump_id: lump.lump_type,
                    offset: lump.file_offset,
                    size: lump.compressed_size,
                    total: bytes.len(),
                },
            )?;

            if end_offset as usize > bytes.len() {
                return Err(FlatWorldError::LumpOutOfBounds {
                    lump_id: lump.lump_type,
                    offset: lump.file_offset,
                    size: lump.compressed_size,
                    total: bytes.len(),
                });
            }

            // Invariant: for uncompressed structural lumps, validate expected element stride against size
            if lump.compression_type == 0 {
                let expected_stride = match lump.lump_type {
                    x if x == LumpType::QBvh4 as u32 => core::mem::size_of::<QBvh4Node>(),
                    x if x == LumpType::Meshlets as u32 => core::mem::size_of::<FlatWorldMeshlet>(),
                    _ => lump.element_stride as usize,
                };
                if lump.element_count > 0 && expected_stride > 0 {
                    let required = lump
                        .element_count
                        .checked_mul(expected_stride as u64)
                        .ok_or(FlatWorldError::LumpOutOfBounds {
                            lump_id: lump.lump_type,
                            offset: lump.file_offset,
                            size: lump.compressed_size,
                            total: bytes.len(),
                        })?;
                    if required > lump.compressed_size {
                        return Err(FlatWorldError::LumpOutOfBounds {
                            lump_id: lump.lump_type,
                            offset: lump.file_offset,
                            size: lump.compressed_size,
                            total: bytes.len(),
                        });
                    }
                }
            }
        }

        Ok(Self {
            raw_bytes: bytes,
            header: header_ref,
            lumps: lumps_ref,
        })
    }

    /// Accesses the container header.
    #[inline(always)]
    pub fn header(&self) -> &'a FlatWorldHeader {
        self.header
    }

    /// Returns the lump directory slice.
    #[inline(always)]
    pub fn lumps(&self) -> &'a [FlatWorldLumpDesc] {
        self.lumps
    }

    /// Finds a lump descriptor by lump type.
    #[inline]
    pub fn find_lump(&self, lump_type: LumpType) -> Option<&'a FlatWorldLumpDesc> {
        let target_id = lump_type as u32;
        self.lumps.iter().find(|l| l.lump_type == target_id)
    }

    /// Returns the raw byte slice of a lump payload.
    #[inline]
    pub fn lump_bytes(&self, lump_type: LumpType) -> Result<&'a [u8], FlatWorldError> {
        let lump = self
            .find_lump(lump_type)
            .ok_or(FlatWorldError::LumpNotFound(lump_type))?;
        let start = lump.file_offset as usize;
        let end = start + lump.compressed_size as usize;
        Ok(&self.raw_bytes[start..end])
    }

    /// Returns decompressed lump payload (zero-copy borrowed if uncompressed, or owned buffer if compressed).
    pub fn decompressed_lump_bytes(
        &self,
        lump_type: LumpType,
    ) -> Result<alloc::borrow::Cow<'a, [u8]>, FlatWorldError> {
        let lump = self
            .find_lump(lump_type)
            .ok_or(FlatWorldError::LumpNotFound(lump_type))?;
        let raw = self.lump_bytes(lump_type)?;

        match lump.compression_type {
            0 => Ok(alloc::borrow::Cow::Borrowed(raw)),
            #[cfg(feature = "zstd")]
            2 => {
                let decompressed =
                    zstd::decode_all(raw).map_err(|_| FlatWorldError::DecompressionFailed)?;
                Ok(alloc::borrow::Cow::Owned(decompressed))
            }
            other => Err(FlatWorldError::UnsupportedCompression(other)),
        }
    }

    /// Returns typed slice of Q-BVH4 collision nodes.
    pub fn qbvh_nodes(&self) -> Result<&'a [QBvh4Node], FlatWorldError> {
        let lump = self
            .find_lump(LumpType::QBvh4)
            .ok_or(FlatWorldError::LumpNotFound(LumpType::QBvh4))?;
        if lump.compression_type != 0 {
            return Err(FlatWorldError::LumpCompressed(
                LumpType::QBvh4,
                lump.compression_type,
            ));
        }
        let bytes = self.lump_bytes(LumpType::QBvh4)?;
        let count = lump.element_count as usize;
        let expected_size = count.checked_mul(core::mem::size_of::<QBvh4Node>()).ok_or(
            FlatWorldError::BufferTooSmall {
                expected: usize::MAX,
                actual: bytes.len(),
            },
        )?;
        if bytes.len() < expected_size {
            return Err(FlatWorldError::BufferTooSmall {
                expected: expected_size,
                actual: bytes.len(),
            });
        }
        let (nodes, _) = <[QBvh4Node]>::ref_from_prefix_with_elems(bytes, count).map_err(|_| {
            FlatWorldError::MisalignedLump {
                lump_id: lump.lump_type,
                offset: lump.file_offset,
                required: core::mem::align_of::<QBvh4Node>(),
            }
        })?;
        Ok(nodes)
    }

    /// Returns typed slice of static world meshlets.
    pub fn meshlets(&self) -> Result<&'a [FlatWorldMeshlet], FlatWorldError> {
        let lump = self
            .find_lump(LumpType::Meshlets)
            .ok_or(FlatWorldError::LumpNotFound(LumpType::Meshlets))?;
        if lump.compression_type != 0 {
            return Err(FlatWorldError::LumpCompressed(
                LumpType::Meshlets,
                lump.compression_type,
            ));
        }
        let bytes = self.lump_bytes(LumpType::Meshlets)?;
        let count = lump.element_count as usize;
        let expected_size = count
            .checked_mul(core::mem::size_of::<FlatWorldMeshlet>())
            .ok_or(FlatWorldError::BufferTooSmall {
                expected: usize::MAX,
                actual: bytes.len(),
            })?;
        if bytes.len() < expected_size {
            return Err(FlatWorldError::BufferTooSmall {
                expected: expected_size,
                actual: bytes.len(),
            });
        }
        let (meshlets, _) = <[FlatWorldMeshlet]>::ref_from_prefix_with_elems(bytes, count)
            .map_err(|_| FlatWorldError::MisalignedLump {
                lump_id: lump.lump_type,
                offset: lump.file_offset,
                required: core::mem::align_of::<FlatWorldMeshlet>(),
            })?;
        Ok(meshlets)
    }

    /// Returns a Succinct PVS viewer.
    pub fn pvs_view(&self) -> Result<PvsView<'a>, FlatWorldError> {
        let lump = self
            .find_lump(LumpType::VisibilityBitmask)
            .ok_or(FlatWorldError::LumpNotFound(LumpType::VisibilityBitmask))?;
        if lump.compression_type != 0 {
            return Err(FlatWorldError::LumpCompressed(
                LumpType::VisibilityBitmask,
                lump.compression_type,
            ));
        }
        let bytes = self.lump_bytes(LumpType::VisibilityBitmask)?;
        let cluster_count = lump.element_count as usize;
        PvsView::new(cluster_count, bytes).ok_or(FlatWorldError::BufferTooSmall {
            expected: cluster_count * cluster_count.div_ceil(8),
            actual: bytes.len(),
        })
    }

    /// Returns the entity scene graph as a UTF-8 KDL string slice.
    pub fn entity_kdl(&self) -> Result<&'a str, FlatWorldError> {
        let lump = self
            .find_lump(LumpType::EntityGraph)
            .ok_or(FlatWorldError::LumpNotFound(LumpType::EntityGraph))?;
        if lump.compression_type != 0 {
            return Err(FlatWorldError::LumpCompressed(
                LumpType::EntityGraph,
                lump.compression_type,
            ));
        }
        let bytes = self.lump_bytes(LumpType::EntityGraph)?;
        core::str::from_utf8(bytes).map_err(|_| FlatWorldError::InvalidUtf8)
    }
}
