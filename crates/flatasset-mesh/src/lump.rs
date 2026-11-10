//! Dynamic Lump Directory Table and Lump Identifiers.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub const LUMP_FLAG_REQUIRED: u16 = 1 << 0;
pub const LUMP_FLAG_COMPRESSED_ZSTD: u16 = 1 << 1;

/// Standard Lump Identifiers (sequential 0..63 mapped to 64-bit lump_mask).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum LumpId {
    /// Stream 0: Vertex Positions ([f32; 3] + u32 cluster_id, 16B aligned).
    PositionStream = 0,
    /// Stream 1: Surface Attributes (A2B10G10R10 Normal + FP16 UV, 8B).
    AttributeStream = 1,
    /// Stream 2: Skeletal Rig (Joints u8x4 + Weights u8x4, 8B).
    SkinRigStream = 2,
    /// Stream 3: Triangle Index Buffer (u16 or u32).
    IndexBuffer = 3,
    /// Meshlet Cluster Lump (FlatMeshletDesc, 32B aligned).
    MeshletClusters = 4,
    /// Meshlet Micro-Indices Lump (Packed vertex indices inside meshlets).
    MeshletIndices = 5,
    /// GPU Indirect Draw Buffer (FlatDrawIndexedIndirect, 20B / 16B aligned).
    IndirectDrawBuffer = 6,
    /// Skeleton Hierarchy & Bind Poses.
    SkeletonHierarchy = 7,
    /// Animation Tracks & Keyframes.
    AnimationTracks = 8,
    /// Interned String Pool (UTF-8, null-delimited).
    StringTable = 9,
    /// Embedded sandboxed script or weapon/model logic (Wasm, Lua, Rhai, KDL Event Graph).
    Script = 10,
}

/// 32-byte Dynamic Lump Entry Descriptor.
///
/// Fits 2 full lump descriptors in a single 64-byte L1 cache line.
/// Zero padding holes.
#[derive(Copy, Clone, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C, align(32))]
pub struct FlatMeshLumpEntry {
    /// 2 bytes: Unique lump type identifier (matches LumpId 0..63).
    pub lump_id: u16,
    /// 2 bytes: Bit 0 = LUMP_FLAG_REQUIRED, Bit 1 = LUMP_FLAG_COMPRESSED_ZSTD.
    pub flags: u16,
    /// 2 bytes: Lump schema generation / version.
    pub version: u16,
    /// 1 byte: Stride of an individual element in bytes (e.g. 16 for position, 8 for attr).
    pub element_stride: u8,
    /// 1 byte: Required byte alignment expressed as log2 (e.g. 4 for 16B, 6 for 64B).
    pub alignment_log2: u8,
    /// 8 bytes: Byte offset from container start (guaranteed aligned to 1 << alignment_log2).
    pub offset: u64,
    /// 8 bytes: Uncompressed byte length of the lump payload.
    pub uncompressed_size: u64,
    /// 4 bytes: Total elements stored in this lump.
    pub element_count: u32,
    /// 4 bytes: CRC32 integrity checksum of payload bytes.
    pub crc32: u32,
}

const _: () = assert!(core::mem::size_of::<FlatMeshLumpEntry>() == 32);
const _: () = assert!(core::mem::align_of::<FlatMeshLumpEntry>() == 32);

impl FlatMeshLumpEntry {
    #[inline(always)]
    pub fn alignment(&self) -> usize {
        1usize
            .checked_shl(self.alignment_log2 as u32)
            .unwrap_or(usize::MAX)
    }

    #[inline(always)]
    pub fn is_required(&self) -> bool {
        (self.flags & LUMP_FLAG_REQUIRED) != 0
    }

    #[inline(always)]
    pub fn is_compressed(&self) -> bool {
        (self.flags & LUMP_FLAG_COMPRESSED_ZSTD) != 0
    }
}
