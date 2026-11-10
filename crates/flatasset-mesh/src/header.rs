//! Container header and packed format signature.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// 8-byte packed format signature (Magic 48-bit + Major 8-bit + Minor 8-bit).
///
/// Enables a single 64-bit integer comparison (`cmp [rsi], rax`) on load,
/// eliminating multi-byte string comparisons and branching.
#[derive(Copy, Clone, Debug, PartialEq, Eq, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(transparent)]
pub struct FormatSignature(pub u64);

impl FormatSignature {
    /// Canonical v1.0 signature for FlatMesh (`b"FLTMSH"` + major 1 + minor 0).
    pub const FLATMESH_V1_0: Self = Self::pack(*b"FLTMSH", 1, 0);

    /// Pack 6-byte ASCII magic and 2-byte version into a canonical 64-bit word.
    pub const fn pack(magic: [u8; 6], major: u8, minor: u8) -> Self {
        let m = u64::from_le_bytes([
            magic[0], magic[1], magic[2], magic[3], magic[4], magic[5], 0, 0,
        ]);
        Self(m | ((major as u64) << 48) | ((minor as u64) << 56))
    }

    /// Fast single-cycle signature validation.
    #[inline(always)]
    pub fn is_valid(&self, expected: Self) -> bool {
        self.0 == expected.0
    }

    /// Extract `(major, minor)` version numbers.
    #[inline(always)]
    pub fn version(&self) -> (u8, u8) {
        ((self.0 >> 48) as u8, (self.0 >> 56) as u8)
    }
}
const _: () = assert!(core::mem::size_of::<FormatSignature>() == 8);
const _: () = assert!(core::mem::align_of::<FormatSignature>() == 8);

/// 64-byte ultra-compact container header.
///
/// Fits into exactly one full 64-byte CPU cache line (Zen/Golden Cove/Apple Silicon).
/// Contains zero padding holes, branchless lump bitmask, and complete culling primitives (Sphere + AABB).
#[derive(Copy, Clone, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C, align(64))]
pub struct FlatMeshHeader {
    /// 8 bytes (offset 0..8): Packed Magic + Major/Minor version.
    pub signature: FormatSignature,
    /// 2 bytes (offset 8..10): Offset to lump table divided by 32 (e.g. 2 means 64 bytes from file start).
    pub lump_table_offset_div32: u16,
    /// 2 bytes (offset 10..12): Number of lump directory entries.
    pub lump_count: u16,
    /// 4 bytes (offset 12..16): Primary feature flags.
    pub flags: u32,
    /// 8 bytes (offset 16..24): Bitmask of active lumps (bits 0..63) for O(1) branchless popcount rank indexing.
    pub lump_mask: u64,
    /// 16 bytes (offset 24..40): Bounding sphere (xyz = center, w = radius) for frustum culling.
    pub bounding_sphere: [f32; 4],
    /// 12 bytes (offset 40..52): Axis-aligned bounding box min coordinate [x, y, z].
    pub aabb_min: [f32; 3],
    /// 12 bytes (offset 52..64): Axis-aligned bounding box max coordinate [x, y, z].
    pub aabb_max: [f32; 3],
}

// Compile-time size and alignment invariants (Scrooge principles)
const _: () = assert!(core::mem::size_of::<FlatMeshHeader>() == 64);
const _: () = assert!(core::mem::align_of::<FlatMeshHeader>() == 64);

impl FlatMeshHeader {
    pub const FLAG_HAS_MESHLETS: u32 = 1 << 0;
    pub const FLAG_HAS_SKELETON: u32 = 1 << 1;
    pub const FLAG_IS_STREAM_ZSTD: u32 = 1 << 2;

    #[inline(always)]
    pub fn lump_table_offset(&self) -> usize {
        (self.lump_table_offset_div32 as usize) * 32
    }

    #[inline(always)]
    pub fn has_meshlets(&self) -> bool {
        (self.flags & Self::FLAG_HAS_MESHLETS) != 0
    }

    #[inline(always)]
    pub fn has_skeleton(&self) -> bool {
        (self.flags & Self::FLAG_HAS_SKELETON) != 0
    }
}
