//! FlatWorld header and lump descriptor definitions.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// File format signature: b"FLW\0"
pub const FLATWORLD_MAGIC: [u8; 4] = *b"FLW\0";
/// Supported format version.
pub const FLATWORLD_VERSION: u32 = 1;
/// Fixed offset for lump payload alignment (4KB NVMe page alignment).
pub const CONTAINER_PAYLOAD_OFFSET: u64 = 4096;

/// Standard Lump Identifiers in FlatWorld.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LumpType {
    /// Visual meshlet geometry lump.
    Meshlets = 0,
    /// 4-way SIMD Quantized BVH collision tree.
    QBvh4 = 1,
    /// Convex hull brush / polyhedral collision data.
    CollisionHulls = 2,
    /// Succinct bitmask Potentially Visible Set (PVS).
    VisibilityBitmask = 3,
    /// Strongly typed KDL scene and entity graph.
    EntityGraph = 4,
    /// Material and FlatTexture manifest.
    MaterialManifest = 5,
    /// Pre-baked lightmap luxels or probe volumes.
    Lightmaps = 6,
    /// 3D navigation mesh (Recast/Detour format).
    NavMesh = 7,
    /// Embedded sandboxed script or event logic (Wasm, Lua, Rhai, KDL Event Graph).
    Script = 8,
    /// Unknown or custom extension lump.
    Unknown = u32::MAX,
}

impl From<u32> for LumpType {
    #[inline]
    fn from(val: u32) -> Self {
        match val {
            0 => Self::Meshlets,
            1 => Self::QBvh4,
            2 => Self::CollisionHulls,
            3 => Self::VisibilityBitmask,
            4 => Self::EntityGraph,
            5 => Self::MaterialManifest,
            6 => Self::Lightmaps,
            7 => Self::NavMesh,
            8 => Self::Script,
            _ => Self::Unknown,
        }
    }
}

/// Compression algorithm applied to a lump.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// Zero compression; direct memory map.
    Uncompressed = 0,
    /// Columnar bit-packed streaming via stitch-data.
    StitchData = 1,
    /// Standard block-compressed Zstd.
    Zstd = 2,
}

impl From<u32> for CompressionType {
    #[inline]
    fn from(val: u32) -> Self {
        match val {
            1 => Self::StitchData,
            2 => Self::Zstd,
            _ => Self::Uncompressed,
        }
    }
}

/// 64-byte cache-line sized FlatWorld file header with zero padding holes.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct FlatWorldHeader {
    /// File magic: b"FLW\0" (offset 0..4)
    pub magic: [u8; 4],
    /// Container format version (offset 4..8)
    pub version: u32,
    /// Feature and format flags (offset 8..12)
    pub flags: u32,
    /// Number of active lump descriptors following the header (offset 12..16)
    pub lump_count: u32,
    /// Axis-Aligned Bounding Box minimum [X, Y, Z] (offset 16..28)
    pub bounds_min: [f32; 3],
    /// Axis-Aligned Bounding Box maximum [X, Y, Z] (offset 28..40)
    pub bounds_max: [f32; 3],
    /// 128-bit truncated BLAKE3 checksum of all lump payloads (offset 40..56)
    pub content_hash_128: [u8; 16],
    /// Reserved space for future extensions (offset 56..64)
    pub _reserved: [u8; 8],
}

const _: () = assert!(core::mem::size_of::<FlatWorldHeader>() == 64);

/// 64-byte Lump Descriptor Table Entry with zero padding holes.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct FlatWorldLumpDesc {
    /// Lump identifier (`LumpType`) (offset 0..4)
    pub lump_type: u32,
    /// Compression applied to this lump (`CompressionType`) (offset 4..8)
    pub compression_type: u32,
    /// Absolute byte offset in container (4KB page-aligned) (offset 8..16)
    pub file_offset: u64,
    /// Byte size on disk (offset 16..24)
    pub compressed_size: u64,
    /// Byte footprint when uncompressed / mapped into RAM (offset 24..32)
    pub uncompressed_size: u64,
    /// Total count of records/elements in the lump (offset 32..40)
    pub element_count: u64,
    /// Stride in bytes per element (offset 40..44)
    pub element_stride: u32,
    /// Padding to align checksum to 16 bytes (offset 44..48)
    pub _pad_align: [u8; 4],
    /// Truncated 128-bit BLAKE3 lump checksum (offset 48..64)
    pub checksum_128: [u8; 16],
}

const _: () = assert!(core::mem::size_of::<FlatWorldLumpDesc>() == 64);
