//! Error types for FlatMesh decoding and validation.

use core::fmt;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FlatMeshError {
    /// Buffer is smaller than the required header size.
    BufferTooSmall { required: usize, actual: usize },
    /// Magic signature mismatch or unsupported major version.
    InvalidSignature { expected: u64, actual: u64 },
    /// Unsupported schema version.
    UnsupportedVersion { major: u8, minor: u8 },
    /// Memory address or buffer slice is not aligned to the required boundary.
    Misaligned { required: usize, actual: usize },
    /// Requested lump or offset is out of bounds.
    OutOfBounds {
        offset: u64,
        size: u64,
        limit: usize,
    },
    /// A required lump (with LUMP_FLAG_REQUIRED) is missing in the container.
    MissingRequiredLump(u16),
    /// Data layout or element count violates struct invariants.
    InvalidLayout,
}

impl fmt::Display for FlatMeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall { required, actual } => {
                write!(
                    f,
                    "Buffer too small: required {required} bytes, got {actual}"
                )
            }
            Self::InvalidSignature { expected, actual } => {
                write!(
                    f,
                    "Invalid magic signature: expected 0x{expected:016X}, got 0x{actual:016X}"
                )
            }
            Self::UnsupportedVersion { major, minor } => {
                write!(f, "Unsupported format version: {major}.{minor}")
            }
            Self::Misaligned { required, actual } => {
                write!(
                    f,
                    "Misaligned buffer: required {required}-byte alignment, pointer has {actual}"
                )
            }
            Self::OutOfBounds {
                offset,
                size,
                limit,
            } => {
                write!(
                    f,
                    "Out of bounds slice: offset {offset} + size {size} > limit {limit}"
                )
            }
            Self::MissingRequiredLump(id) => {
                write!(f, "Missing required lump ID 0x{id:04X}")
            }
            Self::InvalidLayout => {
                write!(f, "Invalid binary structure layout")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FlatMeshError {}

pub type Result<T> = core::result::Result<T, FlatMeshError>;
