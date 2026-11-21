//! Error types for FlatTexture decoding and validation.

use core::fmt;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FlatTextureError {
    /// Buffer is smaller than the required header size.
    BufferTooSmall { required: usize, actual: usize },
    /// Magic signature mismatch or unsupported major version.
    InvalidSignature { expected: u64, actual: u64 },
    /// Unsupported schema version.
    UnsupportedVersion { major: u8, minor: u8 },
    /// Memory address or buffer slice is not aligned to the required boundary.
    Misaligned { required: usize, actual: usize },
    /// Requested mip level is out of bounds.
    MipOutOfBounds { requested: u8, max: u8 },
    /// Payload slice is out of bounds of the container buffer.
    OutOfBounds {
        offset: u64,
        size: usize,
        limit: usize,
    },
    /// Subresource row pitch does not conform to the 256-byte DMA alignment requirement.
    InvalidRowPitch { pitch: u32, required_alignment: u32 },
    /// Data layout violates struct invariants.
    InvalidLayout,
}

impl fmt::Display for FlatTextureError {
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
            Self::MipOutOfBounds { requested, max } => {
                write!(f, "Requested mip {requested} >= total mip levels {max}")
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
            Self::InvalidRowPitch {
                pitch,
                required_alignment,
            } => {
                write!(
                    f,
                    "Row pitch {pitch} is not aligned to {required_alignment} bytes"
                )
            }
            Self::InvalidLayout => {
                write!(f, "Invalid binary structure layout")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FlatTextureError {}

pub type Result<T> = core::result::Result<T, FlatTextureError>;
