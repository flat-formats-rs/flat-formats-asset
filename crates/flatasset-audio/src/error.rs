//! Error types for FlatAudio container processing.

#[derive(Debug, thiserror::Error)]
pub enum FlatAudioError {
    #[error("Buffer too small: required {required}, actual {actual}")]
    BufferTooSmall { required: usize, actual: usize },
    #[error("Invalid FlatAudio magic: expected 0x{expected:08X}, actual 0x{actual:08X}")]
    InvalidMagic { expected: u32, actual: u32 },
    #[error("Unsupported FlatAudio version: {0}")]
    UnsupportedVersion(u32),
    #[error("Misaligned buffer: required {required}, actual {actual}")]
    Misaligned { required: usize, actual: usize },
    #[error("Invalid container layout")]
    InvalidLayout,
    #[error("Packet table out of bounds: offset {offset}, count {count}")]
    PacketTableOutOfBounds { offset: u32, count: u32 },
    #[error("Packet index out of bounds: requested {requested}, total {total}")]
    PacketOutOfBounds { requested: usize, total: usize },
    #[error("Packet payload out of bounds: offset {offset}, size {size}, limit {limit}")]
    OutOfBounds {
        offset: u64,
        size: usize,
        limit: usize,
    },
    #[error("Unsupported codec identifier: 0x{0:04X}")]
    UnsupportedCodec(u16),
    #[error("Invalid WAV format: {0}")]
    InvalidWav(&'static str),
}

pub type Result<T> = core::result::Result<T, FlatAudioError>;
