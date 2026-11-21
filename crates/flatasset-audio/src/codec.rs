//! Open-World Audio Codec Identifiers.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Open-world audio codec identifier.
///
/// Transparent 16-bit identifier enabling custom engine extensions
/// without breaking ABI compatibility or central enum coupling.
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    FromBytes,
    Immutable,
    IntoBytes,
    KnownLayout,
    Default,
)]
#[repr(transparent)]
pub struct AudioCodecId(pub u16);

impl AudioCodecId {
    /// Raw uncompressed 8-bit unsigned PCM (GoldSrc legacy audio).
    pub const PCM8: Self = Self(0x0000);
    /// Raw uncompressed linear 16-bit signed PCM (Direct-to-Mixer, 0 CPU decode cost).
    pub const PCM16: Self = Self(0x0001);
    /// Low-latency lossy Opus audio stream (world standard for games, speech & ambient).
    pub const OPUS: Self = Self(0x0002);
    /// ADPCM 4:1 compression (GoldSrc legacy audio).
    pub const ADPCM: Self = Self(0x0003);
    /// Quite OK Audio (QOA) / Vorbis placeholder.
    pub const QOA: Self = Self(0x0004);

    /// Returns a human-readable display string for canonical codecs.
    pub const fn name(&self) -> &'static str {
        match self.0 {
            0x0000 => "PCM8",
            0x0001 => "PCM16",
            0x0002 => "Opus",
            0x0003 => "ADPCM",
            0x0004 => "QOA",
            _ => "Custom",
        }
    }
}
