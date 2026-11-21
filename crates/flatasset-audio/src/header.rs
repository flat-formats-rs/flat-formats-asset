//! 32-Byte FlatAudio Container Header and 16-Byte Packet Descriptor.

use crate::codec::AudioCodecId;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Container magic for FlatAudio: `b"FLTS"` (0x53544C46 in LE)
pub const FLATAUDIO_MAGIC: u32 = 0x53544C46;
/// Current format version
pub const FLATAUDIO_VERSION: u32 = 1;

/// Exact 32-byte header for FlatAudio (`.fls`).
///
/// Fits into half of a 64-byte L1 CPU cache line.
/// Zero padding holes with natural field alignment (4 bytes).
#[derive(Copy, Clone, Debug, PartialEq, Eq, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C)]
pub struct FlatAudioHeader {
    /// 4 bytes (offset 0..4): Magic `FLTS`
    pub magic: u32,
    /// 4 bytes (offset 4..8): Format version
    pub version: u32,
    /// 2 bytes (offset 8..10): Open-World AudioCodecId
    pub codec_id: AudioCodecId,
    /// 1 byte (offset 10..11): Channel count (1 = mono, 2 = stereo)
    pub channels: u8,
    /// 1 byte (offset 11..12): Flags (bit 0: is_looping, bit 1: positional_3d)
    pub flags: u8,
    /// 4 bytes (offset 12..16): Sample rate in Hz (typically 48000 for Opus)
    pub sample_rate: u32,
    /// 4 bytes (offset 16..20): Total audio length in audio frames / samples
    pub total_samples: u32,
    /// 4 bytes (offset 20..24): Sample index where loop starts (for seamless ambience/music)
    pub loop_start_sample: u32,
    /// 4 bytes (offset 24..28): Sample index where loop ends (0 = loop to end)
    pub loop_end_sample: u32,
    /// 4 bytes (offset 28..32): Total number of encoded packets in this container
    pub packet_count: u32,
}

const _: () = assert!(core::mem::size_of::<FlatAudioHeader>() == 32);
const _: () = assert!(core::mem::align_of::<FlatAudioHeader>() == 4);

impl FlatAudioHeader {
    pub const FLAG_LOOPING: u8 = 1 << 0;
    pub const FLAG_POSITIONAL_3D: u8 = 1 << 1;

    #[inline(always)]
    pub fn is_looping(&self) -> bool {
        (self.flags & Self::FLAG_LOOPING) != 0
    }

    #[inline(always)]
    pub fn is_positional_3d(&self) -> bool {
        (self.flags & Self::FLAG_POSITIONAL_3D) != 0
    }
}

/// Exact 16-byte packet index descriptor for $O(1)$ random access.
///
/// 4 descriptors fit into one 64-byte L1 cache line.
/// Natural field alignment is 8 bytes due to `u64`.
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, FromBytes, Immutable, IntoBytes, KnownLayout,
)]
#[repr(C)]
pub struct FlatAudioPacketDesc {
    /// 8 bytes: Absolute byte offset of the encoded packet from container start
    pub byte_offset: u64,
    /// 4 bytes: Length of the encoded packet in bytes
    pub byte_size: u32,
    /// 4 bytes: Duration of this packet in audio samples (e.g. 960 for 20ms at 48kHz)
    pub sample_count: u32,
}

const _: () = assert!(core::mem::size_of::<FlatAudioPacketDesc>() == 16);
const _: () = assert!(core::mem::align_of::<FlatAudioPacketDesc>() == 8);
