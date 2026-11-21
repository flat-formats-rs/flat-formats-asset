//! Zero-allocation reader view for memory-mapped FlatAudio containers.

use crate::error::{FlatAudioError, Result};
use crate::header::{FlatAudioHeader, FlatAudioPacketDesc, FLATAUDIO_MAGIC, FLATAUDIO_VERSION};
use zerocopy::Ref;

/// Zero-allocation, borrow-based view over a contiguous FlatAudio byte slice.
#[derive(Debug, Clone, Copy)]
pub struct FlatAudioView<'a> {
    bytes: &'a [u8],
    header: &'a FlatAudioHeader,
    packets: &'a [FlatAudioPacketDesc],
}

impl<'a> FlatAudioView<'a> {
    /// Mounts and validates a FlatAudio container from an in-memory slice in $O(1)$ time.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let req_size = core::mem::size_of::<FlatAudioHeader>();
        if bytes.len() < req_size {
            return Err(FlatAudioError::BufferTooSmall {
                required: req_size,
                actual: bytes.len(),
            });
        }

        // 1. Branchless 32-byte alignment check
        let ptr_val = bytes.as_ptr() as usize;
        let align_req = core::mem::align_of::<FlatAudioHeader>();
        if (ptr_val & (align_req - 1)) != 0 {
            return Err(FlatAudioError::Misaligned {
                required: align_req,
                actual: ptr_val & (align_req - 1),
            });
        }

        // 2. Read 32-byte header
        let (header_ref, _) = Ref::<&'a [u8], FlatAudioHeader>::from_prefix(bytes)
            .map_err(|_| FlatAudioError::InvalidLayout)?;
        let header = Ref::into_ref(header_ref);

        if header.magic != FLATAUDIO_MAGIC {
            return Err(FlatAudioError::InvalidMagic {
                expected: FLATAUDIO_MAGIC,
                actual: header.magic,
            });
        }

        if header.version != FLATAUDIO_VERSION {
            return Err(FlatAudioError::UnsupportedVersion(header.version));
        }

        // 3. Validate and slice packet index directory (immediately follows 32-byte header)
        let packet_count = header.packet_count as usize;
        let packet_table_start = 32usize;
        let packet_table_size = packet_count
            .checked_mul(core::mem::size_of::<FlatAudioPacketDesc>())
            .ok_or(FlatAudioError::InvalidLayout)?;
        let packet_table_end = packet_table_start.checked_add(packet_table_size).ok_or(
            FlatAudioError::PacketTableOutOfBounds {
                offset: packet_table_start as u32,
                count: header.packet_count,
            },
        )?;

        if packet_table_end > bytes.len() {
            return Err(FlatAudioError::PacketTableOutOfBounds {
                offset: packet_table_start as u32,
                count: header.packet_count,
            });
        }

        let packet_slice = &bytes[packet_table_start..packet_table_end];
        let (packets_ref, _) = Ref::<&'a [u8], [FlatAudioPacketDesc]>::from_prefix_with_elems(
            packet_slice,
            packet_count,
        )
        .map_err(|_| FlatAudioError::InvalidLayout)?;
        let packets = Ref::into_ref(packets_ref);

        // 4. Verify all packet payloads reside within file bounds
        for p in packets {
            let end = p.byte_offset.checked_add(p.byte_size as u64).ok_or(
                FlatAudioError::OutOfBounds {
                    offset: p.byte_offset,
                    size: p.byte_size as usize,
                    limit: bytes.len(),
                },
            )?;
            if end > bytes.len() as u64 {
                return Err(FlatAudioError::OutOfBounds {
                    offset: p.byte_offset,
                    size: p.byte_size as usize,
                    limit: bytes.len(),
                });
            }
        }

        Ok(Self {
            bytes,
            header,
            packets,
        })
    }

    #[inline(always)]
    pub fn header(&self) -> &FlatAudioHeader {
        self.header
    }

    #[inline(always)]
    pub fn codec_id(&self) -> crate::codec::AudioCodecId {
        self.header.codec_id
    }

    #[inline(always)]
    pub fn channels(&self) -> u8 {
        self.header.channels
    }

    #[inline(always)]
    pub fn sample_rate(&self) -> u32 {
        self.header.sample_rate
    }

    #[inline(always)]
    pub fn total_samples(&self) -> u32 {
        self.header.total_samples
    }

    #[inline(always)]
    pub fn is_looping(&self) -> bool {
        self.header.is_looping()
    }

    #[inline(always)]
    pub fn is_positional_3d(&self) -> bool {
        self.header.is_positional_3d()
    }

    #[inline(always)]
    pub fn loop_start_sample(&self) -> u32 {
        self.header.loop_start_sample
    }

    #[inline(always)]
    pub fn loop_end_sample(&self) -> u32 {
        self.header.loop_end_sample
    }

    #[inline(always)]
    pub fn packet_count(&self) -> usize {
        self.packets.len()
    }

    #[inline(always)]
    pub fn packets(&self) -> &[FlatAudioPacketDesc] {
        self.packets
    }

    /// Accesses the raw encoded bytes of an audio packet by index in $O(1)$.
    pub fn packet_bytes(&self, index: usize) -> Result<&'a [u8]> {
        let p = self
            .packets
            .get(index)
            .ok_or(FlatAudioError::PacketOutOfBounds {
                requested: index,
                total: self.packets.len(),
            })?;
        let start = p.byte_offset as usize;
        let end = start + p.byte_size as usize;
        Ok(&self.bytes[start..end])
    }

    /// Accesses the raw PCM16 samples of an audio packet with zero-copy.
    ///
    /// Available only when container format is `AudioCodecId::PCM16`.
    pub fn packet_pcm16(&self, index: usize) -> Result<&'a [i16]> {
        if self.header.codec_id != crate::codec::AudioCodecId::PCM16 {
            return Err(FlatAudioError::UnsupportedCodec(self.header.codec_id.0));
        }
        let raw = self.packet_bytes(index)?;
        let (samples_ref, _) = Ref::<&'a [u8], [i16]>::from_prefix_with_elems(
            raw,
            raw.len() / core::mem::size_of::<i16>(),
        )
        .map_err(|_| FlatAudioError::InvalidLayout)?;
        Ok(Ref::into_ref(samples_ref))
    }
}
