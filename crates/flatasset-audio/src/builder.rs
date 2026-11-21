//! Direct-to-Mixer FlatAudio Container Builder.

use crate::codec::AudioCodecId;
use crate::error::Result;
use crate::header::{FlatAudioHeader, FlatAudioPacketDesc, FLATAUDIO_MAGIC, FLATAUDIO_VERSION};
use alloc::vec::Vec;
use zerocopy::IntoBytes;

/// Input packet for the audio container builder.
pub struct AudioPacketInput {
    pub sample_count: u32,
    pub payload: Vec<u8>,
}

/// Builder for constructing valid Direct-to-Mixer FlatAudio (.fls) files.
pub struct FlatAudioBuilder {
    codec_id: AudioCodecId,
    channels: u8,
    flags: u8,
    sample_rate: u32,
    loop_start_sample: u32,
    loop_end_sample: u32,
    packets: Vec<AudioPacketInput>,
}

impl FlatAudioBuilder {
    /// Creates a new audio builder.
    pub fn new(codec_id: AudioCodecId, channels: u8, sample_rate: u32) -> Self {
        Self {
            codec_id,
            channels,
            flags: 0,
            sample_rate,
            loop_start_sample: 0,
            loop_end_sample: 0,
            packets: Vec::new(),
        }
    }

    /// Sets seamless looping parameters.
    pub fn with_loop(mut self, start_sample: u32, end_sample: u32) -> Self {
        self.flags |= FlatAudioHeader::FLAG_LOOPING;
        self.loop_start_sample = start_sample;
        self.loop_end_sample = end_sample;
        self
    }

    /// Sets 3D positional flag (for spatial audio emitters).
    pub fn with_positional_3d(mut self, is_3d: bool) -> Self {
        if is_3d {
            self.flags |= FlatAudioHeader::FLAG_POSITIONAL_3D;
        } else {
            self.flags &= !FlatAudioHeader::FLAG_POSITIONAL_3D;
        }
        self
    }

    /// Adds an encoded packet (Opus frame or PCM chunk).
    pub fn add_packet(&mut self, sample_count: u32, payload: Vec<u8>) {
        self.packets.push(AudioPacketInput {
            sample_count,
            payload,
        });
    }

    /// Serializes the complete `.fls` package.
    pub fn build(self) -> Result<Vec<u8>> {
        let packet_count = self.packets.len() as u32;
        let packet_table_offset = 32usize;
        let packet_table_size =
            (packet_count as usize) * core::mem::size_of::<FlatAudioPacketDesc>();

        // Align start of audio payload to 32 bytes
        let payload_start_offset = (packet_table_offset + packet_table_size + 31) & !31;

        let mut payload_data = Vec::new();
        let mut packet_descs = Vec::with_capacity(packet_count as usize);
        let mut total_samples = 0u32;

        for pkt in self.packets {
            let offset = (payload_start_offset + payload_data.len()) as u64;
            let byte_size = pkt.payload.len() as u32;

            packet_descs.push(FlatAudioPacketDesc {
                byte_offset: offset,
                byte_size,
                sample_count: pkt.sample_count,
            });

            total_samples += pkt.sample_count;
            payload_data.extend_from_slice(&pkt.payload);
        }

        let total_file_size = payload_start_offset + payload_data.len();
        let mut buffer = alloc::vec![0u8; total_file_size];

        let header = FlatAudioHeader {
            magic: FLATAUDIO_MAGIC,
            version: FLATAUDIO_VERSION,
            codec_id: self.codec_id,
            channels: self.channels,
            flags: self.flags,
            sample_rate: self.sample_rate,
            total_samples,
            loop_start_sample: self.loop_start_sample,
            loop_end_sample: self.loop_end_sample,
            packet_count,
        };

        // Write 32B header
        buffer[..32].copy_from_slice(header.as_bytes());

        // Write packet index table
        let mut table_cursor = packet_table_offset;
        for desc in &packet_descs {
            buffer[table_cursor..table_cursor + 16].copy_from_slice(desc.as_bytes());
            table_cursor += 16;
        }

        // Write payload
        buffer[payload_start_offset..total_file_size].copy_from_slice(&payload_data);

        Ok(buffer)
    }
}
