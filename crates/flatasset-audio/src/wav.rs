//! Lightweight zero-dependency RIFF WAVE parser and FlatAudio converter.

use crate::builder::FlatAudioBuilder;
use crate::codec::AudioCodecId;
use crate::error::{FlatAudioError, Result};
use alloc::vec::Vec;

/// Parsed metadata from a RIFF WAVE file.
#[derive(Debug, Clone)]
pub struct WavInfo<'a> {
    pub channels: u8,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    pub loop_start_sample: Option<u32>,
    pub loop_end_sample: Option<u32>,
    pub data: &'a [u8],
}

/// Parses a RIFF WAVE byte slice into structured metadata and PCM sample slice.
pub fn parse_wav(bytes: &[u8]) -> Result<WavInfo<'_>> {
    if bytes.len() < 12 {
        return Err(FlatAudioError::InvalidWav("file smaller than RIFF header"));
    }

    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(FlatAudioError::InvalidWav("missing RIFF/WAVE header"));
    }

    let mut pos = 12usize;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut audio_format = None;
    let mut loop_start = None;
    let mut loop_end = None;
    let mut data_slice = None;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(
            bytes[pos + 4..pos + 8]
                .try_into()
                .map_err(|_| FlatAudioError::InvalidWav("chunk size decode error"))?,
        ) as usize;

        let chunk_data_start = pos + 8;
        let chunk_data_end = chunk_data_start + chunk_size;

        if chunk_data_end > bytes.len() {
            return Err(FlatAudioError::InvalidWav(
                "chunk bounds exceed file length",
            ));
        }

        match chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    return Err(FlatAudioError::InvalidWav("fmt chunk too small"));
                }
                let fmt_slice = &bytes[chunk_data_start..chunk_data_end];
                let w_format = u16::from_le_bytes([fmt_slice[0], fmt_slice[1]]);
                let n_channels = u16::from_le_bytes([fmt_slice[2], fmt_slice[3]]);
                let n_samples_per_sec =
                    u32::from_le_bytes([fmt_slice[4], fmt_slice[5], fmt_slice[6], fmt_slice[7]]);
                let w_bits = u16::from_le_bytes([fmt_slice[14], fmt_slice[15]]);

                audio_format = Some(w_format);
                channels = Some(n_channels as u8);
                sample_rate = Some(n_samples_per_sec);
                bits_per_sample = Some(w_bits);
            }
            b"cue " => {
                if chunk_size >= 4 {
                    let cue_slice = &bytes[chunk_data_start..chunk_data_end];
                    let num_cues = u32::from_le_bytes([
                        cue_slice[0],
                        cue_slice[1],
                        cue_slice[2],
                        cue_slice[3],
                    ]);
                    // Each cue point is 24 bytes: id (4), pos (4), fcc (4), chunk_start (4), block_start (4), sample_offset (4)
                    if num_cues >= 1 && chunk_size >= 4 + 24 {
                        let sample_offset = u32::from_le_bytes([
                            cue_slice[24],
                            cue_slice[25],
                            cue_slice[26],
                            cue_slice[27],
                        ]);
                        loop_start = Some(sample_offset);
                    }
                }
            }
            b"smpl" => {
                // Sampler chunk: check for loop points
                // Offset 28: dwNumSampleLoops (4B), Offset 36+: loops array (24B each)
                // SampleLoop: dwIdentifier(4), dwType(4), dwStart(4), dwEnd(4), dwFraction(4), dwPlayCount(4)
                if chunk_size >= 36 + 24 {
                    let smpl_slice = &bytes[chunk_data_start..chunk_data_end];
                    let num_loops = u32::from_le_bytes([
                        smpl_slice[28],
                        smpl_slice[29],
                        smpl_slice[30],
                        smpl_slice[31],
                    ]);
                    if num_loops >= 1 {
                        let loop_start_val = u32::from_le_bytes([
                            smpl_slice[36 + 8],
                            smpl_slice[36 + 9],
                            smpl_slice[36 + 10],
                            smpl_slice[36 + 11],
                        ]);
                        let loop_end_val = u32::from_le_bytes([
                            smpl_slice[36 + 12],
                            smpl_slice[36 + 13],
                            smpl_slice[36 + 14],
                            smpl_slice[36 + 15],
                        ]);
                        loop_start = Some(loop_start_val);
                        loop_end = Some(loop_end_val);
                    }
                }
            }
            b"data" => {
                data_slice = Some(&bytes[chunk_data_start..chunk_data_end]);
            }
            _ => {
                // Ignore other metadata chunks (LIST, INFO, etc.)
            }
        }

        // Chunks are padded to 2-byte word boundary
        let pad = chunk_size & 1;
        pos = chunk_data_end + pad;
    }

    let fmt = audio_format.ok_or(FlatAudioError::InvalidWav("missing fmt chunk"))?;
    if fmt != 1 {
        return Err(FlatAudioError::UnsupportedCodec(fmt));
    }

    let channels = channels.ok_or(FlatAudioError::InvalidWav("missing channel count"))?;
    let sample_rate = sample_rate.ok_or(FlatAudioError::InvalidWav("missing sample rate"))?;
    let bits_per_sample =
        bits_per_sample.ok_or(FlatAudioError::InvalidWav("missing bits per sample"))?;
    let data = data_slice.ok_or(FlatAudioError::InvalidWav("missing data chunk"))?;

    Ok(WavInfo {
        channels,
        sample_rate,
        bits_per_sample,
        loop_start_sample: loop_start,
        loop_end_sample: loop_end,
        data,
    })
}

/// Converts raw WAV bytes into a Direct-to-Mixer FlatAudio (.fls) package.
///
/// If `normalize_to_pcm16` is true, 8-bit unsigned PCM is expanded to signed 16-bit LE PCM.
/// Otherwise, 8-bit PCM is preserved verbatim as `AudioCodecId::PCM8`.
pub fn convert_wav_to_fls(bytes: &[u8], normalize_to_pcm16: bool) -> Result<Vec<u8>> {
    let wav = parse_wav(bytes)?;

    let (codec_id, payload, total_samples) = match wav.bits_per_sample {
        8 => {
            let total_samples = wav.data.len() as u32 / (wav.channels as u32);
            if normalize_to_pcm16 {
                let mut out_pcm16 = Vec::with_capacity(wav.data.len() * 2);
                for &b in wav.data {
                    // Convert unsigned 8-bit [0..255] (center 128) to signed 16-bit [-32768..32767]
                    let sample_i16 = ((b as i16) - 128) * 257;
                    out_pcm16.extend_from_slice(&sample_i16.to_le_bytes());
                }
                (AudioCodecId::PCM16, out_pcm16, total_samples)
            } else {
                (AudioCodecId::PCM8, wav.data.to_vec(), total_samples)
            }
        }
        16 => {
            let bytes_per_sample = 2 * (wav.channels as usize);
            let total_samples = (wav.data.len() / bytes_per_sample) as u32;
            (AudioCodecId::PCM16, wav.data.to_vec(), total_samples)
        }
        other => return Err(FlatAudioError::UnsupportedCodec(other)),
    };

    let mut builder = FlatAudioBuilder::new(codec_id, wav.channels, wav.sample_rate);

    if let Some(start) = wav.loop_start_sample {
        let end = wav.loop_end_sample.unwrap_or(total_samples);
        builder = builder.with_loop(start, end);
    }

    // Single monolithic packet for linear PCM Direct-to-Mixer playback
    builder.add_packet(total_samples, payload);

    builder.build()
}
