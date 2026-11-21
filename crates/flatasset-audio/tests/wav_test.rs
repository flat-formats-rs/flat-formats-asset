use flatasset_audio::{convert_wav_to_fls, parse_wav, AudioCodecId, FlatAudioView};

fn create_mock_wav(
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    samples: &[u8],
    cue_sample: Option<u32>,
) -> Vec<u8> {
    let mut data = Vec::new();
    // RIFF header placeholder
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&0u32.to_le_bytes()); // placeholder
    data.extend_from_slice(b"WAVE");

    // fmt chunk
    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&16u32.to_le_bytes()); // 16 bytes for PCM
    data.extend_from_slice(&1u16.to_le_bytes()); // wFormatTag = 1 (PCM)
    data.extend_from_slice(&channels.to_le_bytes());
    data.extend_from_slice(&sample_rate.to_le_bytes());
    let block_align = channels * (bits_per_sample / 8);
    let avg_bps = sample_rate * (block_align as u32);
    data.extend_from_slice(&avg_bps.to_le_bytes());
    data.extend_from_slice(&block_align.to_le_bytes());
    data.extend_from_slice(&bits_per_sample.to_le_bytes());

    // optional cue chunk
    if let Some(cue_offset) = cue_sample {
        data.extend_from_slice(b"cue ");
        let cue_size = 4 + 24;
        data.extend_from_slice(&(cue_size as u32).to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // 1 cue point
        data.extend_from_slice(&1u32.to_le_bytes()); // id
        data.extend_from_slice(&0u32.to_le_bytes()); // pos
        data.extend_from_slice(b"data"); // fcc
        data.extend_from_slice(&0u32.to_le_bytes()); // chunk start
        data.extend_from_slice(&0u32.to_le_bytes()); // block start
        data.extend_from_slice(&cue_offset.to_le_bytes()); // sample offset
    }

    // data chunk
    data.extend_from_slice(b"data");
    data.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    data.extend_from_slice(samples);

    // fixup RIFF size
    let riff_size = (data.len() - 8) as u32;
    data[4..8].copy_from_slice(&riff_size.to_le_bytes());

    data
}

#[test]
fn test_parse_wav_pcm8_with_cue() {
    let mock_samples = vec![128u8; 100];
    let wav_bytes = create_mock_wav(1, 22050, 8, &mock_samples, Some(42));

    let wav = parse_wav(&wav_bytes).expect("Failed to parse valid WAV");
    assert_eq!(wav.channels, 1);
    assert_eq!(wav.sample_rate, 22050);
    assert_eq!(wav.bits_per_sample, 8);
    assert_eq!(wav.loop_start_sample, Some(42));
    assert_eq!(wav.data.len(), 100);
}

#[test]
fn test_convert_wav_to_fls() {
    let mock_samples = vec![128u8; 100];
    let wav_bytes = create_mock_wav(1, 22050, 8, &mock_samples, Some(10));

    // Convert with normalization to PCM16
    let fls_bytes = convert_wav_to_fls(&wav_bytes, true).expect("WAV conversion failed");

    // Align to 32 bytes as mmap guarantees
    let mut aligned = Vec::with_capacity(fls_bytes.len() + 32);
    let offset = (32 - (aligned.as_ptr() as usize % 32)) % 32;
    aligned.resize(offset, 0);
    aligned.extend_from_slice(&fls_bytes);
    let aligned_slice = &aligned[offset..offset + fls_bytes.len()];

    let view = FlatAudioView::new(aligned_slice).expect("Valid FlatAudioView");

    assert_eq!(view.channels(), 1);
    assert_eq!(view.sample_rate(), 22050);
    assert_eq!(view.total_samples(), 100);
    assert_eq!(view.codec_id(), AudioCodecId::PCM16);
    assert!(view.is_looping());
    assert_eq!(view.loop_start_sample(), 10);
    assert_eq!(view.loop_end_sample(), 100);
    assert_eq!(view.packet_count(), 1);

    let packet_data = view.packet_bytes(0).unwrap();
    assert_eq!(packet_data.len(), 200); // 100 samples * 2 bytes
}
