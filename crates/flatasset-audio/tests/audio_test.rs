use flatasset_audio::{AudioCodecId, FlatAudioBuilder, FlatAudioView};

#[test]
fn test_flat_audio_builder_and_view_roundtrip() {
    let mut builder = FlatAudioBuilder::new(AudioCodecId::OPUS, 1, 48000)
        .with_loop(9600, 48000)
        .with_positional_3d(true);

    // Add 3 mock audio packets
    let packet0 = vec![0xFA, 0xCE, 0x01, 0x02];
    let packet1 = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x55];
    let packet2 = vec![0xCA, 0xFE, 0xBA, 0xBE];

    builder.add_packet(960, packet0.clone());
    builder.add_packet(960, packet1.clone());
    builder.add_packet(960, packet2.clone());

    let bytes = builder.build().expect("build audio package");

    // Wrap in 32-byte aligned slice for zero-copy test
    let mut aligned = Vec::with_capacity(bytes.len() + 32);
    let offset = (32 - (aligned.as_ptr() as usize % 32)) % 32;
    aligned.resize(offset, 0);
    aligned.extend_from_slice(&bytes);
    let aligned_slice = &aligned[offset..offset + bytes.len()];

    let view = FlatAudioView::new(aligned_slice).expect("mount audio view");

    assert_eq!(view.header().codec_id, AudioCodecId::OPUS);
    assert_eq!(view.header().channels, 1);
    assert_eq!(view.header().sample_rate, 48000);
    assert_eq!(view.header().total_samples, 2880);
    assert!(view.header().is_looping());
    assert!(view.header().is_positional_3d());
    assert_eq!(view.header().loop_start_sample, 9600);
    assert_eq!(view.header().loop_end_sample, 48000);
    assert_eq!(view.packets().len(), 3);

    assert_eq!(view.packet_bytes(0).unwrap(), &packet0[..]);
    assert_eq!(view.packet_bytes(1).unwrap(), &packet1[..]);
    assert_eq!(view.packet_bytes(2).unwrap(), &packet2[..]);
    assert!(view.packet_bytes(3).is_err());
}
