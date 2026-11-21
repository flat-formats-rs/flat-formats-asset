use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flatasset_audio::{AudioCodecId, FlatAudioBuilder, FlatAudioView};

fn generate_sample_fls() -> Vec<u8> {
    let mut builder = FlatAudioBuilder::new(AudioCodecId::PCM16, 2, 44100).with_loop(1024, 44100);

    // 100 packets simulating dynamic voice / spatial audio stream
    for _ in 0..100 {
        let samples = 441; // 10ms frame
        let dummy_pcm = vec![0u8; (samples * 4) as usize]; // 16-bit stereo = 4 bytes/sample
        builder.add_packet(samples, dummy_pcm);
    }

    builder.build().expect("build flat audio")
}

fn align_to_32(bytes: &[u8]) -> (Vec<u8>, usize) {
    let mut aligned = Vec::with_capacity(bytes.len() + 32);
    let offset = (32 - (aligned.as_ptr() as usize % 32)) % 32;
    aligned.resize(offset, 0);
    aligned.extend_from_slice(bytes);
    (aligned, offset)
}

fn bench_audio_mount(c: &mut Criterion) {
    let raw_bytes = generate_sample_fls();
    let (aligned_buf, offset) = align_to_32(&raw_bytes);
    let fls_bytes = &aligned_buf[offset..offset + raw_bytes.len()];

    c.bench_function("fls_mount_zero_alloc", |b| {
        b.iter(|| {
            let view = FlatAudioView::new(black_box(fls_bytes)).unwrap();
            black_box(view.header().sample_rate);
            black_box(view.header().packet_count);
        })
    });

    let view = FlatAudioView::new(fls_bytes).unwrap();
    c.bench_function("fls_random_packet_lookup_100x", |b| {
        b.iter(|| {
            for i in 0..100 {
                let packet_slice = view.packet_bytes(black_box(i)).unwrap();
                black_box(packet_slice.len());
            }
        })
    });
}

criterion_group!(benches, bench_audio_mount);
criterion_main!(benches);
