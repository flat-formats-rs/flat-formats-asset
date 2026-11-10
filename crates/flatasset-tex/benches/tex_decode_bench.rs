use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use flatasset_tex::{FlatMipDesc, FlatTexHeader, FlatTexView, FormatSignature, VkFormat};
use image::{ImageBuffer, Rgba};
use std::io::Cursor;
use std::time::Duration;
use zerocopy::IntoBytes;

#[repr(C, align(64))]
struct AlignedBuffer {
    data: Vec<u8>,
    offset: usize,
}

impl AlignedBuffer {
    fn new_zeroed(len: usize) -> Self {
        let data = vec![0u8; len + 64];
        let offset = (64 - (data.as_ptr() as usize % 64)) % 64;
        Self { data, offset }
    }

    fn slice(&self, len: usize) -> &[u8] {
        &self.data[self.offset..self.offset + len]
    }

    fn slice_mut(&mut self, len: usize) -> &mut [u8] {
        &mut self.data[self.offset..self.offset + len]
    }
}

const CONTAINER_PAYLOAD_OFFSET: usize = 512;
const CONTAINER_MIP_TABLE_OFFSET: usize = 32;

struct TextureDataset {
    name: &'static str,
    width: u32,
    height: u32,
    mip_levels: u8,
}

const TEXTURE_DATASETS: [TextureDataset; 3] = [
    TextureDataset {
        name: "256x256_mips9",
        width: 256,
        height: 256,
        mip_levels: 9,
    },
    TextureDataset {
        name: "1024x1024_mips11",
        width: 1024,
        height: 1024,
        mip_levels: 11,
    },
    TextureDataset {
        name: "2048x2048_mips12",
        width: 2048,
        height: 2048,
        mip_levels: 12,
    },
];

fn compute_mip_count(mut w: u32, mut h: u32) -> u8 {
    let mut count = 1u8;
    while w > 1 || h > 1 {
        w = (w >> 1).max(1);
        h = (h >> 1).max(1);
        count += 1;
    }
    count
}

fn create_test_flattex(width: u32, height: u32, mip_levels: u8) -> (AlignedBuffer, usize) {
    let header_size = core::mem::size_of::<FlatTexHeader>();
    let mip_table_size = (mip_levels as usize) * core::mem::size_of::<FlatMipDesc>();
    assert!(CONTAINER_MIP_TABLE_OFFSET + mip_table_size <= CONTAINER_PAYLOAD_OFFSET);

    // Calculate layout for all mips
    let mut mips = Vec::with_capacity(mip_levels as usize);
    let mut cur_offset = CONTAINER_PAYLOAD_OFFSET;

    for level in 0..mip_levels {
        let w = (width >> level).max(1);
        let h = (height >> level).max(1);
        let logical_pitch = (w as usize) * 4;
        let aligned_pitch = (logical_pitch + 255) & !255;
        let byte_size = aligned_pitch * (h as usize);

        mips.push(FlatMipDesc {
            offset: cur_offset as u64,
            byte_size: byte_size as u32,
            aligned_row_pitch_div256: (aligned_pitch / 256) as u16,
            rows_per_image: h as u16,
        });

        // Subresource offset aligned to 256
        cur_offset = (cur_offset + byte_size + 255) & !255;
    }

    let total_size = (cur_offset + 63) & !63;
    let mut buf = AlignedBuffer::new_zeroed(total_size);
    let slice = buf.slice_mut(total_size);

    let total_payload_bytes = cur_offset - CONTAINER_PAYLOAD_OFFSET;
    let total_payload_kb = (total_payload_bytes.div_ceil(1024)) as u32;

    let header = FlatTexHeader {
        signature: FormatSignature::FLATTEX_V1_0,
        total_payload_kb,
        vk_format: VkFormat::R8G8B8A8Srgb as u16,
        width: width as u16,
        height: height as u16,
        depth_or_layers: 1,
        mip_table_offset_div32: (CONTAINER_MIP_TABLE_OFFSET / 32) as u16,
        payload_offset_div256: (CONTAINER_PAYLOAD_OFFSET / 256) as u16,
        mip_levels,
        flags: FlatTexHeader::FLAG_SRGB,
        _reserved: [0; 6],
    };
    slice[..header_size].copy_from_slice(header.as_bytes());

    let mip_table_end = CONTAINER_MIP_TABLE_OFFSET + mip_table_size;
    slice[CONTAINER_MIP_TABLE_OFFSET..mip_table_end].copy_from_slice(mips.as_bytes());

    // Populate mip data with non-zero bytes to avoid zero-page OS optimizations
    for mip in &mips {
        let start = mip.offset as usize;
        let end = start + mip.byte_size as usize;
        slice[start..end].fill(0xAA);
    }

    (buf, total_size)
}

fn create_test_png(width: u32, height: u32) -> Vec<u8> {
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_fn(width, height, |x, y| {
        Rgba([
            (x & 0xFF) as u8,
            (y & 0xFF) as u8,
            ((x + y) & 0xFF) as u8,
            255,
        ])
    });
    let mut png_bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .expect("write to png");
    png_bytes
}

fn create_unpadded_rgba(width: u32, height: u32) -> Vec<u8> {
    let size = (width * height * 4) as usize;
    vec![0xAA; size]
}

fn bench_cold_mount_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cold_Mount_Latency");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(15);

    for ds in &TEXTURE_DATASETS {
        assert_eq!(compute_mip_count(ds.width, ds.height), ds.mip_levels);

        let (flattex_buf, total_size) = create_test_flattex(ds.width, ds.height, ds.mip_levels);
        let flattex_slice = flattex_buf.slice(total_size);
        let png_bytes = create_test_png(ds.width, ds.height);

        // 1. FlatTexView zero-alloc mount
        group.bench_with_input(
            BenchmarkId::new("FlatTexView_mount_zero_alloc", ds.name),
            &flattex_slice,
            |b, &slice| {
                b.iter(|| {
                    let view = FlatTexView::new(black_box(slice)).unwrap();
                    black_box(view.header().signature);
                });
            },
        );

        // 2. image crate PNG parse
        group.bench_with_input(
            BenchmarkId::new("image_crate_png_decode", ds.name),
            &png_bytes,
            |b, bytes| {
                b.iter(|| {
                    let img = image::load_from_memory_with_format(
                        black_box(bytes),
                        image::ImageFormat::Png,
                    )
                    .unwrap();
                    let rgba = img.to_rgba8();
                    black_box(rgba.len());
                });
            },
        );
    }

    group.finish();
}

fn bench_staging_dma_row_pitch_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Staging_DMA_Row_Pitch_Extraction");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);

    for ds in &TEXTURE_DATASETS {
        let (flattex_buf, total_size) = create_test_flattex(ds.width, ds.height, ds.mip_levels);
        let flattex_slice = flattex_buf.slice(total_size);
        let view = FlatTexView::new(flattex_slice).unwrap();

        let unpadded_rgba = create_unpadded_rgba(ds.width, ds.height);

        let logical_pitch = (ds.width as usize) * 4;
        let aligned_pitch = (logical_pitch + 255) & !255;
        let staging_byte_size = aligned_pitch * (ds.height as usize);

        let mut gpu_staging_buffer = vec![0u8; staging_byte_size];

        group.throughput(Throughput::Bytes(staging_byte_size as u64));

        // 1. FlatTex: Direct slice fetch (zero host-side allocations, zero-copy pointer slice)
        group.bench_with_input(
            BenchmarkId::new("FlatTex_PrePadded_Direct_Slice", ds.name),
            &view,
            |b, v| {
                b.iter(|| {
                    let dma_slice = v.mip_bytes(black_box(0)).unwrap();
                    black_box(dma_slice);
                });
            },
        );

        // 2. FlatTex: Pre-padded single contiguous DMA buffer copy
        group.bench_with_input(
            BenchmarkId::new("FlatTex_PrePadded_Contiguous_DMA_Copy", ds.name),
            &view,
            |b, v| {
                let dma_slice = v.mip_bytes(0).unwrap();
                b.iter(|| {
                    gpu_staging_buffer[..dma_slice.len()].copy_from_slice(black_box(dma_slice));
                    black_box(&gpu_staging_buffer[0]);
                });
            },
        );

        // 3. Unpadded: CPU Row-by-Row Staging Repack
        group.bench_with_input(
            BenchmarkId::new("Unpadded_CPU_Row_By_Row_Repack", ds.name),
            &unpadded_rgba,
            |b, unpadded| {
                let height = ds.height as usize;
                b.iter(|| {
                    for row in 0..height {
                        let src_start = row * logical_pitch;
                        let dst_start = row * aligned_pitch;
                        gpu_staging_buffer[dst_start..dst_start + logical_pitch]
                            .copy_from_slice(&unpadded[src_start..src_start + logical_pitch]);
                    }
                    black_box(&gpu_staging_buffer[0]);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cold_mount_latency,
    bench_staging_dma_row_pitch_extraction,
);
criterion_main!(benches);
