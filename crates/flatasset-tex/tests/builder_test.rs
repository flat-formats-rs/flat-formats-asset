use flatasset_tex::{encode_rgba_to_bc1_dma, FlatTexBuilder, FlatTexView, MipInput, VkFormat};

#[test]
fn test_bc1_encoding_and_flattex_builder_roundtrip() {
    let width = 64u32;
    let height = 64u32;

    // Generate a synthetic test gradient
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            rgba[idx] = (x * 255 / width) as u8;
            rgba[idx + 1] = (y * 255 / height) as u8;
            rgba[idx + 2] = 128;
            rgba[idx + 3] = 255;
        }
    }

    let (dma_bytes, aligned_pitch, rows) = encode_rgba_to_bc1_dma(width, height, &rgba);
    assert_eq!(aligned_pitch % 256, 0);

    let mut builder =
        FlatTexBuilder::new_2d(width as u16, height as u16, VkFormat::Bc1RgbUnormBlock);
    builder.add_mip(MipInput {
        width,
        height,
        dma_bytes,
        aligned_row_pitch: aligned_pitch,
        rows_per_image: rows,
    });

    let container_bytes = builder.build().expect("build flattex");

    // Standard FlatTex containers require 32-byte alignment when mounted from memory
    // (e.g. from mmap or aligned buffer). Wrap in 64-byte aligned storage for test.
    let mut aligned_vec = Vec::with_capacity(container_bytes.len() + 64);
    let offset = (64 - (aligned_vec.as_ptr() as usize % 64)) % 64;
    aligned_vec.resize(offset, 0);
    aligned_vec.extend_from_slice(&container_bytes);
    let aligned_slice = &aligned_vec[offset..offset + container_bytes.len()];
    assert_eq!(aligned_slice.as_ptr() as usize % 32, 0);

    let view = FlatTexView::new(aligned_slice).expect("mount flattex view");

    assert_eq!(view.header().width, 64);
    assert_eq!(view.header().height, 64);
    assert_eq!(view.header().mip_levels, 1);
    assert_eq!(view.header().vk_format, VkFormat::Bc1RgbUnormBlock as u16);

    let mip0 = view.mip_bytes(0).expect("read mip 0");
    assert!(!mip0.is_empty());
    assert_eq!(view.mips()[0].aligned_row_pitch() % 256, 0);
}
