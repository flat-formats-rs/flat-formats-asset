use flatasset_tex::{FlatMipDesc, FlatTexHeader, FlatTexView, FormatSignature, VkFormat};
use zerocopy::IntoBytes;

#[repr(C, align(64))]
struct AlignedStorage<const N: usize> {
    data: [u8; N],
}

#[test]
fn test_zero_alloc_flattex_view_valid() {
    let header_size = core::mem::size_of::<FlatTexHeader>();
    let mip_table_offset = 32;
    let mip_count = 2;
    let mip_table_size = mip_count * core::mem::size_of::<FlatMipDesc>();

    // 512-byte aligned payload offset (D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT)
    let payload_offset = 512u64;

    // Mip 0: 64x64 BC1 (16x16 blocks). Logical pitch: 16 * 8 = 128B.
    // Pre-padded to multiple of 256B -> aligned_row_pitch = 256B.
    // Rows = 16. Total byte_size = 256 * 16 = 4096B.
    let mip0_size = 4096u32;
    let mip0_pitch = 256u32;

    // Mip 1: 32x32 BC1 (8x8 blocks). Logical pitch: 8 * 8 = 64B.
    // Pre-padded to multiple of 256B -> aligned_row_pitch = 256B.
    // Rows = 8. Total byte_size = 256 * 8 = 2048B.
    let mip1_offset = payload_offset + mip0_size as u64;
    let mip1_size = 2048u32;
    let mip1_pitch = 256u32;

    let mut storage = AlignedStorage { data: [0u8; 8192] };
    let buffer = &mut storage.data;

    assert_eq!(buffer.as_ptr() as usize % 32, 0);

    let header = FlatTexHeader {
        signature: FormatSignature::FLATTEX_V1_0,
        total_payload_kb: 8,
        vk_format: VkFormat::Bc1RgbSrgbBlock as u16,
        width: 64,
        height: 64,
        depth_or_layers: 1,
        mip_table_offset_div32: 1, // 32 bytes
        payload_offset_div256: 2,  // 512 bytes
        mip_levels: mip_count as u8,
        flags: FlatTexHeader::FLAG_SRGB,
        _reserved: [0; 6],
    };
    buffer[..header_size].copy_from_slice(header.as_bytes());

    let mips = [
        FlatMipDesc {
            offset: payload_offset,
            byte_size: mip0_size,
            aligned_row_pitch_div256: (mip0_pitch / 256) as u16,
            rows_per_image: 16,
        },
        FlatMipDesc {
            offset: mip1_offset,
            byte_size: mip1_size,
            aligned_row_pitch_div256: (mip1_pitch / 256) as u16,
            rows_per_image: 8,
        },
    ];

    let mip_dest = &mut buffer[mip_table_offset..mip_table_offset + mip_table_size];
    mip_dest.copy_from_slice(mips.as_bytes());

    // Mount FlatTexView - Zero heap allocations!
    let view = FlatTexView::new(buffer).expect("Valid container must mount without error");
    assert_eq!(view.header().signature, FormatSignature::FLATTEX_V1_0);
    assert_eq!(view.header().width, 64);
    assert_eq!(view.header().height, 64);
    assert_eq!(view.header().mip_levels, 2);
    assert!(view.header().is_srgb());

    // Verify subresource slices are directly compatible with Queue::write_texture
    let sub0 = view.mip_bytes(0).expect("Must read mip 0");
    assert_eq!(sub0.len(), 4096);
    assert_eq!(view.mips()[0].aligned_row_pitch(), 256);

    let sub1 = view.mip_bytes(1).expect("Must read mip 1");
    assert_eq!(sub1.len(), 2048);
    assert_eq!(view.mips()[1].aligned_row_pitch(), 256);
}
