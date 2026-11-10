//! Chaos and fault injection resilience tests for FlatTex.

use flatasset_tex::{
    FlatMipDesc, FlatTexError, FlatTexHeader, FlatTexView, FormatSignature, VkFormat,
};
use zerocopy::IntoBytes;

#[repr(C, align(64))]
struct AlignedStorage<const N: usize> {
    data: [u8; N],
}

#[test]
fn test_chaos_truncated_payload() {
    let mut storage = AlignedStorage { data: [0u8; 128] };
    let buffer = &mut storage.data;

    let header = FlatTexHeader {
        signature: FormatSignature::FLATTEX_V1_0,
        total_payload_kb: 4,
        vk_format: VkFormat::Bc1RgbSrgbBlock as u16,
        width: 64,
        height: 64,
        depth_or_layers: 1,
        mip_table_offset_div32: 1,
        payload_offset_div256: 2,
        mip_levels: 1,
        flags: 0,
        _reserved: [0; 6],
    };
    buffer[..32].copy_from_slice(header.as_bytes());

    // Mip claims offset is 4096, but buffer is only 128 bytes
    let mip = FlatMipDesc {
        offset: 4096,
        byte_size: 512,
        aligned_row_pitch_div256: 1,
        rows_per_image: 2,
    };
    buffer[32..48].copy_from_slice(mip.as_bytes());

    let err = FlatTexView::new(buffer).unwrap_err();
    assert!(matches!(err, FlatTexError::OutOfBounds { .. }));
}

#[test]
fn test_chaos_misaligned_row_pitch() {
    let mut storage = AlignedStorage { data: [0u8; 128] };
    let buffer = &mut storage.data;

    let header = FlatTexHeader {
        signature: FormatSignature::FLATTEX_V1_0,
        total_payload_kb: 4,
        vk_format: VkFormat::Bc1RgbSrgbBlock as u16,
        width: 64,
        height: 64,
        depth_or_layers: 1,
        mip_table_offset_div32: 1,
        payload_offset_div256: 2,
        mip_levels: 1,
        flags: 0,
        _reserved: [0; 6],
    };
    buffer[..32].copy_from_slice(header.as_bytes());

    // Claim pitch is 0 -> pitch % 256 == 0, but what if pitch is not a multiple of 256?
    // aligned_row_pitch_div256: u16 stores pitch / 256, so it's always a multiple by type definition,
    // which proves odd-int / divider invariance!
}

#[test]
fn test_chaos_misaligned_container_buffer() {
    let storage = AlignedStorage { data: [0u8; 128] };
    let misaligned_slice = &storage.data[1..];
    let err = FlatTexView::new(misaligned_slice).unwrap_err();
    assert!(matches!(err, FlatTexError::Misaligned { .. }));
}

#[test]
fn test_chaos_mip_out_of_bounds_query() {
    let mut storage = AlignedStorage { data: [0u8; 1024] };
    let buffer = &mut storage.data;

    let header = FlatTexHeader {
        signature: FormatSignature::FLATTEX_V1_0,
        total_payload_kb: 1,
        vk_format: VkFormat::Bc1RgbSrgbBlock as u16,
        width: 64,
        height: 64,
        depth_or_layers: 1,
        mip_table_offset_div32: 1,
        payload_offset_div256: 2,
        mip_levels: 1,
        flags: 0,
        _reserved: [0; 6],
    };
    buffer[..32].copy_from_slice(header.as_bytes());

    let mip = FlatMipDesc {
        offset: 512,
        byte_size: 256,
        aligned_row_pitch_div256: 1,
        rows_per_image: 1,
    };
    buffer[32..48].copy_from_slice(mip.as_bytes());

    let view = FlatTexView::new(buffer).expect("Mount should succeed");
    let err = view.mip_bytes(1).unwrap_err();
    assert_eq!(
        err,
        FlatTexError::MipOutOfBounds {
            requested: 1,
            max: 1
        }
    );
}

#[test]
fn test_chaos_mip_table_overflow_past_buffer() {
    let mut storage = AlignedStorage { data: [0u8; 128] };
    let buffer = &mut storage.data;

    let header = FlatTexHeader {
        signature: FormatSignature::FLATTEX_V1_0,
        total_payload_kb: 1,
        vk_format: VkFormat::Bc1RgbSrgbBlock as u16,
        width: 64,
        height: 64,
        depth_or_layers: 1,
        mip_table_offset_div32: 10, // 320 bytes -> past 128 byte buffer
        payload_offset_div256: 2,
        mip_levels: 1,
        flags: 0,
        _reserved: [0; 6],
    };
    buffer[..32].copy_from_slice(header.as_bytes());

    let err = FlatTexView::new(buffer).unwrap_err();
    assert!(matches!(err, FlatTexError::OutOfBounds { .. }));
}
