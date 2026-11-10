//! Adversarial property-based fuzzing and edge-case boundary violation tests for FlatTex.

use flatasset_tex::{
    FlatMipDesc, FlatTexError, FlatTexHeader, FlatTexView, FormatSignature, VkFormat,
};
use proptest::prelude::*;
use zerocopy::IntoBytes;

#[repr(C, align(64))]
struct AlignedStorage<const N: usize> {
    data: [u8; N],
}

fn build_valid_flattex() -> AlignedStorage<8192> {
    let mut storage = AlignedStorage { data: [0u8; 8192] };
    let buffer = &mut storage.data;

    let header_size = core::mem::size_of::<FlatTexHeader>();
    let mip_table_offset = 32;
    let mip_count = 2;
    let mip_table_size = mip_count * core::mem::size_of::<FlatMipDesc>();

    let payload_offset = 512u64;
    let mip0_size = 4096u32;
    let mip0_pitch = 256u32;

    let mip1_offset = payload_offset + mip0_size as u64;
    let mip1_size = 2048u32;
    let mip1_pitch = 256u32;

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

    storage
}

// ---------------------------------------------------------------------------
// 10,000 PROPEST CASES: Arbitrary Byte Payloads & Random Container Mutations
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn test_fuzz_arbitrary_byte_payloads_never_panic(payload in prop::collection::vec(any::<u8>(), 0..1024)) {
        let mut storage = AlignedStorage { data: [0u8; 1024] };
        let slice = &mut storage.data[..payload.len()];
        slice.copy_from_slice(&payload);

        // Reader MUST safely return Result::Err and never panic or trigger undefined behavior
        let _ = FlatTexView::new(slice);
    }

    #[test]
    fn test_fuzz_mutated_valid_containers_never_panic(
        mutations in prop::collection::vec((0usize..1024, any::<u8>()), 1..16)
    ) {
        let mut storage = build_valid_flattex();
        for (idx, val) in mutations {
            storage.data[idx] = val;
        }

        // View mounting must safely return Ok or Err, never panic
        let _ = FlatTexView::new(&storage.data);
    }
}

// ---------------------------------------------------------------------------
// TARGETED ADVERSARIAL EDGE CASE TESTS
// ---------------------------------------------------------------------------

#[test]
fn test_edge_case_header_buffer_too_small() {
    let storage = AlignedStorage { data: [0u8; 64] };
    for len in [0, 1, 2, 7, 15, 16, 31] {
        let slice = &storage.data[..len];
        let err = FlatTexView::new(slice).unwrap_err();
        assert_eq!(
            err,
            FlatTexError::BufferTooSmall {
                required: 32,
                actual: len,
            }
        );
    }
}

#[test]
fn test_edge_case_corrupted_signature() {
    let mut storage = build_valid_flattex();
    for byte_idx in 0..8 {
        let original = storage.data[byte_idx];
        storage.data[byte_idx] = original ^ 0xFF;

        let err = FlatTexView::new(&storage.data).unwrap_err();
        assert!(matches!(err, FlatTexError::InvalidSignature { .. }));

        // Restore byte
        storage.data[byte_idx] = original;
    }
}

#[test]
fn test_edge_case_version_boundaries() {
    let mut storage = build_valid_flattex();

    // Major version != 1 must fail
    for major in [0u8, 2, 3, 255] {
        let sig = FormatSignature::pack(*b"FLTTEX", major, 0);
        storage.data[..8].copy_from_slice(&sig.0.to_le_bytes());
        let err = FlatTexView::new(&storage.data).unwrap_err();
        assert!(matches!(err, FlatTexError::InvalidSignature { .. }));
    }

    // Minor version boundary (minor == 255) must fail
    for minor in [1u8, 128, 255] {
        let sig = FormatSignature::pack(*b"FLTTEX", 1, minor);
        storage.data[..8].copy_from_slice(&sig.0.to_le_bytes());
        let err = FlatTexView::new(&storage.data).unwrap_err();
        assert!(matches!(err, FlatTexError::InvalidSignature { .. }));
    }
}

#[test]
fn test_edge_case_u16_u32_u64_max_overflows() {
    let mut storage = build_valid_flattex();

    // 1. mip_table_offset_div32 overflow (u16::MAX * 32 = 2,097,120 bytes)
    let bad_table_offset = u16::MAX.to_le_bytes();
    storage.data[20..22].copy_from_slice(&bad_table_offset);
    let err = FlatTexView::new(&storage.data).unwrap_err();
    assert!(matches!(err, FlatTexError::OutOfBounds { .. }));

    // Reset table offset
    storage.data[20..22].copy_from_slice(&1u16.to_le_bytes());

    // 2. mip_levels = u8::MAX (255) with 255 * 16 = 4080 bytes
    // Put table close to end so it overflows
    storage.data[20..22].copy_from_slice(&200u16.to_le_bytes()); // 6400 bytes
    storage.data[24] = 255; // 6400 + 4080 = 10480 > 8192
    let err = FlatTexView::new(&storage.data).unwrap_err();
    assert!(matches!(err, FlatTexError::OutOfBounds { .. }));

    // Reset table offset and mip_levels
    storage.data[20..22].copy_from_slice(&1u16.to_le_bytes());
    storage.data[24] = 2;

    // 3. Mip offset = u64::MAX
    let max_u64 = u64::MAX.to_le_bytes();
    // Offset in mip 0 is at offset 32 + 0 = 32
    storage.data[32..40].copy_from_slice(&max_u64);
    let err = FlatTexView::new(&storage.data).unwrap_err();
    assert!(matches!(err, FlatTexError::OutOfBounds { .. }));

    // 4. Mip byte_size = u32::MAX
    storage.data[32..40].copy_from_slice(&512u64.to_le_bytes());
    storage.data[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
    let err = FlatTexView::new(&storage.data).unwrap_err();
    assert!(matches!(err, FlatTexError::OutOfBounds { .. }));

    // 5. Mip checked_add overflow: offset = u64::MAX - 10, byte_size = 100
    let huge_offset = (u64::MAX - 10).to_le_bytes();
    storage.data[32..40].copy_from_slice(&huge_offset);
    storage.data[40..44].copy_from_slice(&100u32.to_le_bytes());
    let err = FlatTexView::new(&storage.data).unwrap_err();
    assert!(matches!(err, FlatTexError::OutOfBounds { .. }));

    // 6. Max row pitch div 256 = u16::MAX
    storage.data[32..40].copy_from_slice(&512u64.to_le_bytes());
    storage.data[40..44].copy_from_slice(&4096u32.to_le_bytes());
    storage.data[44..46].copy_from_slice(&u16::MAX.to_le_bytes());
    let view = FlatTexView::new(&storage.data).expect("Row pitch u16::MAX must mount safely");
    assert_eq!(view.mips()[0].aligned_row_pitch(), 65535 * 256);
}

#[test]
fn test_edge_case_all_non_32_byte_alignments() {
    let storage = build_valid_flattex();
    let base_ptr = storage.data.as_ptr() as usize;
    assert_eq!(base_ptr % 32, 0, "Base buffer must be 32-byte aligned");

    // Slicing at any non-32-byte boundary (1..31 mod 32) must return Misaligned
    for shift in 1..32 {
        let slice = &storage.data[shift..];
        let err = FlatTexView::new(slice).unwrap_err();
        let expected_misalign = (base_ptr + shift) % 32;
        assert_eq!(
            err,
            FlatTexError::Misaligned {
                required: 32,
                actual: expected_misalign,
            }
        );
    }
}

#[test]
fn test_edge_case_circular_or_overlapping_mips() {
    let mut storage = AlignedStorage { data: [0u8; 8192] };
    let buffer = &mut storage.data;

    let header = FlatTexHeader {
        signature: FormatSignature::FLATTEX_V1_0,
        total_payload_kb: 8,
        vk_format: VkFormat::Bc1RgbSrgbBlock as u16,
        width: 64,
        height: 64,
        depth_or_layers: 1,
        mip_table_offset_div32: 1,
        payload_offset_div256: 2,
        mip_levels: 2,
        flags: 0,
        _reserved: [0; 6],
    };
    buffer[..32].copy_from_slice(header.as_bytes());

    // Two overlapping mips within buffer memory (512..1536 and 768..1792)
    let mips = [
        FlatMipDesc {
            offset: 512,
            byte_size: 1024,
            aligned_row_pitch_div256: 1,
            rows_per_image: 4,
        },
        FlatMipDesc {
            offset: 768, // Overlapping mip 0
            byte_size: 1024,
            aligned_row_pitch_div256: 1,
            rows_per_image: 4,
        },
    ];
    buffer[32..64].copy_from_slice(mips.as_bytes());

    let view = FlatTexView::new(buffer).expect("Overlapping in-bounds mips must mount safely");
    assert_eq!(view.mips().len(), 2);
    let s0 = view.mip_bytes(0).unwrap();
    let s1 = view.mip_bytes(1).unwrap();
    assert_eq!(s0.len(), 1024);
    assert_eq!(s1.len(), 1024);

    // Conflicting mip extending past container boundary (8000 + 512 > 8192)
    let bad_mips = [FlatMipDesc {
        offset: 8000,
        byte_size: 512,
        aligned_row_pitch_div256: 1,
        rows_per_image: 2,
    }];
    buffer[32..48].copy_from_slice(bad_mips.as_bytes());
    buffer[24] = 1; // 1 mip level
    let err = FlatTexView::new(buffer).unwrap_err();
    assert!(matches!(err, FlatTexError::OutOfBounds { .. }));
}

#[test]
fn test_edge_case_extreme_dimensions() {
    let mut storage = build_valid_flattex();

    // Test zero dimensions
    let header_zeros = FlatTexHeader {
        signature: FormatSignature::FLATTEX_V1_0,
        total_payload_kb: 0,
        vk_format: VkFormat::R8G8B8A8Unorm as u16,
        width: 0,
        height: 0,
        depth_or_layers: 0,
        mip_table_offset_div32: 1,
        payload_offset_div256: 2,
        mip_levels: 1,
        flags: 0,
        _reserved: [0; 6],
    };
    storage.data[..32].copy_from_slice(header_zeros.as_bytes());
    let view = FlatTexView::new(&storage.data).expect("Zero dimensions must mount safely");
    assert_eq!(view.header().width, 0);
    assert_eq!(view.header().height, 0);

    // Test max dimensions
    let header_max = FlatTexHeader {
        signature: FormatSignature::FLATTEX_V1_0,
        total_payload_kb: u32::MAX,
        vk_format: VkFormat::R8G8B8A8Unorm as u16,
        width: u16::MAX,
        height: u16::MAX,
        depth_or_layers: u16::MAX,
        mip_table_offset_div32: 1,
        payload_offset_div256: 2,
        mip_levels: 1,
        flags: 0xFF,
        _reserved: [0xFF; 6],
    };
    storage.data[..32].copy_from_slice(header_max.as_bytes());
    let view2 = FlatTexView::new(&storage.data).expect("Max dimensions must mount safely");
    assert_eq!(view2.header().width, u16::MAX);
    assert_eq!(view2.header().total_payload_kb, u32::MAX);
}

#[test]
fn test_edge_case_negative_strides_and_zero_element_mips() {
    let mut storage = AlignedStorage { data: [0u8; 1024] };
    let buffer = &mut storage.data;

    let header = FlatTexHeader {
        signature: FormatSignature::FLATTEX_V1_0,
        total_payload_kb: 0,
        vk_format: VkFormat::R8G8B8A8Unorm as u16,
        width: 0,
        height: 0,
        depth_or_layers: 0,
        mip_table_offset_div32: 1,
        payload_offset_div256: 2,
        mip_levels: 0, // 0 mip levels
        flags: 0,
        _reserved: [0; 6],
    };
    buffer[..32].copy_from_slice(header.as_bytes());

    let view = FlatTexView::new(buffer).expect("0 mip levels must mount safely");
    assert_eq!(view.mips().len(), 0);
    let err = view.mip_bytes(0).unwrap_err();
    assert!(matches!(
        err,
        FlatTexError::MipOutOfBounds {
            requested: 0,
            max: 0
        }
    ));

    // Now test 1 mip level with 0 byte_size, 0 rows, 0 row pitch
    buffer[24] = 1;
    let mip_zero = FlatMipDesc {
        offset: 512,
        byte_size: 0,
        aligned_row_pitch_div256: 0,
        rows_per_image: 0,
    };
    buffer[32..48].copy_from_slice(mip_zero.as_bytes());

    let view_zero = FlatTexView::new(buffer).expect("0-sized mip must mount safely");
    let sub = view_zero
        .mip_bytes(0)
        .expect("Must return empty slice for 0 byte_size");
    assert_eq!(sub.len(), 0);
}
