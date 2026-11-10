//! Adversarial property-based fuzzing and edge-case boundary violation tests for FlatMesh.

use flatasset_mesh::{
    FlatAttributeVertex, FlatMeshError, FlatMeshHeader, FlatMeshLumpEntry, FlatMeshView,
    FlatPackedNormal, FlatPositionVertex, FormatSignature, LUMP_FLAG_REQUIRED, LumpId,
};
use proptest::prelude::*;
use zerocopy::IntoBytes;

#[repr(C, align(64))]
struct AlignedStorage<const N: usize> {
    data: [u8; N],
}

fn build_valid_flatmesh() -> AlignedStorage<512> {
    let mut storage = AlignedStorage { data: [0u8; 512] };
    let buffer = &mut storage.data;

    let lump_mask =
        (1u64 << LumpId::PositionStream as u32) | (1u64 << LumpId::AttributeStream as u32);

    let header = FlatMeshHeader {
        signature: FormatSignature::FLATMESH_V1_0,
        lump_table_offset_div32: 2, // 64 bytes
        lump_count: 2,
        flags: FlatMeshHeader::FLAG_HAS_MESHLETS,
        lump_mask,
        bounding_sphere: [0.0, 0.0, 0.0, 10.0],
        aabb_min: [-10.0, -10.0, -10.0],
        aabb_max: [10.0, 10.0, 10.0],
    };
    buffer[..64].copy_from_slice(header.as_bytes());

    let pos_offset = 128u64;
    let pos_count = 4u32;
    let pos_bytes = (pos_count as usize) * core::mem::size_of::<FlatPositionVertex>(); // 64

    let attr_offset = 192u64;
    let attr_count = 4u32;
    let attr_bytes = (attr_count as usize) * core::mem::size_of::<FlatAttributeVertex>(); // 32

    let lumps = [
        FlatMeshLumpEntry {
            lump_id: LumpId::PositionStream as u16,
            flags: LUMP_FLAG_REQUIRED,
            version: 1,
            element_stride: 16,
            alignment_log2: 4, // 16 bytes
            offset: pos_offset,
            uncompressed_size: pos_bytes as u64,
            element_count: pos_count,
            crc32: 0,
        },
        FlatMeshLumpEntry {
            lump_id: LumpId::AttributeStream as u16,
            flags: LUMP_FLAG_REQUIRED,
            version: 1,
            element_stride: 8,
            alignment_log2: 3, // 8 bytes
            offset: attr_offset,
            uncompressed_size: attr_bytes as u64,
            element_count: attr_count,
            crc32: 0,
        },
    ];
    buffer[64..128].copy_from_slice(lumps.as_bytes());

    let positions = [
        FlatPositionVertex {
            position: [0.0, 0.0, 0.0],
            cluster_id: 0,
        },
        FlatPositionVertex {
            position: [1.0, 0.0, 0.0],
            cluster_id: 0,
        },
        FlatPositionVertex {
            position: [1.0, 1.0, 0.0],
            cluster_id: 0,
        },
        FlatPositionVertex {
            position: [0.0, 1.0, 0.0],
            cluster_id: 0,
        },
    ];
    buffer[pos_offset as usize..pos_offset as usize + pos_bytes]
        .copy_from_slice(positions.as_bytes());

    let norm = FlatPackedNormal::pack([0.0, 1.0, 0.0], 1.0);
    let attributes = [
        FlatAttributeVertex {
            normal_tangent: norm,
            uv: [0, 0],
        },
        FlatAttributeVertex {
            normal_tangent: norm,
            uv: [1000, 0],
        },
        FlatAttributeVertex {
            normal_tangent: norm,
            uv: [1000, 1000],
        },
        FlatAttributeVertex {
            normal_tangent: norm,
            uv: [0, 1000],
        },
    ];
    buffer[attr_offset as usize..attr_offset as usize + attr_bytes]
        .copy_from_slice(attributes.as_bytes());

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
        let _ = FlatMeshView::new(slice);
    }

    #[test]
    fn test_fuzz_mutated_valid_containers_never_panic(
        mutations in prop::collection::vec((0usize..512, any::<u8>()), 1..16)
    ) {
        let mut storage = build_valid_flatmesh();
        for (idx, val) in mutations {
            storage.data[idx] = val;
        }

        // View mounting must safely return Ok or Err, never panic
        let _ = FlatMeshView::new(&storage.data);
    }
}

// ---------------------------------------------------------------------------
// TARGETED ADVERSARIAL EDGE CASE TESTS
// ---------------------------------------------------------------------------

#[test]
fn test_edge_case_header_buffer_too_small() {
    let storage = AlignedStorage { data: [0u8; 64] };
    for len in [0, 1, 2, 7, 15, 16, 31, 63] {
        let slice = &storage.data[..len];
        let err = FlatMeshView::new(slice).unwrap_err();
        assert_eq!(
            err,
            FlatMeshError::BufferTooSmall {
                required: 64,
                actual: len,
            }
        );
    }
}

#[test]
fn test_edge_case_corrupted_signature() {
    let mut storage = build_valid_flatmesh();
    for byte_idx in 0..8 {
        let original = storage.data[byte_idx];
        storage.data[byte_idx] = original ^ 0xFF;

        let err = FlatMeshView::new(&storage.data).unwrap_err();
        assert!(matches!(err, FlatMeshError::InvalidSignature { .. }));

        // Restore byte
        storage.data[byte_idx] = original;
    }
}

#[test]
fn test_edge_case_version_boundaries() {
    let mut storage = build_valid_flatmesh();

    // Major version != 1 must fail
    for major in [0u8, 2, 3, 255] {
        let sig = FormatSignature::pack(*b"FLTMSH", major, 0);
        storage.data[..8].copy_from_slice(&sig.0.to_le_bytes());
        let err = FlatMeshView::new(&storage.data).unwrap_err();
        assert!(matches!(err, FlatMeshError::InvalidSignature { .. }));
    }

    // Minor version boundary (minor == 255) must fail
    for minor in [1u8, 128, 255] {
        let sig = FormatSignature::pack(*b"FLTMSH", 1, minor);
        storage.data[..8].copy_from_slice(&sig.0.to_le_bytes());
        let err = FlatMeshView::new(&storage.data).unwrap_err();
        assert!(matches!(err, FlatMeshError::InvalidSignature { .. }));
    }
}

#[test]
fn test_edge_case_u16_u32_u64_max_overflows() {
    let mut storage = build_valid_flatmesh();

    // 1. lump_table_offset_div32 overflow (u16::MAX * 32 = 2,097,120 bytes)
    let bad_table_offset = u16::MAX.to_le_bytes();
    storage.data[8..10].copy_from_slice(&bad_table_offset);
    let err = FlatMeshView::new(&storage.data).unwrap_err();
    assert!(matches!(err, FlatMeshError::OutOfBounds { .. }));

    // Reset table offset
    storage.data[8..10].copy_from_slice(&2u16.to_le_bytes());

    // 2. lump_count = u16::MAX with MAX * 32 = overflow
    storage.data[10..12].copy_from_slice(&u16::MAX.to_le_bytes());
    let err = FlatMeshView::new(&storage.data).unwrap_err();
    assert!(matches!(err, FlatMeshError::OutOfBounds { .. }));

    // Reset lump_count
    storage.data[10..12].copy_from_slice(&2u16.to_le_bytes());

    // 3. Lump offset = u64::MAX (offset in lump 0 is at 64 + 8 = 72)
    let max_u64 = u64::MAX.to_le_bytes();
    storage.data[72..80].copy_from_slice(&max_u64);
    let err = FlatMeshView::new(&storage.data).unwrap_err();
    assert!(matches!(err, FlatMeshError::OutOfBounds { .. }));

    // 4. Lump uncompressed_size = u64::MAX
    storage.data[72..80].copy_from_slice(&128u64.to_le_bytes());
    storage.data[80..88].copy_from_slice(&max_u64);
    let err = FlatMeshView::new(&storage.data).unwrap_err();
    assert!(matches!(err, FlatMeshError::OutOfBounds { .. }));

    // 5. Lump checked_add overflow: offset = u64::MAX - 10, size = 100
    let huge_offset = (u64::MAX - 10).to_le_bytes();
    storage.data[72..80].copy_from_slice(&huge_offset);
    storage.data[80..88].copy_from_slice(&100u64.to_le_bytes());
    let err = FlatMeshView::new(&storage.data).unwrap_err();
    assert!(matches!(err, FlatMeshError::OutOfBounds { .. }));
}

#[test]
fn test_edge_case_all_non_64_byte_alignments() {
    let storage = build_valid_flatmesh();
    let base_ptr = storage.data.as_ptr() as usize;
    assert_eq!(base_ptr % 64, 0, "Base buffer must be 64-byte aligned");

    // Slicing at any non-64-byte boundary (1..63 mod 64) must return Misaligned
    for shift in 1..64 {
        let slice = &storage.data[shift..];
        let err = FlatMeshView::new(slice).unwrap_err();
        let expected_misalign = (base_ptr + shift) % 64;
        assert_eq!(
            err,
            FlatMeshError::Misaligned {
                required: 64,
                actual: expected_misalign,
            }
        );
    }
}

#[test]
fn test_edge_case_circular_or_overlapping_lumps() {
    let mut storage = AlignedStorage { data: [0u8; 512] };
    let buffer = &mut storage.data;

    let header = FlatMeshHeader {
        signature: FormatSignature::FLATMESH_V1_0,
        lump_table_offset_div32: 2,
        lump_count: 2,
        flags: 0,
        lump_mask: (1 << LumpId::PositionStream as u32) | (1 << LumpId::AttributeStream as u32),
        bounding_sphere: [0.0; 4],
        aabb_min: [0.0; 3],
        aabb_max: [0.0; 3],
    };
    buffer[..64].copy_from_slice(header.as_bytes());

    // Two overlapping lumps inside valid memory range (128..256 and 160..288)
    let lumps = [
        FlatMeshLumpEntry {
            lump_id: LumpId::PositionStream as u16,
            flags: LUMP_FLAG_REQUIRED,
            version: 1,
            element_stride: 16,
            alignment_log2: 4,
            offset: 128,
            uncompressed_size: 128,
            element_count: 8,
            crc32: 0,
        },
        FlatMeshLumpEntry {
            lump_id: LumpId::AttributeStream as u16,
            flags: LUMP_FLAG_REQUIRED,
            version: 1,
            element_stride: 8,
            alignment_log2: 3,
            offset: 160, // Overlaps lump 0
            uncompressed_size: 128,
            element_count: 16,
            crc32: 0,
        },
    ];
    buffer[64..128].copy_from_slice(lumps.as_bytes());

    // Both lumps are within container memory [0..512], so FlatMeshView::new must safely succeed
    let view = FlatMeshView::new(buffer).expect("Overlapping in-bounds lumps must mount safely");
    assert_eq!(view.lumps().len(), 2);

    // Conflicting lump extending past container boundary (500 + 32 > 512)
    let bad_lump = FlatMeshLumpEntry {
        lump_id: LumpId::PositionStream as u16,
        flags: LUMP_FLAG_REQUIRED,
        version: 1,
        element_stride: 16,
        alignment_log2: 4,
        offset: 500,
        uncompressed_size: 32,
        element_count: 2,
        crc32: 0,
    };
    buffer[64..96].copy_from_slice(bad_lump.as_bytes());
    let err = FlatMeshView::new(buffer).unwrap_err();
    assert!(matches!(err, FlatMeshError::OutOfBounds { .. }));
}

#[test]
fn test_edge_case_nan_inf_bounding_sphere_and_positions() {
    let mut storage = build_valid_flatmesh();

    // Inject NaN, +Inf, -Inf, -0.0 into bounding sphere
    let exotic_floats = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0f32];
    let header_bytes = FlatMeshHeader {
        signature: FormatSignature::FLATMESH_V1_0,
        lump_table_offset_div32: 2,
        lump_count: 2,
        flags: FlatMeshHeader::FLAG_HAS_MESHLETS,
        lump_mask: (1 << LumpId::PositionStream as u32) | (1 << LumpId::AttributeStream as u32),
        bounding_sphere: exotic_floats,
        aabb_min: [-10.0; 3],
        aabb_max: [10.0; 3],
    };
    storage.data[..64].copy_from_slice(header_bytes.as_bytes());

    // Inject NaN and Inf into position stream
    let exotic_positions = [
        FlatPositionVertex {
            position: [f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            cluster_id: 0,
        },
        FlatPositionVertex {
            position: [1.0, 2.0, 3.0],
            cluster_id: 0,
        },
        FlatPositionVertex {
            position: [f32::MIN_POSITIVE, f32::MAX, -f32::MAX],
            cluster_id: 0,
        },
        FlatPositionVertex {
            position: [0.0, 0.0, 0.0],
            cluster_id: 0,
        },
    ];
    let pos_offset = 128;
    let pos_bytes = 4 * core::mem::size_of::<FlatPositionVertex>();
    storage.data[pos_offset..pos_offset + pos_bytes].copy_from_slice(exotic_positions.as_bytes());

    let view =
        FlatMeshView::new(&storage.data).expect("Exotic floating point values must mount safely");
    assert!(view.header().bounding_sphere[0].is_nan());
    assert!(view.header().bounding_sphere[1].is_infinite());
    assert!(view.header().bounding_sphere[2].is_sign_negative());

    let positions = view
        .positions()
        .expect("Must read positions containing NaN/Inf without panic");
    assert!(positions[0].position[0].is_nan());
    assert!(positions[0].position[1].is_infinite());
    assert_eq!(positions[1].position, [1.0, 2.0, 3.0]);
}

#[test]
fn test_edge_case_negative_strides_and_zero_element_lumps() {
    let mut storage = AlignedStorage { data: [0u8; 512] };
    let buffer = &mut storage.data;

    let header = FlatMeshHeader {
        signature: FormatSignature::FLATMESH_V1_0,
        lump_table_offset_div32: 2,
        lump_count: 2,
        flags: 0,
        lump_mask: (1 << LumpId::PositionStream as u32) | (1 << LumpId::AttributeStream as u32),
        bounding_sphere: [0.0; 4],
        aabb_min: [0.0; 3],
        aabb_max: [0.0; 3],
    };
    buffer[..64].copy_from_slice(header.as_bytes());

    let lumps = [
        FlatMeshLumpEntry {
            lump_id: LumpId::PositionStream as u16,
            flags: LUMP_FLAG_REQUIRED,
            version: 1,
            element_stride: 0,
            alignment_log2: 4, // 16 bytes
            offset: 128,
            uncompressed_size: 0,
            element_count: 0,
            crc32: 0,
        },
        FlatMeshLumpEntry {
            lump_id: LumpId::AttributeStream as u16,
            flags: LUMP_FLAG_REQUIRED,
            version: 1,
            element_stride: 255, // 0xFF
            alignment_log2: 3,   // 8 bytes
            offset: 144,
            uncompressed_size: 0,
            element_count: 0,
            crc32: 0,
        },
    ];
    buffer[64..128].copy_from_slice(lumps.as_bytes());

    let view = FlatMeshView::new(buffer).expect("Zero-element lumps must mount safely");
    let positions = view
        .positions()
        .expect("0-element position stream must return empty slice");
    assert_eq!(positions.len(), 0);

    let attributes = view
        .attributes()
        .expect("0-element attribute stream must return empty slice");
    assert_eq!(attributes.len(), 0);
}
