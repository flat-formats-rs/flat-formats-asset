//! Chaos and fault injection resilience tests for FlatMesh.

use flatasset_mesh::{
    FlatMeshError, FlatMeshHeader, FlatMeshLumpEntry, FlatMeshView, FormatSignature, LumpId,
    LUMP_FLAG_REQUIRED,
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

    let header = FlatMeshHeader {
        signature: FormatSignature::FLATMESH_V1_0,
        lump_table_offset_div32: 2,
        lump_count: 1,
        flags: 0,
        lump_mask: 1 << (LumpId::PositionStream as u32),
        bounding_sphere: [0.0; 4],
        aabb_min: [0.0; 3],
        aabb_max: [0.0; 3],
    };
    buffer[..64].copy_from_slice(header.as_bytes());

    // Claim lump offset is 1024, but buffer is only 128 bytes
    let lump = FlatMeshLumpEntry {
        lump_id: LumpId::PositionStream as u16,
        flags: LUMP_FLAG_REQUIRED,
        version: 1,
        element_stride: 16,
        alignment_log2: 4,
        offset: 1024,
        uncompressed_size: 64,
        element_count: 4,
        crc32: 0,
    };
    buffer[64..96].copy_from_slice(lump.as_bytes());

    let err = FlatMeshView::new(buffer).unwrap_err();
    assert!(matches!(err, FlatMeshError::OutOfBounds { .. }));
}

#[test]
fn test_chaos_misaligned_buffer() {
    let storage = AlignedStorage { data: [0u8; 128] };
    // Pass slice shifted by 1 byte -> misaligned to 64 bytes
    let misaligned_slice = &storage.data[1..];

    let err = FlatMeshView::new(misaligned_slice).unwrap_err();
    assert!(matches!(err, FlatMeshError::Misaligned { .. }));
}

#[test]
fn test_chaos_missing_required_stream() {
    let mut storage = AlignedStorage { data: [0u8; 128] };
    let buffer = &mut storage.data;

    let header = FlatMeshHeader {
        signature: FormatSignature::FLATMESH_V1_0,
        lump_table_offset_div32: 2,
        lump_count: 1,
        flags: 0,
        lump_mask: 1 << (LumpId::AttributeStream as u32),
        bounding_sphere: [0.0; 4],
        aabb_min: [0.0; 3],
        aabb_max: [0.0; 3],
    };
    buffer[..64].copy_from_slice(header.as_bytes());

    // Lump is only AttributeStream, no PositionStream
    let lump = FlatMeshLumpEntry {
        lump_id: LumpId::AttributeStream as u16,
        flags: 0,
        version: 1,
        element_stride: 8,
        alignment_log2: 3,
        offset: 96,
        uncompressed_size: 32,
        element_count: 4,
        crc32: 0,
    };
    buffer[64..96].copy_from_slice(lump.as_bytes());

    let view = FlatMeshView::new(buffer).expect("Mount should succeed");
    let err = view.positions().unwrap_err();
    assert_eq!(
        err,
        FlatMeshError::MissingRequiredLump(LumpId::PositionStream as u16)
    );
}

#[test]
fn test_chaos_lump_misaligned_offset() {
    let mut storage = AlignedStorage { data: [0u8; 256] };
    let buffer = &mut storage.data;

    let header = FlatMeshHeader {
        signature: FormatSignature::FLATMESH_V1_0,
        lump_table_offset_div32: 2,
        lump_count: 1,
        flags: 0,
        lump_mask: 1 << (LumpId::PositionStream as u32),
        bounding_sphere: [0.0; 4],
        aabb_min: [0.0; 3],
        aabb_max: [0.0; 3],
    };
    buffer[..64].copy_from_slice(header.as_bytes());

    // Position stream requiring 16B alignment (alignment_log2 = 4), but placed at offset 65
    let lump = FlatMeshLumpEntry {
        lump_id: LumpId::PositionStream as u16,
        flags: LUMP_FLAG_REQUIRED,
        version: 1,
        element_stride: 16,
        alignment_log2: 4, // 16 bytes
        offset: 65,        // Misaligned
        uncompressed_size: 64,
        element_count: 4,
        crc32: 0,
    };
    buffer[64..96].copy_from_slice(lump.as_bytes());

    let view = FlatMeshView::new(buffer).expect("Container view mounts");
    let err = view.positions().unwrap_err();
    assert!(matches!(
        err,
        FlatMeshError::Misaligned { required: 16, .. }
    ));
}
