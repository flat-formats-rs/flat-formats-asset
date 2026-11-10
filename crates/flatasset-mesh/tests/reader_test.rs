use flatasset_mesh::{
    FlatAttributeVertex, FlatMeshHeader, FlatMeshLumpEntry, FlatMeshView, FlatPackedNormal,
    FlatPositionVertex, FormatSignature, LUMP_FLAG_REQUIRED, LumpId,
};
use zerocopy::IntoBytes;

#[repr(C, align(64))]
struct AlignedStorage<const N: usize> {
    data: [u8; N],
}

#[test]
fn test_zero_alloc_flatmesh_view_valid() {
    // Construct a synthetic in-memory container matching 64-byte alignment invariants
    let header_size = core::mem::size_of::<FlatMeshHeader>();
    let lump_table_offset = 64;
    let lump_count = 2;
    let lump_table_size = lump_count * core::mem::size_of::<FlatMeshLumpEntry>();

    let pos_offset = 128u64;
    let pos_count = 4u32;
    let pos_bytes = (pos_count as usize) * core::mem::size_of::<FlatPositionVertex>();

    let attr_offset = (pos_offset as usize + pos_bytes) as u64;
    let attr_count = 4u32;
    let attr_bytes = (attr_count as usize) * core::mem::size_of::<FlatAttributeVertex>();

    let mut storage = AlignedStorage { data: [0u8; 512] };
    let buffer = &mut storage.data;

    // Ensure 64-byte buffer alignment
    assert_eq!(buffer.as_ptr() as usize % 64, 0);

    let lump_mask =
        (1u64 << LumpId::PositionStream as u32) | (1u64 << LumpId::AttributeStream as u32);

    let header = FlatMeshHeader {
        signature: FormatSignature::FLATMESH_V1_0,
        lump_table_offset_div32: 2, // 64 bytes
        lump_count: lump_count as u16,
        flags: FlatMeshHeader::FLAG_HAS_MESHLETS,
        lump_mask,
        bounding_sphere: [0.0, 0.0, 0.0, 10.0],
        aabb_min: [-10.0, -10.0, -10.0],
        aabb_max: [10.0, 10.0, 10.0],
    };
    buffer[..header_size].copy_from_slice(header.as_bytes());

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

    let lump_dest = &mut buffer[lump_table_offset..lump_table_offset + lump_table_size];
    lump_dest.copy_from_slice(lumps.as_bytes());

    // Populate vertex positions
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
    let pos_dest = &mut buffer[pos_offset as usize..pos_offset as usize + pos_bytes];
    pos_dest.copy_from_slice(positions.as_bytes());

    // Populate attributes with A2B10G10R10 normal
    let packed_norm = FlatPackedNormal::pack([0.0, 1.0, 0.0], 1.0);
    let attributes = [
        FlatAttributeVertex {
            normal_tangent: packed_norm,
            uv: [0, 0],
        },
        FlatAttributeVertex {
            normal_tangent: packed_norm,
            uv: [1000, 0],
        },
        FlatAttributeVertex {
            normal_tangent: packed_norm,
            uv: [1000, 1000],
        },
        FlatAttributeVertex {
            normal_tangent: packed_norm,
            uv: [0, 1000],
        },
    ];
    let attr_dest = &mut buffer[attr_offset as usize..attr_offset as usize + attr_bytes];
    attr_dest.copy_from_slice(attributes.as_bytes());

    // Mount FlatMeshView - Zero heap allocations!
    let view = FlatMeshView::new(buffer).expect("Valid container must mount without error");
    assert_eq!(view.header().signature, FormatSignature::FLATMESH_V1_0);
    assert_eq!(view.header().lump_count, 2);
    assert!(view.header().has_meshlets());

    let pos_slice = view.positions().expect("Must read position stream");
    assert_eq!(pos_slice.len(), 4);
    assert_eq!(pos_slice[1].position, [1.0, 0.0, 0.0]);

    // Test unchecked getter
    let pos_unchecked = unsafe { view.positions_unchecked() };
    assert_eq!(pos_unchecked.len(), 4);
    assert_eq!(pos_unchecked[1].position, [1.0, 0.0, 0.0]);

    let attr_slice = view.attributes().expect("Must read attribute stream");
    assert_eq!(attr_slice.len(), 4);
    let (unpacked_n, sign) = attr_slice[0].normal_tangent.unpack();
    assert!((unpacked_n[1] - 1.0).abs() < 0.01);
    assert_eq!(sign, 1.0);
}
