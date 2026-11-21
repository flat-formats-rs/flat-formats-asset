use flatasset_mesh::{
    FlatAttributeVertex, FlatMeshBuilder, FlatMeshView, FlatPackedNormal, FlatPositionVertex,
    FlatSkinVertex, FormatSignature, LumpId,
};

#[test]
fn test_flatmesh_builder_roundtrip() {
    let positions = vec![
        FlatPositionVertex {
            position: [0.0, 0.0, 0.0],
            cluster_id: 0,
        },
        FlatPositionVertex {
            position: [10.0, 0.0, 0.0],
            cluster_id: 0,
        },
        FlatPositionVertex {
            position: [10.0, 10.0, 0.0],
            cluster_id: 0,
        },
        FlatPositionVertex {
            position: [0.0, 10.0, 0.0],
            cluster_id: 0,
        },
    ];

    let packed_norm = FlatPackedNormal::pack([0.0, 0.0, 1.0], 1.0);
    let attributes = vec![
        FlatAttributeVertex {
            normal_tangent: packed_norm,
            uv: [0, 0],
        },
        FlatAttributeVertex {
            normal_tangent: packed_norm,
            uv: [15360, 0], // 1.0 in f16
        },
        FlatAttributeVertex {
            normal_tangent: packed_norm,
            uv: [15360, 15360],
        },
        FlatAttributeVertex {
            normal_tangent: packed_norm,
            uv: [0, 15360],
        },
    ];

    let skin = vec![
        FlatSkinVertex {
            joint_indices: [1, 0, 0, 0],
            joint_weights: [255, 0, 0, 0],
        },
        FlatSkinVertex {
            joint_indices: [1, 0, 0, 0],
            joint_weights: [255, 0, 0, 0],
        },
        FlatSkinVertex {
            joint_indices: [2, 0, 0, 0],
            joint_weights: [255, 0, 0, 0],
        },
        FlatSkinVertex {
            joint_indices: [2, 0, 0, 0],
            joint_weights: [255, 0, 0, 0],
        },
    ];

    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];

    let builder = FlatMeshBuilder::new()
        .with_positions(positions.clone())
        .with_attributes(attributes.clone())
        .with_skin_rig(skin.clone())
        .with_indices_u16(&indices);

    let container_bytes = builder.build().expect("build flatmesh package");

    // FlatMesh requires 64-byte alignment (e.g. mmap / cache-line alignment)
    let mut aligned = Vec::with_capacity(container_bytes.len() + 64);
    let offset = (64 - (aligned.as_ptr() as usize % 64)) % 64;
    aligned.resize(offset, 0);
    aligned.extend_from_slice(&container_bytes);
    let aligned_slice = &aligned[offset..offset + container_bytes.len()];
    assert_eq!(aligned_slice.as_ptr() as usize % 64, 0);

    let view = FlatMeshView::new(aligned_slice).expect("valid FlatMeshView");

    assert_eq!(view.header().signature, FormatSignature::FLATMESH_V1_0);
    assert_eq!(view.lumps().len(), 4);
    assert!(view.header().has_skeleton());
    assert!(!view.header().has_meshlets());

    // Verify auto bounds
    assert_eq!(view.header().aabb_min, [0.0, 0.0, 0.0]);
    assert_eq!(view.header().aabb_max, [10.0, 10.0, 0.0]);
    assert_eq!(view.header().bounding_sphere[0], 5.0);
    assert_eq!(view.header().bounding_sphere[1], 5.0);
    assert_eq!(view.header().bounding_sphere[2], 0.0);

    // Verify positions stream
    let read_pos = view.positions().expect("read positions");
    assert_eq!(read_pos.len(), 4);
    assert_eq!(read_pos[1].position, [10.0, 0.0, 0.0]);

    // Verify attributes stream
    let read_attr = view.attributes().expect("read attributes");
    assert_eq!(read_attr.len(), 4);
    assert_eq!(read_attr[1].uv, [15360, 0]);

    // Verify indices
    let idx_lump = view
        .find_lump(LumpId::IndexBuffer)
        .expect("find index lump");
    assert_eq!(idx_lump.element_count, 6);
    assert_eq!(idx_lump.element_stride, 2);
    let idx_bytes = view.lump_bytes(idx_lump).expect("index bytes");
    assert_eq!(idx_bytes.len(), 12);
}
