use flatasset_world::{
    header::LumpType, meshlet::FlatWorldMeshlet, qbvh::quantize_aabb, FlatWorldBuilder,
    FlatWorldView, QBvh4Node, QBVH_LEAF_MASK,
};
use zerocopy::IntoBytes;

#[test]
fn test_flatworld_roundtrip_and_simd_raycast() {
    let world_min = [-1024.0, -1024.0, -512.0];
    let world_max = [1024.0, 1024.0, 512.0];

    // 1. Construct QBvh4 node
    let mut node = QBvh4Node::empty();
    let (q_min, q_max) = quantize_aabb(
        [-100.0, -100.0, -10.0],
        [100.0, 100.0, 10.0],
        world_min,
        world_max,
    );
    node.min_x[0] = q_min[0];
    node.max_x[0] = q_max[0];
    node.min_y[0] = q_min[1];
    node.max_y[0] = q_max[1];
    node.min_z[0] = q_min[2];
    node.max_z[0] = q_max[2];
    node.children[0] = QBVH_LEAF_MASK | 42;

    // 2. Construct Meshlet
    let meshlet = FlatWorldMeshlet {
        bounding_sphere: [0.0, 0.0, 0.0, 150.0],
        vertex_offset: 0,
        triangle_offset: 0,
        material_id: 1,
        vertex_count: 32,
        triangle_count: 64,
        cone_axis_cutoff: [0, 127, 0, 0],
    };

    // 3. Construct PVS
    let pvs_matrix = vec![0b0000_0001, 0b0000_0011]; // 2 clusters

    // 4. Construct KDL
    let kdl_text = "entity classname=\"worldspawn\" {\n    wad \"test.wad\"\n}\n".to_string();

    let qbvh_nodes = [node];
    let meshlets = [meshlet];

    let flw_bytes = FlatWorldBuilder::new(world_min, world_max)
        .add_lump(
            LumpType::QBvh4,
            qbvh_nodes.len() as u64,
            size_of::<QBvh4Node>() as u32,
            qbvh_nodes.as_bytes().to_vec(),
        )
        .add_lump(
            LumpType::Meshlets,
            meshlets.len() as u64,
            size_of::<FlatWorldMeshlet>() as u32,
            meshlets.as_bytes().to_vec(),
        )
        .add_lump(LumpType::VisibilityBitmask, 2, 1, pvs_matrix)
        .add_lump(
            LumpType::EntityGraph,
            kdl_text.len() as u64,
            1,
            kdl_text.into_bytes(),
        )
        .build();

    // Verify FlatWorldView
    let view = FlatWorldView::new(&flw_bytes).expect("Failed to parse FlatWorldView");
    let header = view.header();
    assert_eq!(header.bounds_min, world_min);
    assert_eq!(header.bounds_max, world_max);
    assert_eq!(header.lump_count, 4);

    let parsed_nodes = view.qbvh_nodes().expect("Missing QBvh4 nodes");
    assert_eq!(parsed_nodes.len(), 1);
    assert!(parsed_nodes[0].is_leaf(0));
    assert_eq!(parsed_nodes[0].child_index(0), 42);

    // Test raycast against child 0
    let ray_origin = [0.0, 0.0, 100.0];
    let inv_dir = [1e30, 1e30, -1.0];
    let (hit_mask, t_enters) =
        parsed_nodes[0].intersect_ray(ray_origin, inv_dir, 0.0, 1000.0, world_min, world_max);
    assert_eq!(hit_mask & 1, 1, "Ray should hit child 0 AABB");
    assert!(t_enters[0] > 0.0 && t_enters[0] < 100.0);

    // Test PVS
    let pvs = view.pvs_view().expect("Missing PVS");
    assert_eq!(pvs.cluster_count(), 2);
    assert!(pvs.is_visible(0, 0));
    assert!(!pvs.is_visible(0, 1));
    assert!(pvs.is_visible(1, 0));
    assert!(pvs.is_visible(1, 1));

    // Test KDL
    let kdl = view.entity_kdl().expect("Missing entity graph");
    assert!(kdl.contains("worldspawn"));
}
