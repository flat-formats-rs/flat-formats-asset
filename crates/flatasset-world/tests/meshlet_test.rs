use flatasset_world::meshlet::{dequantize_cluster_position, MeshletClusterizer, RawMeshletVertex};
use flatasset_world::{FlatWorldBuilder, FlatWorldView, LumpType};
use std::fs;

#[test]
fn test_meshlet_builder_synthetic_and_roundtrip() {
    let mut clusterizer = MeshletClusterizer::new();

    let v0 = RawMeshletVertex {
        position: [0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    };
    let v1 = RawMeshletVertex {
        position: [100.0, 0.0, 0.0],
        uv: [1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    };
    let v2 = RawMeshletVertex {
        position: [50.0, 80.0, 0.0],
        uv: [0.5, 1.0],
        normal: [0.0, 0.0, 1.0],
    };

    clusterizer.add_triangle(42, [v0, v1, v2]);
    let payload = clusterizer.build();

    let flw = FlatWorldBuilder::new([-500.0; 3], [500.0; 3])
        .add_lump(LumpType::Meshlets, 1, 32, payload)
        .build();

    let world_view = FlatWorldView::new(&flw).expect("parse world view");
    let m_opt = world_view
        .meshlet_lump_view()
        .expect("retrieve meshlet view");
    assert!(m_opt.is_some());
    let m_view = m_opt.unwrap();

    assert_eq!(m_view.meshlets().len(), 1);
    assert_eq!(m_view.vertices().len(), 3);
    assert_eq!(m_view.triangles().len(), 1);

    let meshlet = &m_view.meshlets()[0];
    assert_eq!(meshlet.material_id, 42);
    assert_eq!(meshlet.vertex_count, 3);
    assert_eq!(meshlet.triangle_count, 1);

    let verts = m_view.meshlet_vertices(meshlet).unwrap();
    let center = [
        meshlet.bounding_sphere[0],
        meshlet.bounding_sphere[1],
        meshlet.bounding_sphere[2],
    ];

    for (i, v) in verts.iter().enumerate() {
        let restored = dequantize_cluster_position(v.q_pos, center);
        let orig = [v0, v1, v2][i].position;
        for axis in 0..3 {
            let diff = (restored[axis] - orig[axis]).abs();
            assert!(diff < 0.05, "axis {axis} precision loss: diff={diff}");
        }
    }
}

#[test]
fn test_real_inferno_meshlet_extraction_and_invariants() {
    let flw_path = "d:/Repo/flatworld-rs/de_inferno_full_world.flw";
    let Ok(flw_bytes) = fs::read(flw_path) else {
        eprintln!("de_inferno_full_world.flw not found, skipping real map test");
        return;
    };

    let world_view = FlatWorldView::new(&flw_bytes).expect("parse world");
    let m_opt = world_view.meshlet_lump_view().expect("meshlet view");
    assert!(m_opt.is_some());
    let m_view = m_opt.unwrap();

    let meshlets = m_view.meshlets();
    assert_eq!(meshlets.len(), 2227);
    assert_eq!(m_view.vertices().len(), 31560);
    assert_eq!(m_view.triangles().len(), 25346);

    for m in meshlets {
        assert!(m.vertex_count <= 64);
        assert!(m.triangle_count <= 126);
        let verts = m_view.meshlet_vertices(m).expect("valid verts slice");
        let tris = m_view.meshlet_triangles(m).expect("valid tris slice");

        let center = [
            m.bounding_sphere[0],
            m.bounding_sphere[1],
            m.bounding_sphere[2],
        ];
        let radius = m.bounding_sphere[3];
        assert!(radius > 0.0);

        for tri in tris {
            for &idx in &tri.indices {
                assert!(
                    (idx as usize) < verts.len(),
                    "local triangle index out of bounds"
                );
            }
        }

        for v in verts {
            let p = dequantize_cluster_position(v.q_pos, center);
            let dx = p[0] - center[0];
            let dy = p[1] - center[1];
            let dz = p[2] - center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                dist <= radius + 0.1,
                "vertex outside meshlet sphere: dist={dist}, r={radius}"
            );
        }
    }
}
