//! Meshlet Clusterizer for 3D Geometry and Skeletal Meshes.
//!
//! Partitions indexed triangle meshes into bounded clusters of <= 64 vertices
//! and <= 126 triangles conforming to modern GPU mesh shader hardware requirements.

use crate::meshlet::FlatMeshletDesc;
use crate::vertex::FlatPositionVertex;
use alloc::vec::Vec;

pub struct MeshletClusterResult {
    pub meshlets: Vec<FlatMeshletDesc>,
    pub micro_indices: Vec<u8>,
}

/// Greedy clusterizer partitioning triangle streams into GPU meshlets.
pub fn clusterize_triangles(
    positions: &[FlatPositionVertex],
    indices: &[u16],
) -> MeshletClusterResult {
    let mut meshlets = Vec::new();
    let mut micro_indices = Vec::new();

    let tri_count = indices.len() / 3;
    if tri_count == 0 {
        return MeshletClusterResult {
            meshlets,
            micro_indices,
        };
    }

    let mut cluster_verts: Vec<u16> = Vec::with_capacity(64);
    let mut cluster_tris: Vec<[u8; 3]> = Vec::with_capacity(126);

    let flush_cluster = |meshlets: &mut Vec<FlatMeshletDesc>,
                         micro_indices: &mut Vec<u8>,
                         verts: &mut Vec<u16>,
                         tris: &mut Vec<[u8; 3]>| {
        if verts.is_empty() || tris.is_empty() {
            return;
        }

        // Compute local cluster bounding sphere
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];

        for &v_idx in verts.iter() {
            if let Some(pv) = positions.get(v_idx as usize) {
                for i in 0..3 {
                    if pv.position[i] < min[i] {
                        min[i] = pv.position[i];
                    }
                    if pv.position[i] > max[i] {
                        max[i] = pv.position[i];
                    }
                }
            }
        }

        let center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];

        let mut max_dist_sq = 0.0f32;
        for &v_idx in verts.iter() {
            if let Some(pv) = positions.get(v_idx as usize) {
                let dx = pv.position[0] - center[0];
                let dy = pv.position[1] - center[1];
                let dz = pv.position[2] - center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq > max_dist_sq {
                    max_dist_sq = dist_sq;
                }
            }
        }
        let radius = max_dist_sq.sqrt();

        let vertex_offset = 0u32; // In simplified streaming, direct local micro-indices
        let triangle_offset = micro_indices.len() as u32;

        for tri in tris.iter() {
            micro_indices.push(tri[0]);
            micro_indices.push(tri[1]);
            micro_indices.push(tri[2]);
        }

        meshlets.push(FlatMeshletDesc {
            sphere_center: center,
            sphere_radius: radius,
            cone_axis_cutoff: [0, 127, 0, 0], // Default upper hemisphere
            vertex_offset,
            triangle_offset,
            vertex_count: verts.len() as u8,
            triangle_count: tris.len() as u8,
            flags: 0,
        });

        verts.clear();
        tris.clear();
    };

    for t in 0..tri_count {
        let idx0 = indices[t * 3];
        let idx1 = indices[t * 3 + 1];
        let idx2 = indices[t * 3 + 2];

        // Check how many new vertices this triangle would introduce
        let mut new_verts_needed = 0;
        if !cluster_verts.contains(&idx0) {
            new_verts_needed += 1;
        }
        if !cluster_verts.contains(&idx1) {
            new_verts_needed += 1;
        }
        if !cluster_verts.contains(&idx2) {
            new_verts_needed += 1;
        }

        if cluster_verts.len() + new_verts_needed > 64 || cluster_tris.len() + 1 > 126 {
            flush_cluster(
                &mut meshlets,
                &mut micro_indices,
                &mut cluster_verts,
                &mut cluster_tris,
            );
        }

        let local0 = if let Some(pos) = cluster_verts.iter().position(|&v| v == idx0) {
            pos as u8
        } else {
            let pos = cluster_verts.len() as u8;
            cluster_verts.push(idx0);
            pos
        };

        let local1 = if let Some(pos) = cluster_verts.iter().position(|&v| v == idx1) {
            pos as u8
        } else {
            let pos = cluster_verts.len() as u8;
            cluster_verts.push(idx1);
            pos
        };

        let local2 = if let Some(pos) = cluster_verts.iter().position(|&v| v == idx2) {
            pos as u8
        } else {
            let pos = cluster_verts.len() as u8;
            cluster_verts.push(idx2);
            pos
        };

        cluster_tris.push([local0, local1, local2]);
    }

    flush_cluster(
        &mut meshlets,
        &mut micro_indices,
        &mut cluster_verts,
        &mut cluster_tris,
    );

    MeshletClusterResult {
        meshlets,
        micro_indices,
    }
}
