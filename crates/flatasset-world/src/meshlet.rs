//! FlatWorld static meshlet cluster descriptors.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// 32-byte Meshlet Cluster Descriptor for GPU-driven and indirect rasterization.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct FlatWorldMeshlet {
    /// Bounding sphere: Center [X, Y, Z] and Radius W for frustum culling
    pub bounding_sphere: [f32; 4],
    /// Byte offset into the vertex stream
    pub vertex_offset: u32,
    /// Byte offset into the micro-triangle index stream
    pub triangle_offset: u32,
    /// Material identifier index
    pub material_id: u16,
    /// Number of vertices in cluster (max 64)
    pub vertex_count: u8,
    /// Number of micro-triangles in cluster (max 128)
    pub triangle_count: u8,
    /// Normal cone orientation: Axis [X, Y, Z] as snorm8, and Cutoff angle as snorm8
    pub cone_axis_cutoff: [i8; 4],
}

const _: () = assert!(core::mem::size_of::<FlatWorldMeshlet>() == 32);
