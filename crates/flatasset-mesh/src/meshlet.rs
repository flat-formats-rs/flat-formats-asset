//! Modern GPU Meshlet Clusters and GPU-Driven Indirect Draw Commands.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// 32-byte GPU Meshlet Cluster Descriptor.
///
/// Pre-partitioned into bounded clusters of <= 64 vertices and <= 126 triangles.
/// Fits exactly 2 meshlet descriptors in a single 64-byte L1 CPU/GPU cache line.
/// Zero padding holes.
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C, align(32))]
pub struct FlatMeshletDesc {
    /// 12 bytes (offset 0..12): Bounding sphere center [x, y, z] in local coordinates.
    pub sphere_center: [f32; 3],
    /// 4 bytes (offset 12..16): Bounding sphere radius.
    pub sphere_radius: f32,
    /// 4 bytes (offset 16..20): Normal cone apex/axis packed as [i8; 4] (3-axis SNORM + 1-cutoff SNORM).
    pub cone_axis_cutoff: [i8; 4],
    /// 4 bytes (offset 20..24): Offset into the meshlet vertex indirection index buffer.
    pub vertex_offset: u32,
    /// 4 bytes (offset 24..28): Offset into the local 8-bit triangle micro-indices buffer.
    pub triangle_offset: u32,
    /// 1 byte (offset 28..29): Number of unique vertices in this cluster (<= 64).
    pub vertex_count: u8,
    /// 1 byte (offset 29..30): Number of local triangles in this cluster (<= 126).
    pub triangle_count: u8,
    /// 2 bytes (offset 30..32): Flags and reserved alignment.
    pub flags: u16,
}
const _: () = assert!(core::mem::size_of::<FlatMeshletDesc>() == 32);
const _: () = assert!(core::mem::align_of::<FlatMeshletDesc>() == 32);

/// 20-byte GPU-Driven Indirect Draw Command (Standard Vulkan VkDrawIndexedIndirectCommand).
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, FromBytes, Immutable, IntoBytes, KnownLayout,
)]
#[repr(C)]
pub struct FlatDrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}
const _: () = assert!(core::mem::size_of::<FlatDrawIndexedIndirect>() == 20);
