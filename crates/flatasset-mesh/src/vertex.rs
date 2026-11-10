//! Multi-Stream Vertex Layouts.
//!
//! Decomposed into orthogonal streams for maximum GPU cache and bandwidth efficiency:
//! - Stream 0 (Position only): 100% bus efficiency during Depth Prepass & Shadow mapping.
//! - Stream 1 (Surface attributes): Compact 8-byte packing (A2B10G10R10 Normals + FP16 UV).
//! - Stream 2 (Skeletal Rig): Compact 8-byte packing (Joint Indices + Weights).

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Stream 0: Position Vertex (16 bytes, 16-byte aligned).
///
/// Contains only XYZ position + cluster ID/padding.
/// Fetched exclusively during depth prepasses, shadow map generation, and raytracing BVH builds.
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C, align(16))]
pub struct FlatPositionVertex {
    pub position: [f32; 3],
    pub cluster_id: u32,
}
const _: () = assert!(core::mem::size_of::<FlatPositionVertex>() == 16);
const _: () = assert!(core::mem::align_of::<FlatPositionVertex>() == 16);

/// 32-bit packed normal vector with tangent sign (A2B10G10R10_SNORM / DXGI_FORMAT_R10G10B10A2_SNORM).
///
/// Direct re-export of `silicon_kit::snorm::PackedNormal10`.
/// Unpacked natively by GPU Input Assembler fixed-function silicon into `vec4` registers with 0 ALU instructions.
/// Provides 1024 quantization steps per axis (<0.08° error vs 0.9° for 8-bit octahedral).
pub type FlatPackedNormal = silicon_kit::snorm::PackedNormal10;

const _: () = assert!(core::mem::size_of::<FlatPackedNormal>() == 4);
const _: () = assert!(core::mem::align_of::<FlatPackedNormal>() == 4);

/// Stream 1: Surface Attributes (8 bytes, 8-byte aligned).
///
/// Packs unit normals and tangent sign into A2B10G10R10_SNORM (4B) and FP16 UVs (4B).
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, FromBytes, Immutable, IntoBytes, KnownLayout,
)]
#[repr(C, align(8))]
pub struct FlatAttributeVertex {
    /// 4 bytes: A2B10G10R10_SNORM normal + tangent sign (0 ALU GPU fixed-function unpack).
    pub normal_tangent: FlatPackedNormal,
    /// 4 bytes: Primary texture coordinates stored as IEEE 754 binary16 (half-precision).
    pub uv: [u16; 2],
}
const _: () = assert!(core::mem::size_of::<FlatAttributeVertex>() == 8);
const _: () = assert!(core::mem::align_of::<FlatAttributeVertex>() == 8);

/// Stream 2: Skeletal Rig Stream (8 bytes, 8-byte aligned).
///
/// Contains up to 4 joint indices and normalized weights summing to 255.
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, FromBytes, Immutable, IntoBytes, KnownLayout,
)]
#[repr(C, align(8))]
pub struct FlatSkinVertex {
    /// 4 bytes: Indices of influencing bones/joints.
    pub joint_indices: [u8; 4],
    /// 4 bytes: Normalized influence weights (0..255, where 255 = 1.0).
    pub joint_weights: [u8; 4],
}
const _: () = assert!(core::mem::size_of::<FlatSkinVertex>() == 8);
const _: () = assert!(core::mem::align_of::<FlatSkinVertex>() == 8);
