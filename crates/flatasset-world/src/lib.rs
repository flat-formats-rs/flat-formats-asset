//! # `flatworld-core`
//!
//! Next-generation, zero-allocation game world container format (`.flw`).
//!
//! Synthesizes modern GPU meshlets, 4-way SIMD Quantized BVH (Q-BVH4),
//! succinct bitmask PVS, and strongly-typed KDL scene graphs.

pub mod header;
pub mod meshlet;
pub mod pvs;
pub mod qbvh;
pub mod reader;
pub mod writer;

pub use header::{
    CompressionType, FLATWORLD_MAGIC, FLATWORLD_VERSION, FlatWorldHeader, FlatWorldLumpDesc,
    LumpType,
};
pub use meshlet::FlatWorldMeshlet;
pub use pvs::PvsView;
pub use qbvh::{QBVH_EMPTY_CHILD, QBVH_LEAF_MASK, QBvh4Node, quantize_aabb};
pub use reader::{FlatWorldError, FlatWorldView};
pub use writer::FlatWorldBuilder;
