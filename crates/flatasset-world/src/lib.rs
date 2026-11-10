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
    CompressionType, FlatWorldHeader, FlatWorldLumpDesc, LumpType, FLATWORLD_MAGIC,
    FLATWORLD_VERSION,
};
pub use meshlet::FlatWorldMeshlet;
pub use pvs::PvsView;
pub use qbvh::{quantize_aabb, QBvh4Node, QBVH_EMPTY_CHILD, QBVH_LEAF_MASK};
pub use reader::{FlatWorldError, FlatWorldView};
pub use writer::FlatWorldBuilder;
