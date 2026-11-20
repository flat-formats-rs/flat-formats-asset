//! # `flatworld-core`
//!
//! Next-generation, zero-allocation game world container format (`.flw`).
//!
//! Synthesizes modern GPU meshlets, 4-way SIMD Quantized BVH (Q-BVH4),
//! succinct bitmask PVS, and strongly-typed KDL scene graphs.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bvh8;
pub mod header;
pub mod meshlet;
pub mod pvs;
pub mod qbvh;
pub mod reader;
#[cfg(feature = "builder")]
pub mod writer;

pub use bvh8::{ray_octant, Bvh8Node, BVH8_EMPTY_CHILD, BVH8_LEAF_MASK, OCTANT_ORDER_LUT};
pub use header::{
    CompressionType, FlatWorldHeader, FlatWorldLumpDesc, LumpType, FLATWORLD_MAGIC,
    FLATWORLD_VERSION,
};
pub use meshlet::FlatWorldMeshlet;
pub use pvs::{
    encode_chunked_pvs, encode_two_tier_pvs, encode_word_sparse_pvs, ChunkedPvsHeader,
    ChunkedPvsRowDesc, ChunkedPvsView, ClusterRange, PvsView, TwoTierPvsHeader, TwoTierPvsView,
    TwoTierRowDesc, WordSparsePvsHeader, WordSparsePvsView, WordSparseRowDesc, WorldPvs,
    CHUNKED_PVS_MAGIC, PVS_CHUNK_BITS, PVS_CHUNK_BYTES, TWO_TIER_PVS_MAGIC, WORD_SPARSE_PVS_MAGIC,
};
#[cfg(target_arch = "x86_64")]
pub use qbvh::PreparedScaledRay;
pub use qbvh::{
    quantize_aabb, CBvh4Node, PreparedRay, QBvh4Node, ScaledRay, QBVH_EMPTY_CHILD, QBVH_LEAF_MASK,
};
pub use reader::{FlatWorldError, FlatWorldView};
#[cfg(feature = "builder")]
pub use writer::FlatWorldBuilder;
