//! # `flatworld-core`
//!
//! Next-generation, zero-allocation game world container format (`.flw`).
//!
//! Synthesizes modern GPU meshlets, 4-way SIMD Quantized BVH (Q-BVH4),
//! succinct bitmask PVS, and strongly-typed KDL scene graphs.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod header;
pub mod meshlet;
pub mod pvs;
pub mod qbvh;
pub mod reader;
#[cfg(feature = "builder")]
pub mod writer;

pub use header::{
    CompressionType, FlatWorldHeader, FlatWorldLumpDesc, LumpType, FLATWORLD_MAGIC,
    FLATWORLD_VERSION,
};
pub use meshlet::FlatWorldMeshlet;
pub use pvs::{
    encode_chunked_pvs, encode_word_sparse_pvs, ChunkedPvsHeader, ChunkedPvsRowDesc,
    ChunkedPvsView, PvsView, WordSparsePvsHeader, WordSparsePvsView, WordSparseRowDesc, WorldPvs,
    CHUNKED_PVS_MAGIC, PVS_CHUNK_BITS, PVS_CHUNK_BYTES, WORD_SPARSE_PVS_MAGIC,
};
pub use qbvh::{quantize_aabb, PreparedRay, QBvh4Node, QBVH_EMPTY_CHILD, QBVH_LEAF_MASK};
pub use reader::{FlatWorldError, FlatWorldView};
#[cfg(feature = "builder")]
pub use writer::FlatWorldBuilder;
