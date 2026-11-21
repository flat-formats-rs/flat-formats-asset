//! FlatAsset: Universal Zero-Alloc Binary Asset Ecosystem & Bundle Container.
//!
//! Provides unified discovery, verification, and inspection across all Flat formats:
//! - `.flm`: FlatMesh (Skeletal rigs, meshlets, animations)
//! - `.flt`: FlatTex (GPU Direct-DMA textures & compressed blocks)
//! - `.flw`: FlatWorld (Meshlets, Q-BVH4 collision, succinct PVS)
//! - `.fld`: FlatDemo (Columnar deterministic replay streams)
//! - `.flb`: FlatBundle (Multi-asset zero-alloc package container)

#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod bundle;
pub mod cas;
pub mod identify;
pub mod script;

pub use bundle::{
    FlatBundleBuilder, FlatBundleError, FlatBundleHeader, FlatBundleItemDesc, FlatBundleView,
    COMPRESSION_ALGO_MASK, COMPRESSION_RAW, COMPRESSION_ZSTD, FLAG_CAS_REFERENCE,
};
pub use cas::FlatCasPool;
pub use identify::{identify_flat_asset, FlatAssetKind};
pub use script::{ScriptEngineId, ScriptExecutionFlags, ScriptLumpDesc};

// Re-export underlying ecosystem core crates
pub use flatasset_audio as audio;
pub use flatasset_mesh as mesh;
pub use flatasset_tex as tex;
pub use flatasset_world as world;
pub use flatdemo_core as demo;
