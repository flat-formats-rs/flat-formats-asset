//! # flatmesh-core
//!
//! High-performance, zero-allocation, memory-mapped 3D geometry container format.
//!
//! Designed with mechanical sympathy, 32/64-byte cache-line alignment, and zero heap
//! allocations during parse, iteration, and direct GPU buffer uploads.

#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod error;
pub mod header;
pub mod lump;
pub mod meshlet;
pub mod reader;
pub mod vertex;

#[cfg(feature = "std")]
pub mod builder;
#[cfg(feature = "std")]
pub mod clusterizer;
#[cfg(feature = "std")]
pub mod mdl;

pub use error::{FlatMeshError, Result};
pub use header::{FlatMeshHeader, FormatSignature};
pub use lump::{FlatMeshLumpEntry, LumpId, LUMP_FLAG_REQUIRED};
pub use meshlet::{FlatDrawIndexedIndirect, FlatMeshletDesc};
pub use reader::FlatMeshView;
pub use vertex::{FlatAttributeVertex, FlatPackedNormal, FlatPositionVertex, FlatSkinVertex};

#[cfg(feature = "std")]
pub use builder::FlatMeshBuilder;
#[cfg(feature = "std")]
pub use clusterizer::clusterize_triangles;
#[cfg(feature = "std")]
pub use mdl::{MdlBone, MdlModelData, MdlTexture, StudioMdlParser};
