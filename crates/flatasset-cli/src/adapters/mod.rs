//! Legacy GoldSrc format adapters for the `fla` CLI converter tool.
//!
//! Separates legacy binary parsing (StudioMDL, BSP, WAD, etc.) from the core
//! zero-allocation engine container libraries (`flatasset-mesh`, `flatasset-texture`, etc.).

pub mod mdl;
