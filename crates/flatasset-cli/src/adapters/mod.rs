//! Legacy format adapters for the `fla` CLI converter tool.
//!
//! Separates legacy binary parsing (StudioMDL, RIFF WAV, etc.) from the core
//! zero-allocation engine container libraries (`flatasset-mesh`, `flatasset-audio`, etc.).

pub mod mdl;
pub mod spr;
pub mod wav;
