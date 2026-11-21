//! # flatasset-audio
//!
//! Direct-to-Mixer zero-copy game audio container format (`.fls` - FlatSound).
//!
//! Features:
//! - 32-byte cache-line friendly header (`FlatAudioHeader`)
//! - Open-World codec identifier (`AudioCodecId`)
//! - Sample-accurate seamless looping markers (`loop_start_sample`, `loop_end_sample`)
//! - Monolithic seek & frame offset table for $O(1)$ random seek
//! - Zero-allocation slice mounting (`FlatAudioView`)

#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod codec;
pub mod error;
pub mod header;
pub mod reader;

pub use codec::AudioCodecId;
pub use error::{FlatAudioError, Result};
pub use header::{FlatAudioHeader, FlatAudioPacketDesc, FLATAUDIO_MAGIC, FLATAUDIO_VERSION};
pub use reader::FlatAudioView;

#[cfg(feature = "std")]
pub mod builder;

#[cfg(feature = "std")]
pub use builder::FlatAudioBuilder;
