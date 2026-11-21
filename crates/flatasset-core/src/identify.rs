//! Unified magic identification for all FlatAsset binary container formats.

/// Enum representing the recognized binary asset kinds in the Flat ecosystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum FlatAssetKind {
    /// `.flm`: FlatMesh (Magic: `b"FLTM"`, `0x4D544C46`)
    Mesh = 0x4D544C46,
    /// `.flt`: FlatTexture (Magic: `b"FLTT"`, `0x54544C46`)
    Texture = 0x54544C46,
    /// `.flw`: FlatWorld (Magic: `b"FLW\0"`, `0x00574C46`)
    World = 0x00574C46,
    /// `.fld`: FlatDemo (Magic: `b"FLD\0"`, `0x00444C46`)
    Demo = 0x00444C46,
    /// `.flb`: FlatBundle (Magic: `b"FLTB"`, `0x42544C46`)
    Bundle = 0x42544C46,
    /// `.fls`: FlatAudio / FlatSound (Magic: `b"FLTS"`, `0x53544C46`)
    Audio = 0x53544C46,
    /// Unknown or unaligned binary payload
    Unknown = 0,
}

impl FlatAssetKind {
    /// Returns the standard file extension without dot.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mesh => "flm",
            Self::Texture => "flt",
            Self::World => "flw",
            Self::Demo => "fld",
            Self::Bundle => "flb",
            Self::Audio => "fls",
            Self::Unknown => "bin",
        }
    }

    /// Human-readable title of the asset kind.
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Mesh => "FlatMesh (Skeletal Mesh/Animation)",
            Self::Texture => "FlatTexture (Direct-DMA GPU Texture)",
            Self::World => "FlatWorld (Map/Geometry/BVH/PVS)",
            Self::Demo => "FlatDemo (Columnar Replay Stream)",
            Self::Bundle => "FlatBundle (Multi-Asset Container)",
            Self::Audio => "FlatAudio (Direct-to-Mixer Audio/Opus)",
            Self::Unknown => "Unknown Binary Payload",
        }
    }
}

/// Identifies the asset kind from a byte slice prefix.
/// Requires at least 4 bytes to check the magic word.
#[inline]
pub fn identify_flat_asset(bytes: &[u8]) -> FlatAssetKind {
    if bytes.len() < 4 {
        return FlatAssetKind::Unknown;
    }

    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    match magic {
        0x4D544C46 => FlatAssetKind::Mesh,
        0x54544C46 => FlatAssetKind::Texture,
        0x00574C46 => FlatAssetKind::World,
        0x00444C46 => FlatAssetKind::Demo,
        0x42544C46 => FlatAssetKind::Bundle,
        0x53544C46 => FlatAssetKind::Audio,
        _ => FlatAssetKind::Unknown,
    }
}
