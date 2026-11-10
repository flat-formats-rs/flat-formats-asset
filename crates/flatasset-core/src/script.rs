//! Open-World Scripting Architecture for the FlatAsset Ecosystem.
//!
//! Provides extensible FourCC runtime identification (`ScriptEngineId`) and
//! exact 64-byte aligned descriptors (`ScriptLumpDesc`) for embedding sandboxed
//! scripts (WebAssembly, Wasmtime AOT, Lua, Rhai, QuickJS, KDL Event Graphs)
//! directly inside `.flw` (world logic) and `.flm` (model/weapon behavior).

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Open-World FourCC identifier for the targeted script runtime engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(transparent)]
pub struct ScriptEngineId(pub u32);

impl ScriptEngineId {
    /// No script embedded.
    pub const NONE: Self = Self(0);

    /// Standard WebAssembly Core or Component Model bytecode (`b"WASM"`, 0x4D534157 LE).
    pub const WASM: Self = Self(u32::from_le_bytes(*b"WASM"));

    /// Wasmtime Ahead-of-Time (AOT) compiled native module (`b"CWAS"`, 0x53415743 LE).
    pub const CWASM: Self = Self(u32::from_le_bytes(*b"CWAS"));

    /// Lua or LuaJIT compiled bytecode (`b"LUAC"`, 0x4341554C LE).
    pub const LUA: Self = Self(u32::from_le_bytes(*b"LUAC"));

    /// Rhai embedded Rust-like script (`b"RHAI"`, 0x49414852 LE).
    pub const RHAI: Self = Self(u32::from_le_bytes(*b"RHAI"));

    /// QuickJS or JavaScript engine (`b"QJS\0"`, 0x00534A51 LE).
    pub const QUICKJS: Self = Self(u32::from_le_bytes(*b"QJS\0"));

    /// Declarative KDL Event Graph (`b"KDLS"`, 0x534C444B LE).
    pub const KDL_GRAPH: Self = Self(u32::from_le_bytes(*b"KDLS"));

    /// Constructs a custom runtime identifier from an arbitrary 4-byte ASCII code.
    #[inline(always)]
    pub const fn from_fourcc(fourcc: [u8; 4]) -> Self {
        Self(u32::from_le_bytes(fourcc))
    }

    /// Extracts the 4-byte ASCII mnemonic.
    #[inline(always)]
    pub const fn as_fourcc(&self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    /// Human-readable name for well-known runtimes.
    pub const fn display_name(&self) -> &'static str {
        match self.0 {
            0 => "None",
            0x4D534157 => "WebAssembly (.wasm)",
            0x53415743 => "Wasmtime AOT (.cwasm)",
            0x4341554C => "Lua / LuaJIT (.luac)",
            0x49414852 => "Rhai Script (.rhai)",
            0x00534A51 => "QuickJS (.js)",
            0x534C444B => "KDL Event Graph (.kdl)",
            _ => "Custom Engine",
        }
    }
}

/// Bitflags defining where and how the script is permitted to execute.
pub struct ScriptExecutionFlags;

impl ScriptExecutionFlags {
    /// Client-side execution permitted (visual effects, procedural IK, camera shake, prediction).
    pub const CLIENT_PERMITTED: u32 = 1 << 0;
    /// Server-side authoritative execution required (damage, state mutation, physics events).
    pub const SERVER_REQUIRED: u32 = 1 << 1;
    /// Headless AI simulation (server-side bots, NPCs).
    pub const BOT_AI: u32 = 1 << 2;
    /// Hot-reloading enabled in development mode.
    pub const HOT_RELOAD: u32 = 1 << 3;
}

/// Exact 64-byte descriptor for an embedded script lump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct ScriptLumpDesc {
    /// FourCC runtime identifier (e.g. `b"WASM"`).
    pub engine_id: ScriptEngineId,
    /// Bitmask of `ScriptExecutionFlags`.
    pub execution_flags: u32,
    /// BLAKE3 64-bit ABI contract fingerprint for interface compatibility checks.
    pub abi_hash: u64,
    /// Absolute file offset where the script payload begins (64-byte aligned).
    pub bytecode_offset: u64,
    /// Size of bytecode payload on disk in bytes.
    pub bytecode_size: u64,
    /// Uncompressed memory size in bytes (if compressed in container).
    pub uncompressed_size: u64,
    /// Offset to entrypoint string within the container (0 for standard entry).
    pub entrypoint_offset: u32,
    /// Reserved zero padding to ensure exact 64-byte size and alignment.
    pub _pad: [u8; 20],
}

const _: () = assert!(core::mem::size_of::<ScriptLumpDesc>() == 64);
