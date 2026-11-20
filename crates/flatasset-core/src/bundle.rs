//! Zero-alloc Multi-Asset Package Container (`.flb`).
//!
//! Stores multiple heterogeneous Flat assets (`.flm`, `.flt`, `.flw`, `.fld`) in a single
//! contiguous zero-copy container with 64-byte alignment and BLAKE3 integrity.

use crate::identify::identify_flat_asset;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Container magic for FlatBundle: `b"FLTB"` (0x42544C46 in LE)
pub const FLATBUNDLE_MAGIC: u32 = 0x42544C46;
/// Current format version
pub const FLATBUNDLE_VERSION: u32 = 1;

/// Exact 64-byte container header for FlatBundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct FlatBundleHeader {
    /// Magic identifier: `FLTB`
    pub magic: u32,
    /// Format version
    pub version: u32,
    /// Number of bundled items
    pub item_count: u32,
    /// Offset to items index table
    pub index_offset: u32,
    /// Total container size on disk
    pub total_size: u64,
    /// BLAKE3 Merkle root across items
    pub checksum: [u8; 16],
    /// Reserved zero padding to align to 64 bytes
    pub _pad: [u8; 24],
}

const _: () = assert!(core::mem::size_of::<FlatBundleHeader>() == 64);

/// Exact 64-byte item descriptor inside a FlatBundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct FlatBundleItemDesc {
    /// Name hash (Murmur3/CityHash/BLAKE3-truncated)
    pub name_hash: u64,
    /// Offset in file where the asset begins (64-byte aligned)
    pub file_offset: u64,
    /// Length of the asset on disk in bytes
    pub byte_length: u64,
    /// Asset kind (FlatAssetKind as u32)
    pub asset_kind: u32,
    /// In-container compression (0 = raw mmap, 1 = Zstd)
    pub compression: u32,
    /// Uncompressed memory size in bytes
    pub uncompressed_size: u64,
    /// Reserved zero padding
    pub _pad: [u8; 24],
}

const _: () = assert!(core::mem::size_of::<FlatBundleItemDesc>() == 64);

#[derive(Debug, thiserror::Error)]
pub enum FlatBundleError {
    #[error("Buffer too small: expected {expected}, actual {actual}")]
    BufferTooSmall { expected: usize, actual: usize },
    #[error("Invalid magic: expected 0x{expected:08X}, actual 0x{actual:08X}")]
    InvalidMagic { expected: u32, actual: u32 },
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("Index table out of bounds: offset {offset}, count {count}")]
    IndexOutOfBounds { offset: u32, count: u32 },
    #[error("Item index out of bounds: {0}")]
    ItemOutOfBounds(usize),
    #[error("Item payload out of bounds: offset {offset} + size {size} > total {limit}")]
    ItemRangeOutOfBounds {
        offset: u64,
        size: u64,
        limit: usize,
    },
}

fn item_range_error(item: &FlatBundleItemDesc, limit: usize) -> FlatBundleError {
    FlatBundleError::ItemRangeOutOfBounds {
        offset: item.file_offset,
        size: item.byte_length,
        limit,
    }
}

/// Zero-copy read-only view over a FlatBundle container.
#[derive(Debug, Clone, Copy)]
pub struct FlatBundleView<'a> {
    raw_bytes: &'a [u8],
    header: &'a FlatBundleHeader,
    items: &'a [FlatBundleItemDesc],
}

impl<'a> FlatBundleView<'a> {
    /// Mounts and validates a FlatBundle container from an in-memory slice in $O(1)$ time.
    pub fn new(bytes: &'a [u8]) -> Result<Self, FlatBundleError> {
        if bytes.len() < core::mem::size_of::<FlatBundleHeader>() {
            return Err(FlatBundleError::BufferTooSmall {
                expected: core::mem::size_of::<FlatBundleHeader>(),
                actual: bytes.len(),
            });
        }

        let (header, _) = FlatBundleHeader::ref_from_prefix(bytes).map_err(|_| {
            FlatBundleError::BufferTooSmall {
                expected: core::mem::size_of::<FlatBundleHeader>(),
                actual: bytes.len(),
            }
        })?;

        if header.magic != FLATBUNDLE_MAGIC {
            return Err(FlatBundleError::InvalidMagic {
                expected: FLATBUNDLE_MAGIC,
                actual: header.magic,
            });
        }
        if header.version != FLATBUNDLE_VERSION {
            return Err(FlatBundleError::UnsupportedVersion(header.version));
        }

        let count = header.item_count as usize;
        let index_start = usize::try_from(header.index_offset).map_err(|_| {
            FlatBundleError::IndexOutOfBounds {
                offset: header.index_offset,
                count: header.item_count,
            }
        })?;
        let index_size = count
            .checked_mul(core::mem::size_of::<FlatBundleItemDesc>())
            .ok_or(FlatBundleError::IndexOutOfBounds {
                offset: header.index_offset,
                count: header.item_count,
            })?;
        let index_end =
            index_start
                .checked_add(index_size)
                .ok_or(FlatBundleError::IndexOutOfBounds {
                    offset: header.index_offset,
                    count: header.item_count,
                })?;

        if index_end > bytes.len() {
            return Err(FlatBundleError::IndexOutOfBounds {
                offset: header.index_offset,
                count: header.item_count,
            });
        }

        let (items, _) = <[FlatBundleItemDesc]>::ref_from_prefix_with_elems(
            &bytes[index_start..index_end],
            count,
        )
        .map_err(|_| FlatBundleError::IndexOutOfBounds {
            offset: header.index_offset,
            count: header.item_count,
        })?;

        for item in items.iter() {
            let start = usize::try_from(item.file_offset)
                .map_err(|_| item_range_error(item, bytes.len()))?;
            let size = usize::try_from(item.byte_length)
                .map_err(|_| item_range_error(item, bytes.len()))?;
            let end = start
                .checked_add(size)
                .ok_or_else(|| item_range_error(item, bytes.len()))?;
            if end > bytes.len() {
                return Err(item_range_error(item, bytes.len()));
            }
        }

        Ok(Self {
            raw_bytes: bytes,
            header,
            items,
        })
    }

    #[inline(always)]
    pub fn header(&self) -> &'a FlatBundleHeader {
        self.header
    }

    #[inline(always)]
    pub fn items(&self) -> &'a [FlatBundleItemDesc] {
        self.items
    }

    /// Accesses the raw byte payload of an asset item by descriptor.
    ///
    /// # Errors
    /// Returns [`FlatBundleError::ItemRangeOutOfBounds`] if the descriptor
    /// points outside the mounted buffer or overflows `usize`.
    #[inline]
    pub fn item_bytes(&self, item: &FlatBundleItemDesc) -> Result<&'a [u8], FlatBundleError> {
        let limit = self.raw_bytes.len();
        let start = usize::try_from(item.file_offset).map_err(|_| item_range_error(item, limit))?;
        let size = usize::try_from(item.byte_length).map_err(|_| item_range_error(item, limit))?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| item_range_error(item, limit))?;
        self.raw_bytes
            .get(start..end)
            .ok_or_else(|| item_range_error(item, limit))
    }

    /// Accesses item bytes by item index.
    pub fn get_item(
        &self,
        index: usize,
    ) -> Result<(&'a FlatBundleItemDesc, &'a [u8]), FlatBundleError> {
        let item = self
            .items
            .get(index)
            .ok_or(FlatBundleError::ItemOutOfBounds(index))?;
        Ok((item, self.item_bytes(item)?))
    }
}

#[cfg(feature = "std")]
#[derive(Default)]
pub struct FlatBundleBuilder {
    items: std::vec::Vec<FlatBundleItemDesc>,
    payload_data: std::vec::Vec<u8>,
}

#[cfg(feature = "std")]
impl FlatBundleBuilder {
    pub fn new() -> Self {
        Self {
            items: std::vec::Vec::new(),
            payload_data: std::vec::Vec::new(),
        }
    }

    /// Adds an asset to the bundle, automatically detecting its kind.
    pub fn add_asset(&mut self, name_hash: u64, asset_bytes: &[u8], compress: bool) {
        let kind = identify_flat_asset(asset_bytes);

        // Align current payload position to 64 bytes
        let current_len = self.payload_data.len();
        let aligned_len = (current_len + 63) & !63;
        self.payload_data.resize(aligned_len, 0);

        let file_offset = 64 + self.payload_data.len() as u64; // Header is 64 bytes

        let (data_to_write, is_compressed, compressed_len) = if compress {
            #[cfg(feature = "zstd")]
            {
                let compressed =
                    zstd::bulk::compress(asset_bytes, 3).unwrap_or_else(|_| asset_bytes.to_vec());
                let c_len = compressed.len();
                (compressed, 1u32, c_len)
            }
            #[cfg(not(feature = "zstd"))]
            {
                (asset_bytes.to_vec(), 0u32, asset_bytes.len())
            }
        } else {
            (asset_bytes.to_vec(), 0u32, asset_bytes.len())
        };

        let desc = FlatBundleItemDesc {
            name_hash,
            file_offset,
            byte_length: compressed_len as u64,
            asset_kind: kind as u32,
            compression: is_compressed,
            uncompressed_size: asset_bytes.len() as u64,
            _pad: [0; 24],
        };

        self.payload_data.extend_from_slice(&data_to_write);
        self.items.push(desc);
    }

    /// Serializes the complete `.flb` container with index table and BLAKE3 checksum.
    pub fn build(mut self) -> std::vec::Vec<u8> {
        let item_count = self.items.len() as u32;

        // Align end of payload before index table
        let current_payload_len = self.payload_data.len();
        let aligned_payload_len = (current_payload_len + 63) & !63;
        self.payload_data.resize(aligned_payload_len, 0);

        let index_offset = (64 + self.payload_data.len()) as u32;
        let total_size = (index_offset as usize
            + self.items.len() * core::mem::size_of::<FlatBundleItemDesc>())
            as u64;

        let mut checksum = [0u8; 16];
        #[cfg(feature = "blake3")]
        {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&self.payload_data);
            let hash = hasher.finalize();
            checksum.copy_from_slice(&hash.as_bytes()[..16]);
        }

        let header = FlatBundleHeader {
            magic: FLATBUNDLE_MAGIC,
            version: FLATBUNDLE_VERSION,
            item_count,
            index_offset,
            total_size,
            checksum,
            _pad: [0; 24],
        };

        let mut output = std::vec::Vec::with_capacity(total_size as usize);
        output.extend_from_slice(header.as_bytes());
        output.extend_from_slice(&self.payload_data);

        for item in &self.items {
            output.extend_from_slice(item.as_bytes());
        }

        output
    }
}
