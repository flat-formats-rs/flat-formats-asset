//! Content-Addressable Storage (CAS) Asset Pool and deduplication engine.
//!
//! Provides zero-copy, memory-mapped shared asset pools (`shared/pool.flb` or individual items)
//! indexed by BLAKE3 cryptographic hashes. Allows multiple maps/bundles to reference identical
//! textures, sounds, and models without duplicating bytes on disk or in GPU/RAM.

use crate::bundle::{FlatBundleError, FlatBundleItemDesc, FlatBundleView};
use alloc::collections::BTreeMap;

/// High-performance Content-Addressable Storage (CAS) Pool.
///
/// Holds references to one or more shared `.flb` pools or memory-mapped buffers,
/// mapping 64-bit truncated BLAKE3 hashes to asset payloads.
#[derive(Default)]
pub struct FlatCasPool<'a> {
    /// Lookup table from 64-bit BLAKE3 hash prefix to slice and descriptor.
    entries: BTreeMap<u64, (&'a FlatBundleItemDesc, &'a [u8])>,
}

impl<'a> FlatCasPool<'a> {
    /// Creates a new empty CAS pool.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Mounts and indexes all assets from a shared [`FlatBundleView`].
    pub fn register_bundle(
        &mut self,
        bundle: &'a FlatBundleView<'a>,
    ) -> Result<usize, FlatBundleError> {
        let mut count = 0;
        for item in bundle.items() {
            if !item.is_cas_reference() || item.byte_length > 0 {
                let bytes = bundle.item_bytes(item)?;
                self.entries.insert(item.name_hash, (item, bytes));
                count += 1;
            }
        }
        Ok(count)
    }

    /// Registers a raw standalone flat asset by its explicit BLAKE3 hash.
    pub fn register_raw_asset(
        &mut self,
        blake3_hash: [u8; 32],
        desc: &'a FlatBundleItemDesc,
        bytes: &'a [u8],
    ) {
        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&blake3_hash[..8]);
        let hash_key = u64::from_le_bytes(prefix);
        self.entries.insert(hash_key, (desc, bytes));
    }

    /// Resolves an asset payload by 64-bit BLAKE3 prefix key in $O(\log N)$ (or $O(1)$).
    #[inline]
    pub fn get_by_hash(&self, hash_key: u64) -> Option<(&'a FlatBundleItemDesc, &'a [u8])> {
        self.entries.get(&hash_key).copied()
    }

    /// Resolves a CAS reference item descriptor from a bundle against this pool.
    pub fn resolve_item(
        &self,
        item: &FlatBundleItemDesc,
        bundle: &'a FlatBundleView<'a>,
    ) -> Result<&'a [u8], FlatBundleError> {
        if item.is_cas_reference() {
            if let Some((_, bytes)) = self.get_by_hash(item.name_hash) {
                return Ok(bytes);
            }
        }
        // Fallback to inline item in bundle
        bundle.item_bytes(item)
    }

    /// Returns the total number of unique assets registered in this CAS pool.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the pool contains no assets.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
