//! Extreme Performance Streaming and DirectStorage DMA Helpers.
//!
//! Provides non-temporal streaming stores to bypass L1/L2/L3 CPU cache during large texture transfers,
//! preventing cache eviction of critical game simulation and render state.

/// NVMe DirectStorage and OS Page Alignment boundary (4096 bytes).
///
/// DirectStorage (`FILE_FLAG_NO_BUFFERING`) and Linux `io_uring` with `O_DIRECT`
/// require sector alignment (4KB) for direct-to-GPU DMA disk transfers.
pub const DIRECT_STORAGE_SECTOR_ALIGNMENT: usize = 4096;

/// Default container payload offset aligned to 4KB for DirectStorage NVMe DMA bypass.
pub const CONTAINER_PAYLOAD_OFFSET_4KB: usize = 4096;

/// High-throughput non-temporal memory copy bypassing CPU caches.
///
/// Direct re-export of `silicon_kit::streaming::stream_copy`.
/// Uses AVX2 streaming stores (`_mm256_stream_si256`) when available,
/// writing directly to write-combining (WC) GPU staging buffers or host RAM controllers.
#[inline(always)]
pub fn stream_copy(src: &[u8], dst: &mut [u8]) {
    silicon_kit::streaming::stream_copy(src, dst);
}
