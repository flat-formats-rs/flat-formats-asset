//! Succinct Bitmask Potentially Visible Set (PVS).
//!
//! Stores cluster-to-cluster visibility as an aligned bitmask matrix.
//! Enables $O(1)$ visibility tests and rank64 SIMD acceleration.

/// Succinct PVS Bitmask View.
#[derive(Debug, Clone, Copy)]
pub struct PvsView<'a> {
    cluster_count: usize,
    row_stride_bytes: usize,
    data: &'a [u8],
}

impl<'a> PvsView<'a> {
    /// Creates a PvsView over raw memory-mapped slice.
    #[inline]
    pub fn new(cluster_count: usize, data: &'a [u8]) -> Option<Self> {
        let row_stride_bytes = cluster_count.checked_add(7)? / 8;
        let required_bytes = cluster_count.checked_mul(row_stride_bytes)?;
        if data.len() < required_bytes {
            return None;
        }
        Some(Self {
            cluster_count,
            row_stride_bytes,
            data: &data[..required_bytes],
        })
    }

    /// Total number of clusters in the PVS table.
    #[inline(always)]
    pub fn cluster_count(&self) -> usize {
        self.cluster_count
    }

    /// Tests whether `target_cluster` is visible from `source_cluster` in $O(1)$ time.
    #[inline]
    pub fn is_visible(&self, source_cluster: usize, target_cluster: usize) -> bool {
        if source_cluster >= self.cluster_count || target_cluster >= self.cluster_count {
            return false;
        }
        let byte_offset = source_cluster * self.row_stride_bytes + (target_cluster / 8);
        let bit_mask = 1u8 << (target_cluster % 8);
        (self.data[byte_offset] & bit_mask) != 0
    }

    /// Returns the raw slice of visibility bits for `source_cluster`.
    #[inline]
    pub fn cluster_row(&self, source_cluster: usize) -> Option<&'a [u8]> {
        if source_cluster >= self.cluster_count {
            return None;
        }
        let start = source_cluster * self.row_stride_bytes;
        Some(&self.data[start..start + self.row_stride_bytes])
    }

    /// Counts visible clusters for `source_cluster` using fast bit popcounts.
    #[inline]
    pub fn count_visible(&self, source_cluster: usize) -> usize {
        let Some(row) = self.cluster_row(source_cluster) else {
            return 0;
        };
        let mut count = 0usize;
        let mut chunks = row.chunks_exact(8);
        for chunk in &mut chunks {
            let val = u64::from_le_bytes(chunk.try_into().unwrap());
            count += val.count_ones() as usize;
        }
        for &byte in chunks.remainder() {
            count += byte.count_ones() as usize;
        }
        count
    }
}
