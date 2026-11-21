//! 8-Way Wide Bounding Volume Hierarchy (BVH8).
//!
//! Node fits in exactly two 64-byte CPU cache lines (128 bytes).
//! Children are partitioned along canonical octant split planes (Level 1: X, Level 2: Y, Level 3: Z).
//! Traversal order across hit children is branchless and determined in 4 cycles via an 8 KB L1D Octant LUT.

use super::qbvh::ScaledRay;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Mask indicating that a child index points to a leaf primitive rather than an internal node.
pub const BVH8_LEAF_MASK: u32 = 0x8000_0000;
/// Sentinel value representing an empty/null child slot.
pub const BVH8_EMPTY_CHILD: u32 = 0xFFFF_FFFF;

/// 128-byte 8-wide BVH Node (BVH8) fitting exactly into two 64-byte CPU cache lines.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct Bvh8Node {
    /// Quantized minimum X coordinates for 8 child bounding boxes.
    pub min_x: [u16; 8],
    /// Quantized maximum X coordinates for 8 child bounding boxes.
    pub max_x: [u16; 8],
    /// Quantized minimum Y coordinates for 8 child bounding boxes.
    pub min_y: [u16; 8],
    /// Quantized maximum Y coordinates for 8 child bounding boxes.
    pub max_y: [u16; 8],
    /// Quantized minimum Z coordinates for 8 child bounding boxes.
    pub min_z: [u16; 8],
    /// Quantized maximum Z coordinates for 8 child bounding boxes.
    pub max_z: [u16; 8],
    /// Child pointers for 8 children (leaf if MSB set, node index otherwise, empty if 0xFFFF_FFFF).
    pub children: [u32; 8],
}

const _: () = assert!(core::mem::size_of::<Bvh8Node>() == 128);
const _: () = assert!(core::mem::align_of::<Bvh8Node>() == 64);

impl Bvh8Node {
    /// Creates an empty Bvh8Node with all children unassigned.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            min_x: [u16::MAX; 8],
            max_x: [0; 8],
            min_y: [u16::MAX; 8],
            max_y: [0; 8],
            min_z: [u16::MAX; 8],
            max_z: [0; 8],
            children: [BVH8_EMPTY_CHILD; 8],
        }
    }

    /// Checks if a child reference at `slot` (0..8) is a leaf.
    #[inline(always)]
    pub fn is_leaf(&self, slot: usize) -> bool {
        (self.children[slot] & BVH8_LEAF_MASK) != 0 && self.children[slot] != BVH8_EMPTY_CHILD
    }

    /// Checks if a child reference at `slot` is active (not empty).
    #[inline(always)]
    pub fn is_active(&self, slot: usize) -> bool {
        self.children[slot] != BVH8_EMPTY_CHILD
    }

    /// Decodes the payload index (node index or leaf primitive index).
    #[inline(always)]
    pub fn child_index(&self, slot: usize) -> u32 {
        self.children[slot] & !BVH8_LEAF_MASK
    }

    /// Tests a pre-scaled ray against all 8 child AABBs and returns the hit mask and entry distances.
    #[inline]
    pub fn intersect_scaled_ray(&self, ray: &ScaledRay) -> (u8, [f32; 8]) {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: On x86_64, SSE2 is guaranteed.
            unsafe { self.intersect_scaled_ray_sse2(ray) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.intersect_scaled_ray_scalar(ray)
        }
    }

    /// Tests all 8 children using two SSE2 passes.
    ///
    /// # Safety
    ///
    /// Caller must ensure that SSE2 instructions are supported by the target CPU (baseline on x86_64).
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub unsafe fn intersect_scaled_ray_sse2(&self, ray: &ScaledRay) -> (u8, [f32; 8]) {
        use core::arch::x86_64::*;

        let orig_x = _mm_set1_ps(ray.orig_scaled[0]);
        let orig_y = _mm_set1_ps(ray.orig_scaled[1]);
        let orig_z = _mm_set1_ps(ray.orig_scaled[2]);

        let idir_x = _mm_set1_ps(ray.inv_dir_scaled[0]);
        let idir_y = _mm_set1_ps(ray.inv_dir_scaled[1]);
        let idir_z = _mm_set1_ps(ray.inv_dir_scaled[2]);

        let tmin_vec = _mm_set1_ps(ray.t_min);
        let tmax_vec = _mm_set1_ps(ray.t_max);
        let max_float = _mm_set1_ps(f32::MAX);
        let zero_i = _mm_setzero_si128();
        let empty_vec = _mm_set1_epi32(-1);

        let unpack_u16_f32 = |ptr: *const u16, offset: usize| -> __m128 {
            let val64 = (ptr.add(offset) as *const i64).read_unaligned();
            let vec64 = _mm_set_epi64x(0, val64);
            let vec32 = _mm_unpacklo_epi16(vec64, zero_i);
            _mm_cvtepi32_ps(vec32)
        };

        let mut final_mask = 0u8;
        let mut t_enters = [f32::MAX; 8];

        // Process 2 halves: children 0..4 and 4..8
        for half in 0..2 {
            let offset = half * 4;

            let min_x_f = unpack_u16_f32(self.min_x.as_ptr(), offset);
            let max_x_f = unpack_u16_f32(self.max_x.as_ptr(), offset);
            let min_y_f = unpack_u16_f32(self.min_y.as_ptr(), offset);
            let max_y_f = unpack_u16_f32(self.max_y.as_ptr(), offset);
            let min_z_f = unpack_u16_f32(self.min_z.as_ptr(), offset);
            let max_z_f = unpack_u16_f32(self.max_z.as_ptr(), offset);

            let t0_x = _mm_mul_ps(_mm_sub_ps(min_x_f, orig_x), idir_x);
            let t1_x = _mm_mul_ps(_mm_sub_ps(max_x_f, orig_x), idir_x);
            let tx_min = _mm_min_ps(t0_x, t1_x);
            let tx_max = _mm_max_ps(t0_x, t1_x);

            let t0_y = _mm_mul_ps(_mm_sub_ps(min_y_f, orig_y), idir_y);
            let t1_y = _mm_mul_ps(_mm_sub_ps(max_y_f, orig_y), idir_y);
            let ty_min = _mm_min_ps(t0_y, t1_y);
            let ty_max = _mm_max_ps(t0_y, t1_y);

            let t0_z = _mm_mul_ps(_mm_sub_ps(min_z_f, orig_z), idir_z);
            let t1_z = _mm_mul_ps(_mm_sub_ps(max_z_f, orig_z), idir_z);
            let tz_min = _mm_min_ps(t0_z, t1_z);
            let tz_max = _mm_max_ps(t0_z, t1_z);

            let entry = _mm_max_ps(_mm_max_ps(tx_min, ty_min), _mm_max_ps(tz_min, tmin_vec));
            let exit = _mm_min_ps(_mm_min_ps(tx_max, ty_max), _mm_min_ps(tz_max, tmax_vec));

            let children_vec =
                (self.children.as_ptr().add(offset) as *const __m128i).read_unaligned();
            let is_empty = _mm_cmpeq_epi32(children_vec, empty_vec);
            let is_active = _mm_castsi128_ps(_mm_andnot_si128(is_empty, empty_vec));

            let hit_cmp = _mm_cmple_ps(entry, exit);
            let valid_hits = _mm_and_ps(hit_cmp, is_active);
            let mask = (_mm_movemask_ps(valid_hits) & 0x0F) as u8;
            final_mask |= mask << offset;

            let final_entry = _mm_or_ps(
                _mm_and_ps(valid_hits, entry),
                _mm_andnot_ps(valid_hits, max_float),
            );

            _mm_storeu_ps(t_enters.as_mut_ptr().add(offset), final_entry);
        }

        (final_mask, t_enters)
    }

    /// Scalar fallback for `ScaledRay` against 8 children.
    #[inline]
    pub fn intersect_scaled_ray_scalar(&self, ray: &ScaledRay) -> (u8, [f32; 8]) {
        let mut hit_mask = 0u8;
        let mut t_enters = [f32::MAX; 8];

        for (i, t_enter) in t_enters.iter_mut().enumerate() {
            if !self.is_active(i) {
                continue;
            }

            let t0_x = (self.min_x[i] as f32 - ray.orig_scaled[0]) * ray.inv_dir_scaled[0];
            let t1_x = (self.max_x[i] as f32 - ray.orig_scaled[0]) * ray.inv_dir_scaled[0];
            let tx_min = t0_x.min(t1_x);
            let tx_max = t0_x.max(t1_x);

            let t0_y = (self.min_y[i] as f32 - ray.orig_scaled[1]) * ray.inv_dir_scaled[1];
            let t1_y = (self.max_y[i] as f32 - ray.orig_scaled[1]) * ray.inv_dir_scaled[1];
            let ty_min = t0_y.min(t1_y);
            let ty_max = t0_y.max(t1_y);

            let t0_z = (self.min_z[i] as f32 - ray.orig_scaled[2]) * ray.inv_dir_scaled[2];
            let t1_z = (self.max_z[i] as f32 - ray.orig_scaled[2]) * ray.inv_dir_scaled[2];
            let tz_min = t0_z.min(t1_z);
            let tz_max = t0_z.max(t1_z);

            let entry = tx_min.max(ty_min).max(tz_min).max(ray.t_min);
            let exit = tx_max.min(ty_max).min(tz_max).min(ray.t_max);

            if entry <= exit {
                hit_mask |= 1 << i;
                *t_enter = entry;
            }
        }

        (hit_mask, t_enters)
    }

    /// Queries the precomputed 8 KB Octant LUT to obtain the front-to-back traversal order
    /// across hit children with zero runtime sorting loops.
    ///
    /// # Arguments
    /// - `octant`: Ray sign octant index (0..7: bit 0 = sign X, bit 1 = sign Y, bit 2 = sign Z).
    /// - `hit_mask`: 8-bit hit mask from `intersect_scaled_ray`.
    ///
    /// Returns `(ordered_children, count)` where `ordered_children[0..count]` are child indices
    /// sorted front-to-back.
    #[inline(always)]
    pub fn sorted_traversal_order(octant: u8, hit_mask: u8) -> ([u8; 8], usize) {
        let key = ((octant as usize & 0x07) << 8) | (hit_mask as usize);
        let entry = OCTANT_ORDER_LUT[key];
        let count = (entry >> 28) as usize;
        let mut order = [0u8; 8];
        for (i, item) in order.iter_mut().enumerate().take(count) {
            *item = ((entry >> (i * 3)) & 0x07) as u8;
        }
        (order, count)
    }
}

/// Computes the 3-bit ray octant index from ray direction signs.
#[inline(always)]
pub fn ray_octant(dir: [f32; 3]) -> u8 {
    let mut octant = 0u8;
    if dir[0] < 0.0 {
        octant |= 1;
    }
    if dir[1] < 0.0 {
        octant |= 2;
    }
    if dir[2] < 0.0 {
        octant |= 4;
    }
    octant
}

/// Static 8 KB L1D Octant Traversal Table (2048 entries x 4 bytes).
///
/// Encodes the front-to-back topological traversal order of 8 children
/// for all 8 ray sign octants and all 256 possible hit masks.
/// Traversal order retrieval takes strictly 1 L1D cache read (4 clock cycles).
pub static OCTANT_ORDER_LUT: [u32; 2048] = generate_octant_order_lut();

/// Const-eval generator for the 8 KB Octant Traversal Table.
const fn generate_octant_order_lut() -> [u32; 2048] {
    let mut table = [0u32; 2048];
    let mut octant = 0;
    while octant < 8 {
        let mut mask = 0;
        while mask < 256 {
            let mut hit_children = [0usize; 8];
            let mut count = 0;
            let mut c = 0;
            while c < 8 {
                if (mask & (1 << c)) != 0 {
                    hit_children[count] = c;
                    count += 1;
                }
                c += 1;
            }

            // Insertion sort by topological priority: child ^ octant ascending
            let mut i = 1;
            while i < count {
                let key = hit_children[i];
                let key_priority = key ^ octant;
                let mut j = i;
                while j > 0 && (hit_children[j - 1] ^ octant) > key_priority {
                    hit_children[j] = hit_children[j - 1];
                    j -= 1;
                }
                hit_children[j] = key;
                i += 1;
            }

            // Pack: [31..28]: count, [23..0]: 8 x 3-bit child indices
            let mut entry = (count as u32) << 28;
            let mut k = 0;
            while k < count {
                entry |= (hit_children[k] as u32) << (k * 3);
                k += 1;
            }

            table[(octant << 8) | mask] = entry;
            mask += 1;
        }
        octant += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bvh8_layout_and_octant_lut() {
        assert_eq!(core::mem::size_of::<Bvh8Node>(), 128);
        assert_eq!(core::mem::align_of::<Bvh8Node>(), 64);
        assert_eq!(core::mem::size_of_val(&OCTANT_ORDER_LUT), 8192);

        // Test Octant 0 (+X, +Y, +Z): Ray travels from origin into positive octant.
        // Child 0 (000: nearest), Child 7 (111: farthest).
        let (order, count) = Bvh8Node::sorted_traversal_order(0, 0b1000_0001);
        assert_eq!(count, 2);
        assert_eq!(order[0], 0); // nearest
        assert_eq!(order[1], 7); // farthest

        // Test Octant 7 (-X, -Y, -Z): Ray travels into negative octant.
        // Child 7 (111: nearest), Child 0 (000: farthest).
        let (order7, count7) = Bvh8Node::sorted_traversal_order(7, 0b1000_0001);
        assert_eq!(count7, 2);
        assert_eq!(order7[0], 7); // nearest
        assert_eq!(order7[1], 0); // farthest
    }

    #[test]
    fn test_bvh8_simd_matches_scalar() {
        use crate::qbvh::quantize_aabb;

        let world_min = [-1024.0, -1024.0, -512.0];
        let world_max = [1024.0, 1024.0, 512.0];

        let mut node = Bvh8Node::empty();
        for i in 0..8 {
            let offset = i as f32 * 40.0;
            let (q_min, q_max) = quantize_aabb(
                [-100.0 + offset, -50.0, -20.0],
                [50.0 + offset, 50.0, 20.0],
                world_min,
                world_max,
            );
            node.min_x[i] = q_min[0];
            node.max_x[i] = q_max[0];
            node.min_y[i] = q_min[1];
            node.max_y[i] = q_max[1];
            node.min_z[i] = q_min[2];
            node.max_z[i] = q_max[2];
            node.children[i] = BVH8_LEAF_MASK | (i as u32);
        }

        for step in 0..50 {
            let angle = (step as f32) * 0.15;
            let origin = [angle.cos() * 200.0, angle.sin() * 200.0, 120.0];
            let dir = [-origin[0], -origin[1], -origin[2]];
            let inv_dir = [
                if dir[0].abs() < 1e-6 {
                    1e30
                } else {
                    1.0 / dir[0]
                },
                if dir[1].abs() < 1e-6 {
                    1e30
                } else {
                    1.0 / dir[1]
                },
                if dir[2].abs() < 1e-6 {
                    1e30
                } else {
                    1.0 / dir[2]
                },
            ];

            let scaled_ray =
                ScaledRay::from_world(origin, inv_dir, 0.0, 1e10, world_min, world_max);

            #[cfg(target_arch = "x86_64")]
            let (scalar_mask, scalar_t) = node.intersect_scaled_ray_scalar(&scaled_ray);
            #[cfg(not(target_arch = "x86_64"))]
            let (_scalar_mask, _scalar_t) = node.intersect_scaled_ray_scalar(&scaled_ray);

            #[cfg(target_arch = "x86_64")]
            unsafe {
                let (simd_mask, simd_t) = node.intersect_scaled_ray_sse2(&scaled_ray);
                assert_eq!(
                    simd_mask, scalar_mask,
                    "BVH8 mask mismatch at step {step}: SIMD {simd_mask:#b} vs Scalar {scalar_mask:#b}"
                );
                for i in 0..8 {
                    if (scalar_mask & (1 << i)) != 0 {
                        assert!(
                            (simd_t[i] - scalar_t[i]).abs() < 1e-3,
                            "BVH8 distance mismatch at child {i}: SIMD {} vs Scalar {}",
                            simd_t[i],
                            scalar_t[i]
                        );
                    }
                }
            }
        }
    }
}
