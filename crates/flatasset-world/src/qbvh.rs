//! 4-Way Quantized Bounding Volume Hierarchy (Q-BVH4).
//!
//! Node data fits in a single 64-byte CPU cache line. Raycast tests 4 children in parallel
//! with SIMD intrinsics or branchless scalar operations.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Mask indicating that a child index points to a leaf primitive rather than an internal node.
pub const QBVH_LEAF_MASK: u32 = 0x8000_0000;
/// Sentinel value representing an empty/null child slot.
pub const QBVH_EMPTY_CHILD: u32 = 0xFFFF_FFFF;

/// 64-byte SIMD 4-way Quantized BVH Node (Q-BVH4).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct QBvh4Node {
    /// Quantized minimum X coordinates for 4 child bounding boxes.
    pub min_x: [u16; 4],
    /// Quantized maximum X coordinates for 4 child bounding boxes.
    pub max_x: [u16; 4],
    /// Quantized minimum Y coordinates for 4 child bounding boxes.
    pub min_y: [u16; 4],
    /// Quantized maximum Y coordinates for 4 child bounding boxes.
    pub max_y: [u16; 4],
    /// Quantized minimum Z coordinates for 4 child bounding boxes.
    pub min_z: [u16; 4],
    /// Quantized maximum Z coordinates for 4 child bounding boxes.
    pub max_z: [u16; 4],
    /// Child pointers: If bit 31 is set, lower 31 bits point to a leaf cluster.
    /// Otherwise, index into the node array. If `0xFFFF_FFFF`, child is empty.
    pub children: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<QBvh4Node>() == 64);

impl QBvh4Node {
    /// Creates an empty QBvh4Node with all children unassigned.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            min_x: [u16::MAX; 4],
            max_x: [0; 4],
            min_y: [u16::MAX; 4],
            max_y: [0; 4],
            min_z: [u16::MAX; 4],
            max_z: [0; 4],
            children: [QBVH_EMPTY_CHILD; 4],
        }
    }

    /// Checks if a child reference at `slot` (0..4) is a leaf.
    #[inline(always)]
    pub fn is_leaf(&self, slot: usize) -> bool {
        (self.children[slot] & QBVH_LEAF_MASK) != 0 && self.children[slot] != QBVH_EMPTY_CHILD
    }

    /// Checks if a child reference at `slot` is active (not empty).
    #[inline(always)]
    pub fn is_active(&self, slot: usize) -> bool {
        self.children[slot] != QBVH_EMPTY_CHILD
    }

    /// Decodes the payload index (node index or leaf primitive index).
    #[inline(always)]
    pub fn child_index(&self, slot: usize) -> u32 {
        self.children[slot] & !QBVH_LEAF_MASK
    }

    /// Dequantizes the child AABB in world coordinates given the world reference bounds.
    #[inline]
    pub fn dequantize_child_aabb(
        &self,
        slot: usize,
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> ([f32; 3], [f32; 3]) {
        let extent_x = (world_max[0] - world_min[0]) / 65535.0;
        let extent_y = (world_max[1] - world_min[1]) / 65535.0;
        let extent_z = (world_max[2] - world_min[2]) / 65535.0;

        let aabb_min = [
            world_min[0] + self.min_x[slot] as f32 * extent_x,
            world_min[1] + self.min_y[slot] as f32 * extent_y,
            world_min[2] + self.min_z[slot] as f32 * extent_z,
        ];
        let aabb_max = [
            world_min[0] + self.max_x[slot] as f32 * extent_x,
            world_min[1] + self.max_y[slot] as f32 * extent_y,
            world_min[2] + self.max_z[slot] as f32 * extent_z,
        ];
        (aabb_min, aabb_max)
    }

    /// Tests a ray against all 4 child AABBs in parallel.
    ///
    /// Returns a 4-bit hit mask (bits 0..3) indicating which children intersect
    /// the ray within `[t_min, t_max]`, and the respective entry distances.
    #[inline]
    pub fn intersect_ray(
        &self,
        origin: [f32; 3],
        inv_dir: [f32; 3],
        t_min: f32,
        t_max: f32,
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> (u8, [f32; 4]) {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: On x86_64, SSE2 is baseline and always available.
            unsafe { self.intersect_ray_sse2(origin, inv_dir, t_min, t_max, world_min, world_max) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.intersect_ray_scalar(origin, inv_dir, t_min, t_max, world_min, world_max)
        }
    }

    /// Tests a ray against all 4 child AABBs using pure scalar arithmetic.
    #[inline]
    pub fn intersect_ray_scalar(
        &self,
        origin: [f32; 3],
        inv_dir: [f32; 3],
        t_min: f32,
        t_max: f32,
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> (u8, [f32; 4]) {
        let mut hit_mask = 0u8;
        let mut t_enters = [f32::MAX; 4];

        for (i, t_enter) in t_enters.iter_mut().enumerate() {
            if !self.is_active(i) {
                continue;
            }

            let (box_min, box_max) = self.dequantize_child_aabb(i, world_min, world_max);

            let t0_x = (box_min[0] - origin[0]) * inv_dir[0];
            let t1_x = (box_max[0] - origin[0]) * inv_dir[0];
            let tx_min = t0_x.min(t1_x);
            let tx_max = t0_x.max(t1_x);

            let t0_y = (box_min[1] - origin[1]) * inv_dir[1];
            let t1_y = (box_max[1] - origin[1]) * inv_dir[1];
            let ty_min = t0_y.min(t1_y);
            let ty_max = t0_y.max(t1_y);

            let t0_z = (box_min[2] - origin[2]) * inv_dir[2];
            let t1_z = (box_max[2] - origin[2]) * inv_dir[2];
            let tz_min = t0_z.min(t1_z);
            let tz_max = t0_z.max(t1_z);

            let entry = tx_min.max(ty_min).max(tz_min).max(t_min);
            let exit = tx_max.min(ty_max).min(tz_max).min(t_max);

            if entry <= exit {
                hit_mask |= 1 << i;
                *t_enter = entry;
            }
        }

        (hit_mask, t_enters)
    }

    /// Vectorized x86_64 SSE2 ray-AABB intersection test for 4 children simultaneously.
    ///
    /// # Safety
    ///
    /// Caller must ensure that SSE2 instructions are supported by the target CPU (always true on standard x86_64).
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub unsafe fn intersect_ray_sse2(
        &self,
        origin: [f32; 3],
        inv_dir: [f32; 3],
        t_min: f32,
        t_max: f32,
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> (u8, [f32; 4]) {
        use core::arch::x86_64::*;

        let extent_x = _mm_set1_ps((world_max[0] - world_min[0]) / 65535.0);
        let extent_y = _mm_set1_ps((world_max[1] - world_min[1]) / 65535.0);
        let extent_z = _mm_set1_ps((world_max[2] - world_min[2]) / 65535.0);

        let wmin_x = _mm_set1_ps(world_min[0]);
        let wmin_y = _mm_set1_ps(world_min[1]);
        let wmin_z = _mm_set1_ps(world_min[2]);

        let orig_x = _mm_set1_ps(origin[0]);
        let orig_y = _mm_set1_ps(origin[1]);
        let orig_z = _mm_set1_ps(origin[2]);

        let idir_x = _mm_set1_ps(inv_dir[0]);
        let idir_y = _mm_set1_ps(inv_dir[1]);
        let idir_z = _mm_set1_ps(inv_dir[2]);

        let tmin_vec = _mm_set1_ps(t_min);
        let tmax_vec = _mm_set1_ps(t_max);

        let zero_i = _mm_setzero_si128();

        let unpack_u16_f32 = |ptr: *const u16| -> __m128 {
            let val64 = (ptr as *const i64).read_unaligned();
            let vec64 = _mm_set_epi64x(0, val64);
            let vec32 = _mm_unpacklo_epi16(vec64, zero_i);
            _mm_cvtepi32_ps(vec32)
        };

        let min_x_f = unpack_u16_f32(self.min_x.as_ptr());
        let max_x_f = unpack_u16_f32(self.max_x.as_ptr());
        let min_y_f = unpack_u16_f32(self.min_y.as_ptr());
        let max_y_f = unpack_u16_f32(self.max_y.as_ptr());
        let min_z_f = unpack_u16_f32(self.min_z.as_ptr());
        let max_z_f = unpack_u16_f32(self.max_z.as_ptr());

        // Dequantize: box = world_min + u16_val * extent
        let bmin_x = _mm_add_ps(wmin_x, _mm_mul_ps(min_x_f, extent_x));
        let bmax_x = _mm_add_ps(wmin_x, _mm_mul_ps(max_x_f, extent_x));
        let bmin_y = _mm_add_ps(wmin_y, _mm_mul_ps(min_y_f, extent_y));
        let bmax_y = _mm_add_ps(wmin_y, _mm_mul_ps(max_y_f, extent_y));
        let bmin_z = _mm_add_ps(wmin_z, _mm_mul_ps(min_z_f, extent_z));
        let bmax_z = _mm_add_ps(wmin_z, _mm_mul_ps(max_z_f, extent_z));

        // Slab test
        let t0_x = _mm_mul_ps(_mm_sub_ps(bmin_x, orig_x), idir_x);
        let t1_x = _mm_mul_ps(_mm_sub_ps(bmax_x, orig_x), idir_x);
        let tx_min = _mm_min_ps(t0_x, t1_x);
        let tx_max = _mm_max_ps(t0_x, t1_x);

        let t0_y = _mm_mul_ps(_mm_sub_ps(bmin_y, orig_y), idir_y);
        let t1_y = _mm_mul_ps(_mm_sub_ps(bmax_y, orig_y), idir_y);
        let ty_min = _mm_min_ps(t0_y, t1_y);
        let ty_max = _mm_max_ps(t0_y, t1_y);

        let t0_z = _mm_mul_ps(_mm_sub_ps(bmin_z, orig_z), idir_z);
        let t1_z = _mm_mul_ps(_mm_sub_ps(bmax_z, orig_z), idir_z);
        let tz_min = _mm_min_ps(t0_z, t1_z);
        let tz_max = _mm_max_ps(t0_z, t1_z);

        let entry = _mm_max_ps(_mm_max_ps(tx_min, ty_min), _mm_max_ps(tz_min, tmin_vec));
        let exit = _mm_min_ps(_mm_min_ps(tx_max, ty_max), _mm_min_ps(tz_max, tmax_vec));

        // Active children mask: child != QBVH_EMPTY_CHILD
        let children_vec = (self.children.as_ptr() as *const __m128i).read_unaligned();
        let empty_vec = _mm_set1_epi32(-1);
        let is_empty = _mm_cmpeq_epi32(children_vec, empty_vec);
        let is_active = _mm_castsi128_ps(_mm_andnot_si128(is_empty, empty_vec));

        let hit_cmp = _mm_cmple_ps(entry, exit);
        let valid_hits = _mm_and_ps(hit_cmp, is_active);
        let hit_mask = (_mm_movemask_ps(valid_hits) & 0x0F) as u8;

        let max_float = _mm_set1_ps(f32::MAX);
        let final_entry = _mm_or_ps(
            _mm_and_ps(valid_hits, entry),
            _mm_andnot_ps(valid_hits, max_float),
        );

        let mut t_enters = [f32::MAX; 4];
        _mm_storeu_ps(t_enters.as_mut_ptr(), final_entry);

        (hit_mask, t_enters)
    }

    /// Tests an AABB (e.g. player swept capsule/box) against all 4 child AABBs.
    /// Returns 4-bit hit mask.
    #[inline]
    pub fn intersect_aabb(
        &self,
        query_min: [f32; 3],
        query_max: [f32; 3],
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> u8 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: On x86_64, SSE2 is baseline.
            unsafe { self.intersect_aabb_sse2(query_min, query_max, world_min, world_max) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.intersect_aabb_scalar(query_min, query_max, world_min, world_max)
        }
    }

    /// Scalar fallback for AABB intersection.
    #[inline]
    pub fn intersect_aabb_scalar(
        &self,
        query_min: [f32; 3],
        query_max: [f32; 3],
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> u8 {
        let mut hit_mask = 0u8;
        for i in 0..4 {
            if !self.is_active(i) {
                continue;
            }
            let (box_min, box_max) = self.dequantize_child_aabb(i, world_min, world_max);
            let overlaps = box_min[0] <= query_max[0]
                && box_max[0] >= query_min[0]
                && box_min[1] <= query_max[1]
                && box_max[1] >= query_min[1]
                && box_min[2] <= query_max[2]
                && box_max[2] >= query_min[2];

            if overlaps {
                hit_mask |= 1 << i;
            }
        }
        hit_mask
    }

    /// Vectorized x86_64 SSE2 AABB overlap test against all 4 children.
    ///
    /// # Safety
    ///
    /// Caller must ensure that SSE2 instructions are supported by the target CPU (always true on standard x86_64).
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub unsafe fn intersect_aabb_sse2(
        &self,
        query_min: [f32; 3],
        query_max: [f32; 3],
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> u8 {
        use core::arch::x86_64::*;

        let extent_x = _mm_set1_ps((world_max[0] - world_min[0]) / 65535.0);
        let extent_y = _mm_set1_ps((world_max[1] - world_min[1]) / 65535.0);
        let extent_z = _mm_set1_ps((world_max[2] - world_min[2]) / 65535.0);

        let wmin_x = _mm_set1_ps(world_min[0]);
        let wmin_y = _mm_set1_ps(world_min[1]);
        let wmin_z = _mm_set1_ps(world_min[2]);

        let qmin_x = _mm_set1_ps(query_min[0]);
        let qmax_x = _mm_set1_ps(query_max[0]);
        let qmin_y = _mm_set1_ps(query_min[1]);
        let qmax_y = _mm_set1_ps(query_max[1]);
        let qmin_z = _mm_set1_ps(query_min[2]);
        let qmax_z = _mm_set1_ps(query_max[2]);

        let zero_i = _mm_setzero_si128();
        let unpack_u16_f32 = |ptr: *const u16| -> __m128 {
            let val64 = (ptr as *const i64).read_unaligned();
            let vec64 = _mm_set_epi64x(0, val64);
            let vec32 = _mm_unpacklo_epi16(vec64, zero_i);
            _mm_cvtepi32_ps(vec32)
        };

        let min_x_f = unpack_u16_f32(self.min_x.as_ptr());
        let max_x_f = unpack_u16_f32(self.max_x.as_ptr());
        let min_y_f = unpack_u16_f32(self.min_y.as_ptr());
        let max_y_f = unpack_u16_f32(self.max_y.as_ptr());
        let min_z_f = unpack_u16_f32(self.min_z.as_ptr());
        let max_z_f = unpack_u16_f32(self.max_z.as_ptr());

        let bmin_x = _mm_add_ps(wmin_x, _mm_mul_ps(min_x_f, extent_x));
        let bmax_x = _mm_add_ps(wmin_x, _mm_mul_ps(max_x_f, extent_x));
        let bmin_y = _mm_add_ps(wmin_y, _mm_mul_ps(min_y_f, extent_y));
        let bmax_y = _mm_add_ps(wmin_y, _mm_mul_ps(max_y_f, extent_y));
        let bmin_z = _mm_add_ps(wmin_z, _mm_mul_ps(min_z_f, extent_z));
        let bmax_z = _mm_add_ps(wmin_z, _mm_mul_ps(max_z_f, extent_z));

        let ox = _mm_and_ps(_mm_cmple_ps(bmin_x, qmax_x), _mm_cmpge_ps(bmax_x, qmin_x));
        let oy = _mm_and_ps(_mm_cmple_ps(bmin_y, qmax_y), _mm_cmpge_ps(bmax_y, qmin_y));
        let oz = _mm_and_ps(_mm_cmple_ps(bmin_z, qmax_z), _mm_cmpge_ps(bmax_z, qmin_z));

        let children_vec = (self.children.as_ptr() as *const __m128i).read_unaligned();
        let empty_vec = _mm_set1_epi32(-1);
        let is_empty = _mm_cmpeq_epi32(children_vec, empty_vec);
        let is_active = _mm_castsi128_ps(_mm_andnot_si128(is_empty, empty_vec));

        let overlaps = _mm_and_ps(_mm_and_ps(ox, oy), _mm_and_ps(oz, is_active));
        (_mm_movemask_ps(overlaps) & 0x0F) as u8
    }
}

/// Quantizes an AABB `[min, max]` into `u16` coordinates relative to world bounds.
#[inline]
pub fn quantize_aabb(
    min: [f32; 3],
    max: [f32; 3],
    world_min: [f32; 3],
    world_max: [f32; 3],
) -> ([u16; 3], [u16; 3]) {
    let scale_x = 65535.0 / (world_max[0] - world_min[0]).max(1e-6);
    let scale_y = 65535.0 / (world_max[1] - world_min[1]).max(1e-6);
    let scale_z = 65535.0 / (world_max[2] - world_min[2]).max(1e-6);

    let q_min = [
        ((min[0] - world_min[0]) * scale_x).clamp(0.0, 65535.0) as u16,
        ((min[1] - world_min[1]) * scale_y).clamp(0.0, 65535.0) as u16,
        ((min[2] - world_min[2]) * scale_z).clamp(0.0, 65535.0) as u16,
    ];

    let q_max = [
        ((max[0] - world_min[0]) * scale_x)
            .clamp(0.0, 65535.0)
            .ceil() as u16,
        ((max[1] - world_min[1]) * scale_y)
            .clamp(0.0, 65535.0)
            .ceil() as u16,
        ((max[2] - world_min[2]) * scale_z)
            .clamp(0.0, 65535.0)
            .ceil() as u16,
    ];

    (q_min, q_max)
}

/// Precomputed ray parameters for accelerated traversal through world bounds.
#[derive(Debug, Clone, Copy)]
pub struct PreparedRay {
    pub origin: [f32; 3],
    pub inv_dir: [f32; 3],
    pub t_min: f32,
    pub t_max: f32,
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
}

impl PreparedRay {
    /// Creates a new PreparedRay.
    #[inline]
    pub fn new(
        origin: [f32; 3],
        inv_dir: [f32; 3],
        t_min: f32,
        t_max: f32,
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> Self {
        Self {
            origin,
            inv_dir,
            t_min,
            t_max,
            world_min,
            world_max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_matches_scalar_raycast() {
        let world_min = [-1024.0, -1024.0, -512.0];
        let world_max = [1024.0, 1024.0, 512.0];

        let mut node = QBvh4Node::empty();
        for i in 0..4 {
            let offset = i as f32 * 80.0;
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
            node.children[i] = QBVH_LEAF_MASK | (i as u32);
        }

        // Test multiple rays from different angles and origins
        for step in 0..100 {
            let angle = (step as f32) * 0.1;
            let origin = [angle.cos() * 200.0, angle.sin() * 200.0, 150.0];
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

            let (scalar_mask, scalar_t) =
                node.intersect_ray_scalar(origin, inv_dir, 0.0, 1e10, world_min, world_max);

            #[cfg(target_arch = "x86_64")]
            unsafe {
                let (simd_mask, simd_t) =
                    node.intersect_ray_sse2(origin, inv_dir, 0.0, 1e10, world_min, world_max);
                assert_eq!(
                    simd_mask, scalar_mask,
                    "Raycast mask mismatch at step {step}: SIMD {simd_mask:#b} vs Scalar {scalar_mask:#b}"
                );
                for i in 0..4 {
                    if (scalar_mask & (1 << i)) != 0 {
                        assert!(
                            (simd_t[i] - scalar_t[i]).abs() < 1e-3,
                            "Entry distance mismatch at child {i}: SIMD {} vs Scalar {}",
                            simd_t[i],
                            scalar_t[i]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_simd_matches_scalar_aabb() {
        let world_min = [-1024.0, -1024.0, -512.0];
        let world_max = [1024.0, 1024.0, 512.0];

        let mut node = QBvh4Node::empty();
        for i in 0..4 {
            let offset = i as f32 * 80.0;
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
            node.children[i] = QBVH_LEAF_MASK | (i as u32);
        }

        for step in 0..50 {
            let offset = step as f32 * 10.0;
            let query_min = [-50.0 + offset, -30.0, -10.0];
            let query_max = [50.0 + offset, 30.0, 10.0];

            let scalar_mask =
                node.intersect_aabb_scalar(query_min, query_max, world_min, world_max);

            #[cfg(target_arch = "x86_64")]
            unsafe {
                let simd_mask =
                    node.intersect_aabb_sse2(query_min, query_max, world_min, world_max);
                assert_eq!(
                    simd_mask, scalar_mask,
                    "AABB mask mismatch at step {step}: SIMD {simd_mask:#b} vs Scalar {scalar_mask:#b}"
                );
            }
        }
    }
}
