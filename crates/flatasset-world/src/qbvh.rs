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
