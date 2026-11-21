//! Direct-to-GPU FlatMesh Container Builder.
//!
//! Synthesizes zero-copy `.flm` 3D geometry containers conforming to 64-byte L1 cache-line
//! alignment, multi-stream vertex layout, and Scrooge mechanical sympathy principles.

use crate::error::{FlatMeshError, Result};
use crate::header::{FlatMeshHeader, FormatSignature};
use crate::lump::{FlatMeshLumpEntry, LumpId, LUMP_FLAG_REQUIRED};
use crate::meshlet::{FlatDrawIndexedIndirect, FlatMeshletDesc};
use crate::vertex::{FlatAttributeVertex, FlatPositionVertex, FlatSkinVertex};
use alloc::vec::Vec;
use zerocopy::IntoBytes;

struct LumpPayload {
    lump_id: u16,
    flags: u16,
    version: u16,
    element_stride: u8,
    alignment_log2: u8,
    element_count: u32,
    data: Vec<u8>,
}

/// Builder for constructing valid zero-allocation `.flm` mesh packages.
#[derive(Default)]
pub struct FlatMeshBuilder {
    positions: Option<Vec<FlatPositionVertex>>,
    attributes: Option<Vec<FlatAttributeVertex>>,
    skin_rig: Option<Vec<FlatSkinVertex>>,
    indices: Option<(Vec<u8>, u8, u32)>, // (raw_bytes, stride, count)
    meshlet_clusters: Option<Vec<FlatMeshletDesc>>,
    meshlet_indices: Option<Vec<u8>>,
    indirect_draw: Option<Vec<FlatDrawIndexedIndirect>>,
    skeleton_hierarchy: Option<(Vec<u8>, u8, u32)>,
    animation_tracks: Option<(Vec<u8>, u8, u32)>,
    string_table: Option<Vec<u8>>,
    custom_lumps: Vec<LumpPayload>,
    manual_bounds: Option<([f32; 4], [f32; 3], [f32; 3])>,
}

impl FlatMeshBuilder {
    /// Creates a new builder instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets Stream 0: Vertex Positions (XYZ + cluster_id).
    pub fn with_positions(mut self, positions: Vec<FlatPositionVertex>) -> Self {
        self.positions = Some(positions);
        self
    }

    /// Sets Stream 1: Surface Attributes (A2B10G10R10 Normal + FP16 UV).
    pub fn with_attributes(mut self, attributes: Vec<FlatAttributeVertex>) -> Self {
        self.attributes = Some(attributes);
        self
    }

    /// Sets Stream 2: Skeletal Rig (Joints u8x4 + Weights u8x4).
    pub fn with_skin_rig(mut self, skin_rig: Vec<FlatSkinVertex>) -> Self {
        self.skin_rig = Some(skin_rig);
        self
    }

    /// Sets Stream 3: Triangle Index Buffer using 16-bit unsigned integers.
    pub fn with_indices_u16(mut self, indices: &[u16]) -> Self {
        let mut raw = Vec::with_capacity(indices.len() * 2);
        for &idx in indices {
            raw.extend_from_slice(&idx.to_le_bytes());
        }
        self.indices = Some((raw, 2, indices.len() as u32));
        self
    }

    /// Sets Stream 3: Triangle Index Buffer using 32-bit unsigned integers.
    pub fn with_indices_u32(mut self, indices: &[u32]) -> Self {
        let mut raw = Vec::with_capacity(indices.len() * 4);
        for &idx in indices {
            raw.extend_from_slice(&idx.to_le_bytes());
        }
        self.indices = Some((raw, 4, indices.len() as u32));
        self
    }

    /// Sets meshlet cluster descriptors and packed micro-indices.
    pub fn with_meshlets(mut self, clusters: Vec<FlatMeshletDesc>, micro_indices: Vec<u8>) -> Self {
        self.meshlet_clusters = Some(clusters);
        self.meshlet_indices = Some(micro_indices);
        self
    }

    /// Sets GPU Indirect Draw commands.
    pub fn with_indirect_draw(mut self, indirect: Vec<FlatDrawIndexedIndirect>) -> Self {
        self.indirect_draw = Some(indirect);
        self
    }

    /// Sets skeleton hierarchy lump.
    pub fn with_skeleton_hierarchy(
        mut self,
        data: Vec<u8>,
        element_stride: u8,
        element_count: u32,
    ) -> Self {
        self.skeleton_hierarchy = Some((data, element_stride, element_count));
        self
    }

    /// Sets animation tracks lump.
    pub fn with_animation_tracks(
        mut self,
        data: Vec<u8>,
        element_stride: u8,
        element_count: u32,
    ) -> Self {
        self.animation_tracks = Some((data, element_stride, element_count));
        self
    }

    /// Sets interned string table lump.
    pub fn with_string_table(mut self, string_table: Vec<u8>) -> Self {
        self.string_table = Some(string_table);
        self
    }

    /// Overrides automatically computed culling bounds (bounding sphere and AABB).
    pub fn with_bounds(
        mut self,
        bounding_sphere: [f32; 4],
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
    ) -> Self {
        self.manual_bounds = Some((bounding_sphere, aabb_min, aabb_max));
        self
    }

    /// Adds an arbitrary lump to the container.
    pub fn add_custom_lump(
        mut self,
        lump_id: u16,
        flags: u16,
        element_stride: u8,
        alignment_log2: u8,
        element_count: u32,
        data: Vec<u8>,
    ) -> Self {
        self.custom_lumps.push(LumpPayload {
            lump_id,
            flags,
            version: 1,
            element_stride,
            alignment_log2,
            element_count,
            data,
        });
        self
    }

    /// Serializes the complete `.flm` binary package.
    pub fn build(mut self) -> Result<Vec<u8>> {
        let mut payloads: Vec<LumpPayload> = Vec::new();

        // 1. Position Stream (Lump 0, 16B align)
        if let Some(pos) = self.positions.take() {
            let count = pos.len() as u32;
            let mut data = Vec::with_capacity(pos.len() * 16);
            for p in &pos {
                data.extend_from_slice(p.as_bytes());
            }
            payloads.push(LumpPayload {
                lump_id: LumpId::PositionStream as u16,
                flags: LUMP_FLAG_REQUIRED,
                version: 1,
                element_stride: 16,
                alignment_log2: 4, // 16 bytes
                element_count: count,
                data,
            });
        }

        // 2. Attribute Stream (Lump 1, 8B align)
        if let Some(attr) = self.attributes.take() {
            let count = attr.len() as u32;
            let mut data = Vec::with_capacity(attr.len() * 8);
            for a in &attr {
                data.extend_from_slice(a.as_bytes());
            }
            payloads.push(LumpPayload {
                lump_id: LumpId::AttributeStream as u16,
                flags: LUMP_FLAG_REQUIRED,
                version: 1,
                element_stride: 8,
                alignment_log2: 3, // 8 bytes
                element_count: count,
                data,
            });
        }

        // 3. Skin Rig Stream (Lump 2, 8B align)
        if let Some(skin) = self.skin_rig.take() {
            let count = skin.len() as u32;
            let mut data = Vec::with_capacity(skin.len() * 8);
            for s in &skin {
                data.extend_from_slice(s.as_bytes());
            }
            payloads.push(LumpPayload {
                lump_id: LumpId::SkinRigStream as u16,
                flags: 0,
                version: 1,
                element_stride: 8,
                alignment_log2: 3, // 8 bytes
                element_count: count,
                data,
            });
        }

        // 4. Index Buffer (Lump 3, 4B align)
        if let Some((data, stride, count)) = self.indices.take() {
            payloads.push(LumpPayload {
                lump_id: LumpId::IndexBuffer as u16,
                flags: 0,
                version: 1,
                element_stride: stride,
                alignment_log2: 2, // 4 bytes
                element_count: count,
                data,
            });
        }

        // 5. Meshlet Clusters (Lump 4, 32B align)
        if let Some(clusters) = self.meshlet_clusters.take() {
            let count = clusters.len() as u32;
            let mut data = Vec::with_capacity(clusters.len() * 32);
            for c in &clusters {
                data.extend_from_slice(c.as_bytes());
            }
            payloads.push(LumpPayload {
                lump_id: LumpId::MeshletClusters as u16,
                flags: 0,
                version: 1,
                element_stride: 32,
                alignment_log2: 5, // 32 bytes
                element_count: count,
                data,
            });
        }

        // 6. Meshlet Indices (Lump 5, 4B align)
        if let Some(micro) = self.meshlet_indices.take() {
            let count = micro.len() as u32;
            payloads.push(LumpPayload {
                lump_id: LumpId::MeshletIndices as u16,
                flags: 0,
                version: 1,
                element_stride: 1,
                alignment_log2: 2, // 4 bytes
                element_count: count,
                data: micro,
            });
        }

        // 7. Indirect Draw Buffer (Lump 6, 16B align)
        if let Some(indirect) = self.indirect_draw.take() {
            let count = indirect.len() as u32;
            let mut data = Vec::with_capacity(indirect.len() * 20);
            for ind in &indirect {
                data.extend_from_slice(ind.as_bytes());
            }
            payloads.push(LumpPayload {
                lump_id: LumpId::IndirectDrawBuffer as u16,
                flags: 0,
                version: 1,
                element_stride: 20,
                alignment_log2: 4, // 16 bytes
                element_count: count,
                data,
            });
        }

        // 8. Skeleton Hierarchy (Lump 7, 16B align)
        if let Some((data, stride, count)) = self.skeleton_hierarchy.take() {
            payloads.push(LumpPayload {
                lump_id: LumpId::SkeletonHierarchy as u16,
                flags: 0,
                version: 1,
                element_stride: stride,
                alignment_log2: 4, // 16 bytes
                element_count: count,
                data,
            });
        }

        // 9. Animation Tracks (Lump 8, 16B align)
        if let Some((data, stride, count)) = self.animation_tracks.take() {
            payloads.push(LumpPayload {
                lump_id: LumpId::AnimationTracks as u16,
                flags: 0,
                version: 1,
                element_stride: stride,
                alignment_log2: 4, // 16 bytes
                element_count: count,
                data,
            });
        }

        // 10. String Table (Lump 9, 1B align)
        if let Some(strings) = self.string_table.take() {
            let count = strings.len() as u32;
            payloads.push(LumpPayload {
                lump_id: LumpId::StringTable as u16,
                flags: 0,
                version: 1,
                element_stride: 1,
                alignment_log2: 0, // 1 byte
                element_count: count,
                data: strings,
            });
        }

        // Append custom lumps
        payloads.extend(self.custom_lumps);

        if payloads.is_empty() {
            return Err(FlatMeshError::InvalidLayout);
        }

        // CRITICAL INVARIANT: Sort lumps by lump_id ascending!
        // This is strictly required for O(1) branchless BMI2 rank64 indexing in FlatMeshView::find_lump.
        payloads.sort_by_key(|p| p.lump_id);

        let lump_count = payloads.len() as u16;
        let lump_table_offset = 64usize; // Header is exactly 64 bytes
        let lump_table_size = (lump_count as usize) * core::mem::size_of::<FlatMeshLumpEntry>();

        // Align start of all lump payload data to 64 bytes
        let mut cursor = (lump_table_offset + lump_table_size + 63) & !63;

        let mut lump_entries = Vec::with_capacity(lump_count as usize);
        let mut lump_mask = 0u64;
        let mut final_payload_bytes = Vec::new();

        for p in &payloads {
            let align = 1usize << (p.alignment_log2 as usize);
            // Advance cursor to match lump alignment
            let pad = (align - (cursor % align)) % align;
            if pad > 0 {
                final_payload_bytes.resize(final_payload_bytes.len() + pad, 0);
                cursor += pad;
            }

            let offset = cursor as u64;
            let uncompressed_size = p.data.len() as u64;
            let crc32 = calculate_crc32(&p.data);

            lump_entries.push(FlatMeshLumpEntry {
                lump_id: p.lump_id,
                flags: p.flags,
                version: p.version,
                element_stride: p.element_stride,
                alignment_log2: p.alignment_log2,
                offset,
                uncompressed_size,
                element_count: p.element_count,
                crc32,
            });

            if p.lump_id < 64 {
                lump_mask |= 1u64 << (p.lump_id as u32);
            }

            final_payload_bytes.extend_from_slice(&p.data);
            cursor += p.data.len();
        }

        // Compute culling bounds
        let (bounding_sphere, aabb_min, aabb_max) = self
            .manual_bounds
            .unwrap_or_else(|| compute_bounds_from_payloads(&payloads));

        // Determine feature flags
        let mut flags = 0u32;
        if lump_mask & (1u64 << LumpId::MeshletClusters as u32) != 0 {
            flags |= FlatMeshHeader::FLAG_HAS_MESHLETS;
        }
        if (lump_mask & (1u64 << LumpId::SkinRigStream as u32) != 0)
            || (lump_mask & (1u64 << LumpId::SkeletonHierarchy as u32) != 0)
        {
            flags |= FlatMeshHeader::FLAG_HAS_SKELETON;
        }

        let total_file_size = cursor;
        let mut container = alloc::vec![0u8; total_file_size];

        let header = FlatMeshHeader {
            signature: FormatSignature::FLATMESH_V1_0,
            lump_table_offset_div32: (lump_table_offset / 32) as u16, // = 2
            lump_count,
            flags,
            lump_mask,
            bounding_sphere,
            aabb_min,
            aabb_max,
        };

        // Write 64B header
        container[..64].copy_from_slice(header.as_bytes());

        // Write lump table
        let mut t_cursor = lump_table_offset;
        for entry in &lump_entries {
            container[t_cursor..t_cursor + 32].copy_from_slice(entry.as_bytes());
            t_cursor += 32;
        }

        // Write payload data
        let payload_start = (lump_table_offset + lump_table_size + 63) & !63;
        container[payload_start..payload_start + final_payload_bytes.len()]
            .copy_from_slice(&final_payload_bytes);

        Ok(container)
    }
}

/// Computes bounding sphere and AABB from position stream if present.
fn compute_bounds_from_payloads(payloads: &[LumpPayload]) -> ([f32; 4], [f32; 3], [f32; 3]) {
    for p in payloads {
        if p.lump_id == LumpId::PositionStream as u16 && p.element_stride == 16 {
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];

            let count = p.element_count as usize;
            for i in 0..count {
                let off = i * 16;
                if off + 12 <= p.data.len() {
                    let x = f32::from_le_bytes([
                        p.data[off],
                        p.data[off + 1],
                        p.data[off + 2],
                        p.data[off + 3],
                    ]);
                    let y = f32::from_le_bytes([
                        p.data[off + 4],
                        p.data[off + 5],
                        p.data[off + 6],
                        p.data[off + 7],
                    ]);
                    let z = f32::from_le_bytes([
                        p.data[off + 8],
                        p.data[off + 9],
                        p.data[off + 10],
                        p.data[off + 11],
                    ]);

                    if x < min[0] {
                        min[0] = x;
                    }
                    if y < min[1] {
                        min[1] = y;
                    }
                    if z < min[2] {
                        min[2] = z;
                    }

                    if x > max[0] {
                        max[0] = x;
                    }
                    if y > max[1] {
                        max[1] = y;
                    }
                    if z > max[2] {
                        max[2] = z;
                    }
                }
            }

            if min[0] <= max[0] {
                let center = [
                    (min[0] + max[0]) * 0.5,
                    (min[1] + max[1]) * 0.5,
                    (min[2] + max[2]) * 0.5,
                ];

                let mut max_dist_sq = 0.0f32;
                for i in 0..count {
                    let off = i * 16;
                    if off + 12 <= p.data.len() {
                        let x = f32::from_le_bytes([
                            p.data[off],
                            p.data[off + 1],
                            p.data[off + 2],
                            p.data[off + 3],
                        ]);
                        let y = f32::from_le_bytes([
                            p.data[off + 4],
                            p.data[off + 5],
                            p.data[off + 6],
                            p.data[off + 7],
                        ]);
                        let z = f32::from_le_bytes([
                            p.data[off + 8],
                            p.data[off + 9],
                            p.data[off + 10],
                            p.data[off + 11],
                        ]);

                        let dx = x - center[0];
                        let dy = y - center[1];
                        let dz = z - center[2];
                        let dist_sq = dx * dx + dy * dy + dz * dz;
                        if dist_sq > max_dist_sq {
                            max_dist_sq = dist_sq;
                        }
                    }
                }

                let radius = max_dist_sq.sqrt();
                return ([center[0], center[1], center[2], radius], min, max);
            }
        }
    }

    ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
}

/// Standard IEEE 802.3 CRC32 calculation.
fn calculate_crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB88320 & (!((crc & 1).wrapping_sub(1))));
        }
    }
    !crc
}
