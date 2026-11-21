//! FlatWorld static meshlet cluster descriptors and cluster-quantized geometry.
//!
//! Synthesizes GPU-driven Mesh Shaders / Indirect Draw:
//! - 32-byte cache-line friendly cluster descriptors (`FlatWorldMeshlet`)
//! - Cluster-relative snorm16 position quantization (~0.1mm sub-millimeter precision)
//! - Compact 16-byte quantized vertex layout (`QuantizedMeshletVertex`)
//! - 8-bit local micro-triangle indices (`MicroTriangle` = 3 bytes)

use core::mem::size_of;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Maximum vertices per meshlet cluster (hardware-optimal for NV/AMD mesh shading).
pub const MAX_MESHLET_VERTICES: usize = 64;
/// Maximum micro-triangles per meshlet cluster.
pub const MAX_MESHLET_TRIANGLES: usize = 126;

/// 32-byte Meshlet Cluster Descriptor for GPU-driven and indirect rasterization.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct FlatWorldMeshlet {
    /// Bounding sphere: Center [X, Y, Z] and Radius W for frustum culling (offset 0..16)
    pub bounding_sphere: [f32; 4],
    /// Byte offset into the vertex stream (offset 16..20)
    pub vertex_offset: u32,
    /// Byte offset into the micro-triangle index stream (offset 20..24)
    pub triangle_offset: u32,
    /// Material identifier index (offset 24..26)
    pub material_id: u16,
    /// Number of vertices in cluster (max 64) (offset 26..27)
    pub vertex_count: u8,
    /// Number of micro-triangles in cluster (max 126) (offset 27..28)
    pub triangle_count: u8,
    /// Normal cone orientation: Axis [X, Y, Z] as snorm8, and Cutoff angle as snorm8 (offset 28..32)
    pub cone_axis_cutoff: [i8; 4],
}

const _: () = assert!(size_of::<FlatWorldMeshlet>() == 32);

/// 16-byte Cluster-Relative Quantized Vertex layout.
///
/// Compresses 3D world positions down to 6 bytes (`[i16; 3]`) normalized to the
/// meshlet bounding sphere radius: `world_pos = sphere_center + (q_pos / 32767.0) * sphere_radius`.
///
/// Unpacked on GPU with 1 fused-multiply-add (`fma`) instruction.
#[repr(C, align(16))]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Default,
)]
pub struct QuantizedMeshletVertex {
    /// Position offset relative to cluster bounding sphere center: normalized `[-1.0..1.0]` as snorm16.
    pub q_pos: [i16; 3],
    /// Primary texture coordinates encoded as IEEE 754 binary16 (half-float).
    pub uv: [u16; 2],
    /// Surface normal vector packed as snorm8 `[Nx, Ny, Nz, TangentSign]`.
    pub normal: [i8; 4],
    /// Alignment padding to guarantee 16-byte GPU bus alignment.
    pub _pad: [u8; 2],
}

const _: () = assert!(size_of::<QuantizedMeshletVertex>() == 16);
const _: () = assert!(core::mem::align_of::<QuantizedMeshletVertex>() == 16);

/// 3-byte local micro-triangle index tuple (indexing 0..63 within cluster).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Default,
)]
pub struct MicroTriangle {
    /// Three local vertex indices (each in range 0..63).
    pub indices: [u8; 3],
}

const _: () = assert!(size_of::<MicroTriangle>() == 3);

/// Unquantized input vertex for meshlet building.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawMeshletVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub normal: [f32; 3],
}

/// Helper function to convert 32-bit float to 16-bit half-float (IEEE 754 binary16).
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf / NaN
        let m = if mant != 0 { 0x200 } else { 0 };
        return ((sign << 15) | 0x7C00 | m) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow to infinity
        return ((sign << 15) | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        // Underflow / denormal
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let m = (mant | 0x800000) >> (1 - new_exp);
        return ((sign << 15) | (m >> 13)) as u16;
    }

    ((sign << 15) | ((new_exp as u32) << 10) | (mant >> 13)) as u16
}

/// Quantizes an uncompressed 3D position relative to a cluster bounding sphere.
#[inline]
pub fn quantize_cluster_position(pos: [f32; 3], center: [f32; 3], radius: f32) -> [i16; 3] {
    let inv_r = if radius > 1e-6 { 1.0 / radius } else { 0.0 };
    let mut q = [0i16; 3];
    for i in 0..3 {
        let normalized = (pos[i] - center[i]) * inv_r;
        let clamped = normalized.clamp(-1.0, 1.0);
        q[i] = (clamped * 32767.0).round() as i16;
    }
    q
}

/// Dequantizes a cluster-space snorm16 position back to world space.
#[inline]
pub fn dequantize_cluster_position(q_pos: [i16; 3], center: [f32; 3], radius: f32) -> [f32; 3] {
    let factor = radius / 32767.0;
    [
        center[0] + f32::from(q_pos[0]) * factor,
        center[1] + f32::from(q_pos[1]) * factor,
        center[2] + f32::from(q_pos[2]) * factor,
    ]
}

/// File format signature for Meshlet lump payload: b"MSHL"
pub const MESHLET_MAGIC: [u8; 4] = *b"MSHL";
/// Supported format version.
pub const MESHLET_VERSION: u32 = 1;

/// 64-byte Header for the Lump 0 (Meshlets) payload.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct MeshletLumpHeader {
    /// Magic signature: b"MSHL" (offset 0..4)
    pub magic: [u8; 4],
    /// Format version (offset 4..8)
    pub version: u32,
    /// Number of meshlet cluster descriptors (offset 8..12)
    pub meshlet_count: u32,
    /// Total number of quantized vertices (offset 12..16)
    pub vertex_count: u32,
    /// Total number of micro-triangles (offset 16..20)
    pub triangle_count: u32,
    /// Byte offset to `[FlatWorldMeshlet]` array (offset 20..24)
    pub meshlets_offset: u32,
    /// Byte offset to `[QuantizedMeshletVertex]` array (offset 24..28)
    pub vertices_offset: u32,
    /// Byte offset to `[MicroTriangle]` array (offset 28..32)
    pub triangles_offset: u32,
    /// Reserved zero padding to 64 bytes (offset 32..64)
    pub _reserved: [u8; 32],
}

const _: () = assert!(size_of::<MeshletLumpHeader>() == 64);

/// Zero-copy viewer over Lump 0 (`LumpType::Meshlets`).
#[derive(Debug, Clone, Copy)]
pub struct MeshletLumpView<'a> {
    header: &'a MeshletLumpHeader,
    meshlets: &'a [FlatWorldMeshlet],
    vertices: &'a [QuantizedMeshletVertex],
    triangles: &'a [MicroTriangle],
}

impl<'a> MeshletLumpView<'a> {
    /// Parses and validates a raw Lump 0 slice.
    pub fn new(slice: &'a [u8]) -> Option<Self> {
        if slice.len() < size_of::<MeshletLumpHeader>() {
            return None;
        }
        let (header, _) = MeshletLumpHeader::ref_from_prefix(slice).ok()?;
        if header.magic != MESHLET_MAGIC || header.version != MESHLET_VERSION {
            return None;
        }

        let m_start = header.meshlets_offset as usize;
        let m_len = (header.meshlet_count as usize).checked_mul(size_of::<FlatWorldMeshlet>())?;
        let m_end = m_start.checked_add(m_len)?;
        if m_end > slice.len() {
            return None;
        }
        let (meshlets, _) = <[FlatWorldMeshlet]>::ref_from_prefix_with_elems(
            &slice[m_start..m_end],
            header.meshlet_count as usize,
        )
        .ok()?;

        let v_start = header.vertices_offset as usize;
        let v_len =
            (header.vertex_count as usize).checked_mul(size_of::<QuantizedMeshletVertex>())?;
        let v_end = v_start.checked_add(v_len)?;
        if v_end > slice.len() {
            return None;
        }
        let (vertices, _) = <[QuantizedMeshletVertex]>::ref_from_prefix_with_elems(
            &slice[v_start..v_end],
            header.vertex_count as usize,
        )
        .ok()?;

        let t_start = header.triangles_offset as usize;
        let t_len = (header.triangle_count as usize).checked_mul(size_of::<MicroTriangle>())?;
        let t_end = t_start.checked_add(t_len)?;
        if t_end > slice.len() {
            return None;
        }
        let (triangles, _) = <[MicroTriangle]>::ref_from_prefix_with_elems(
            &slice[t_start..t_end],
            header.triangle_count as usize,
        )
        .ok()?;

        Some(Self {
            header,
            meshlets,
            vertices,
            triangles,
        })
    }

    #[inline]
    pub fn header(&self) -> &MeshletLumpHeader {
        self.header
    }

    #[inline]
    pub fn meshlets(&self) -> &'a [FlatWorldMeshlet] {
        self.meshlets
    }

    #[inline]
    pub fn vertices(&self) -> &'a [QuantizedMeshletVertex] {
        self.vertices
    }

    #[inline]
    pub fn triangles(&self) -> &'a [MicroTriangle] {
        self.triangles
    }

    #[inline]
    pub fn meshlet_vertices(&self, m: &FlatWorldMeshlet) -> Option<&'a [QuantizedMeshletVertex]> {
        let start = (m.vertex_offset as usize) / size_of::<QuantizedMeshletVertex>();
        let end = start.checked_add(m.vertex_count as usize)?;
        self.vertices.get(start..end)
    }

    #[inline]
    pub fn meshlet_triangles(&self, m: &FlatWorldMeshlet) -> Option<&'a [MicroTriangle]> {
        let start = (m.triangle_offset as usize) / size_of::<MicroTriangle>();
        let end = start.checked_add(m.triangle_count as usize)?;
        self.triangles.get(start..end)
    }
}

/// Input triangle polygon for meshlet generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputTriangle {
    pub material_id: u16,
    pub vertices: [RawMeshletVertex; 3],
}

/// High-throughput greedy Meshlet Clusterizer.
#[derive(Default)]
pub struct MeshletClusterizer {
    triangles: alloc::vec::Vec<InputTriangle>,
}

impl MeshletClusterizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_triangle(&mut self, material_id: u16, vertices: [RawMeshletVertex; 3]) {
        self.triangles.push(InputTriangle {
            material_id,
            vertices,
        });
    }

    /// Clusters all added triangles and produces a contiguous Lump 0 binary payload.
    pub fn build(self) -> alloc::vec::Vec<u8> {
        if self.triangles.is_empty() {
            let header = MeshletLumpHeader {
                magic: MESHLET_MAGIC,
                version: MESHLET_VERSION,
                meshlet_count: 0,
                vertex_count: 0,
                triangle_count: 0,
                meshlets_offset: 64,
                vertices_offset: 64,
                triangles_offset: 64,
                _reserved: [0; 32],
            };
            return header.as_bytes().to_vec();
        }

        // Sort triangles by material_id to maximize batching within clusters
        let mut sorted_tris = self.triangles;
        sorted_tris.sort_by_key(|t| t.material_id);

        let mut out_meshlets = alloc::vec::Vec::new();
        let mut out_vertices = alloc::vec::Vec::new();
        let mut out_triangles = alloc::vec::Vec::new();

        let mut cur_mat = sorted_tris[0].material_id;
        let mut cluster_verts: alloc::vec::Vec<RawMeshletVertex> = alloc::vec::Vec::new();
        let mut cluster_indices: alloc::vec::Vec<[u8; 3]> = alloc::vec::Vec::new();

        let flush_cluster =
            |mat_id: u16,
             verts: &mut alloc::vec::Vec<RawMeshletVertex>,
             indices: &mut alloc::vec::Vec<[u8; 3]>,
             out_m: &mut alloc::vec::Vec<FlatWorldMeshlet>,
             out_v: &mut alloc::vec::Vec<QuantizedMeshletVertex>,
             out_t: &mut alloc::vec::Vec<MicroTriangle>| {
                if verts.is_empty() || indices.is_empty() {
                    return;
                }

                let positions: alloc::vec::Vec<[f32; 3]> =
                    verts.iter().map(|v| v.position).collect();
                let sphere = compute_bounding_sphere(&positions);
                let center = [sphere[0], sphere[1], sphere[2]];
                let radius = sphere[3].max(0.001);

                let v_offset = (out_v.len() * size_of::<QuantizedMeshletVertex>()) as u32;
                let t_offset = (out_t.len() * size_of::<MicroTriangle>()) as u32;

                // Quantize vertices relative to cluster bounding sphere
                for v in verts.iter() {
                    let q_pos = quantize_cluster_position(v.position, center, radius);
                    let uv = [f32_to_f16(v.uv[0]), f32_to_f16(v.uv[1])];
                    let norm = [
                        (v.normal[0].clamp(-1.0, 1.0) * 127.0).round() as i8,
                        (v.normal[1].clamp(-1.0, 1.0) * 127.0).round() as i8,
                        (v.normal[2].clamp(-1.0, 1.0) * 127.0).round() as i8,
                        127, // tangent sign positive
                    ];
                    out_v.push(QuantizedMeshletVertex {
                        q_pos,
                        uv,
                        normal: norm,
                        _pad: [0; 2],
                    });
                }

                for &tri in indices.iter() {
                    out_t.push(MicroTriangle { indices: tri });
                }

                out_m.push(FlatWorldMeshlet {
                    bounding_sphere: sphere,
                    vertex_offset: v_offset,
                    triangle_offset: t_offset,
                    material_id: mat_id,
                    vertex_count: verts.len() as u8,
                    triangle_count: indices.len() as u8,
                    cone_axis_cutoff: [0, 0, 127, 0], // default upward cone
                });

                verts.clear();
                indices.clear();
            };

        for tri in sorted_tris {
            if tri.material_id != cur_mat {
                flush_cluster(
                    cur_mat,
                    &mut cluster_verts,
                    &mut cluster_indices,
                    &mut out_meshlets,
                    &mut out_vertices,
                    &mut out_triangles,
                );
                cur_mat = tri.material_id;
            }

            // Count how many new unique vertices this triangle would introduce
            let mut new_verts_count = 0;
            for &v in &tri.vertices {
                let existing = cluster_verts.iter().any(|ev| {
                    dist2(ev.position, v.position) < 1e-4
                        && (ev.uv[0] - v.uv[0]).abs() < 1e-3
                        && (ev.uv[1] - v.uv[1]).abs() < 1e-3
                });
                if !existing {
                    new_verts_count += 1;
                }
            }

            // If triangle doesn't fit either by vertex limit or triangle limit, flush cluster FIRST
            if cluster_verts.len() + new_verts_count > MAX_MESHLET_VERTICES
                || cluster_indices.len() >= MAX_MESHLET_TRIANGLES
            {
                flush_cluster(
                    cur_mat,
                    &mut cluster_verts,
                    &mut cluster_indices,
                    &mut out_meshlets,
                    &mut out_vertices,
                    &mut out_triangles,
                );
            }

            // Now all vertices of this triangle are guaranteed to fit in the same cluster
            let mut local_idx = [0u8; 3];
            for (i, &v) in tri.vertices.iter().enumerate() {
                let existing = cluster_verts.iter().position(|ev| {
                    dist2(ev.position, v.position) < 1e-4
                        && (ev.uv[0] - v.uv[0]).abs() < 1e-3
                        && (ev.uv[1] - v.uv[1]).abs() < 1e-3
                });

                if let Some(pos) = existing {
                    local_idx[i] = pos as u8;
                } else {
                    local_idx[i] = cluster_verts.len() as u8;
                    cluster_verts.push(v);
                }
            }

            cluster_indices.push(local_idx);
        }

        flush_cluster(
            cur_mat,
            &mut cluster_verts,
            &mut cluster_indices,
            &mut out_meshlets,
            &mut out_vertices,
            &mut out_triangles,
        );

        // Layout container with 64-byte alignments
        let m_offset = 64u32;
        let m_size = (out_meshlets.len() * size_of::<FlatWorldMeshlet>()) as u32;
        let v_offset = (m_offset + m_size).div_ceil(16) * 16;
        let v_size = (out_vertices.len() * size_of::<QuantizedMeshletVertex>()) as u32;
        let t_offset = (v_offset + v_size).div_ceil(16) * 16;
        let t_size = (out_triangles.len() * size_of::<MicroTriangle>()) as u32;
        let total_size = (t_offset + t_size).div_ceil(64) * 64;

        let header = MeshletLumpHeader {
            magic: MESHLET_MAGIC,
            version: MESHLET_VERSION,
            meshlet_count: out_meshlets.len() as u32,
            vertex_count: out_vertices.len() as u32,
            triangle_count: out_triangles.len() as u32,
            meshlets_offset: m_offset,
            vertices_offset: v_offset,
            triangles_offset: t_offset,
            _reserved: [0; 32],
        };

        let mut buffer = alloc::vec![0u8; total_size as usize];
        buffer[..64].copy_from_slice(header.as_bytes());

        let m_bytes = out_meshlets.as_bytes();
        buffer[m_offset as usize..m_offset as usize + m_bytes.len()].copy_from_slice(m_bytes);

        let v_bytes = out_vertices.as_bytes();
        buffer[v_offset as usize..v_offset as usize + v_bytes.len()].copy_from_slice(v_bytes);

        let t_bytes = out_triangles.as_bytes();
        buffer[t_offset as usize..t_offset as usize + t_bytes.len()].copy_from_slice(t_bytes);

        buffer
    }
}
pub fn compute_bounding_sphere(points: &[[f32; 3]]) -> [f32; 4] {
    if points.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let p0 = points[0];
    // Find point furthest from p0
    let mut p_far = p0;
    let mut max_d2 = 0.0f32;
    for &p in points {
        let d2 = dist2(p0, p);
        if d2 > max_d2 {
            max_d2 = d2;
            p_far = p;
        }
    }

    // Find point furthest from p_far
    let mut p_opp = p_far;
    let mut max_d2 = 0.0f32;
    for &p in points {
        let d2 = dist2(p_far, p);
        if d2 > max_d2 {
            max_d2 = d2;
            p_opp = p;
        }
    }

    // Initial sphere center and radius
    let mut center = [
        (p_far[0] + p_opp[0]) * 0.5,
        (p_far[1] + p_opp[1]) * 0.5,
        (p_far[2] + p_opp[2]) * 0.5,
    ];
    let mut radius = max_d2.sqrt() * 0.5;

    // Expand sphere to enclose any remaining outlier points
    for &p in points {
        let d2 = dist2(center, p);
        if d2 > radius * radius {
            let d = d2.sqrt();
            let new_radius = (radius + d) * 0.5;
            let k = (new_radius - radius) / d;
            center[0] += (p[0] - center[0]) * k;
            center[1] += (p[1] - center[1]) * k;
            center[2] += (p[2] - center[2]) * k;
            radius = new_radius;
        }
    }

    [center[0], center[1], center[2], radius]
}

#[inline]
fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meshlet_struct_sizes() {
        assert_eq!(size_of::<FlatWorldMeshlet>(), 32);
        assert_eq!(size_of::<QuantizedMeshletVertex>(), 16);
        assert_eq!(size_of::<MicroTriangle>(), 3);
    }

    #[test]
    fn test_quantization_submillimeter_precision() {
        let center = [100.0, 200.0, 300.0];
        let radius = 256.0; // 256 units (~6.5 meters)

        let original = [150.25, 230.125, 290.875];
        let q = quantize_cluster_position(original, center, radius);
        let restored = dequantize_cluster_position(q, center, radius);

        for axis in 0..3 {
            let diff = (original[axis] - restored[axis]).abs();
            // Expected precision: 256.0 / 32767.0 ~= 0.0078 units (<0.01 unit error)
            assert!(
                diff < 0.015,
                "axis {axis} precision loss too high: diff = {diff}"
            );
        }
    }

    #[test]
    fn test_bounding_sphere_encloses_all() {
        let points = [
            [-10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, -10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, -10.0],
            [0.0, 0.0, 10.0],
        ];

        let sphere = compute_bounding_sphere(&points);
        let center = [sphere[0], sphere[1], sphere[2]];
        let radius = sphere[3];

        for &p in &points {
            let d = dist2(center, p).sqrt();
            assert!(
                d <= radius + 1e-4,
                "point outside bounding sphere: d={d}, r={radius}"
            );
        }
    }

    #[test]
    fn test_meshlet_clusterizer_and_view_roundtrip() {
        let mut clusterizer = MeshletClusterizer::new();

        // Add 5 triangles with 2 different materials
        for i in 0..5 {
            let mat = if i < 3 { 1 } else { 2 };
            let offset = i as f32 * 10.0;
            let v0 = RawMeshletVertex {
                position: [offset, 0.0, 0.0],
                uv: [0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            };
            let v1 = RawMeshletVertex {
                position: [offset + 5.0, 0.0, 0.0],
                uv: [1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            };
            let v2 = RawMeshletVertex {
                position: [offset + 2.5, 5.0, 0.0],
                uv: [0.5, 1.0],
                normal: [0.0, 0.0, 1.0],
            };
            clusterizer.add_triangle(mat, [v0, v1, v2]);
        }

        let payload = clusterizer.build();
        assert!(payload.len() >= 64);

        let view = MeshletLumpView::new(&payload).expect("valid view");
        assert_eq!(view.header().magic, MESHLET_MAGIC);
        assert_eq!(view.meshlets().len(), 2); // 2 materials -> 2 meshlets
        assert_eq!(view.triangles().len(), 5);

        let m0 = &view.meshlets()[0];
        assert_eq!(m0.material_id, 1);
        assert_eq!(m0.triangle_count, 3);

        let m1 = &view.meshlets()[1];
        assert_eq!(m1.material_id, 2);
        assert_eq!(m1.triangle_count, 2);

        let verts0 = view.meshlet_vertices(m0).unwrap();
        assert_eq!(verts0.len(), m0.vertex_count as usize);

        let tris0 = view.meshlet_triangles(m0).unwrap();
        assert_eq!(tris0.len(), 3);
    }
}
