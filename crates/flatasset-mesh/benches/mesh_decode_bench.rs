use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use flatasset_mesh::{
    FlatAttributeVertex, FlatMeshHeader, FlatMeshLumpEntry, FlatMeshView, FlatPositionVertex,
    FormatSignature, LumpId, LUMP_FLAG_REQUIRED,
};
use zerocopy::IntoBytes;

#[repr(C, align(64))]
struct AlignedBuffer {
    data: Vec<u8>,
    offset: usize,
}

impl AlignedBuffer {
    fn new_zeroed(len: usize) -> Self {
        let data = vec![0u8; len + 64];
        let offset = (64 - (data.as_ptr() as usize % 64)) % 64;
        Self { data, offset }
    }

    fn slice(&self, len: usize) -> &[u8] {
        &self.data[self.offset..self.offset + len]
    }

    fn slice_mut(&mut self, len: usize) -> &mut [u8] {
        &mut self.data[self.offset..self.offset + len]
    }
}

/// Traditional 48-byte Interleaved AoS (Array-of-Structures) Vertex:
/// [Position: 12B, Normal: 12B, Tangent: 16B, TexCoord: 8B] = 48 Bytes
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct AoSVertex48 {
    position: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 4],
    tex_coord: [f32; 2],
}
const _: () = assert!(core::mem::size_of::<AoSVertex48>() == 48);

fn create_test_flatmesh(count: usize) -> (AlignedBuffer, usize) {
    let header_size = core::mem::size_of::<FlatMeshHeader>(); // 64 bytes
    let lump_table_offset = 64usize;
    let lump_count = 3usize;
    let lump_table_size = lump_count * core::mem::size_of::<FlatMeshLumpEntry>();

    let pos_offset = 256usize;
    let pos_bytes = count * core::mem::size_of::<FlatPositionVertex>();

    let attr_offset = pos_offset + pos_bytes;
    let attr_bytes = count * core::mem::size_of::<FlatAttributeVertex>();

    let idx_offset = attr_offset + attr_bytes;
    let idx_bytes = count * core::mem::size_of::<u16>();

    let total_size = (idx_offset + idx_bytes + 63) & !63;

    let mut buf = AlignedBuffer::new_zeroed(total_size);
    let slice = buf.slice_mut(total_size);

    let lump_mask = (1u64 << LumpId::PositionStream as u32)
        | (1u64 << LumpId::AttributeStream as u32)
        | (1u64 << LumpId::IndexBuffer as u32);

    let header = FlatMeshHeader {
        signature: FormatSignature::FLATMESH_V1_0,
        lump_table_offset_div32: (lump_table_offset / 32) as u16,
        lump_count: lump_count as u16,
        flags: 0,
        lump_mask,
        bounding_sphere: [0.0, 0.0, 0.0, 100.0],
        aabb_min: [-100.0; 3],
        aabb_max: [100.0; 3],
    };
    slice[..header_size].copy_from_slice(header.as_bytes());

    let lumps = [
        FlatMeshLumpEntry {
            lump_id: LumpId::PositionStream as u16,
            flags: LUMP_FLAG_REQUIRED,
            version: 1,
            element_stride: 16,
            alignment_log2: 4,
            offset: pos_offset as u64,
            uncompressed_size: pos_bytes as u64,
            element_count: count as u32,
            crc32: 0,
        },
        FlatMeshLumpEntry {
            lump_id: LumpId::AttributeStream as u16,
            flags: LUMP_FLAG_REQUIRED,
            version: 1,
            element_stride: 8,
            alignment_log2: 3,
            offset: attr_offset as u64,
            uncompressed_size: attr_bytes as u64,
            element_count: count as u32,
            crc32: 0,
        },
        FlatMeshLumpEntry {
            lump_id: LumpId::IndexBuffer as u16,
            flags: LUMP_FLAG_REQUIRED,
            version: 1,
            element_stride: 2,
            alignment_log2: 1,
            offset: idx_offset as u64,
            uncompressed_size: idx_bytes as u64,
            element_count: count as u32,
            crc32: 0,
        },
    ];

    slice[lump_table_offset..lump_table_offset + lump_table_size].copy_from_slice(lumps.as_bytes());

    for i in 0..count {
        let pos = FlatPositionVertex {
            position: [i as f32 * 0.1, (i * 2) as f32 * 0.1, (i * 3) as f32 * 0.1],
            cluster_id: (i / 64) as u32,
        };
        let start = pos_offset + i * 16;
        slice[start..start + 16].copy_from_slice(pos.as_bytes());
    }

    (buf, total_size)
}

fn create_gltf_json_buffer(count: usize) -> Vec<u8> {
    let pos_bytes = count * 12;
    let norm_bytes = count * 12;
    let idx_bytes = count * 2;
    let total_bytes = pos_bytes + norm_bytes + idx_bytes;

    let json = format!(
        r#"{{
  "asset": {{"version": "2.0", "generator": "flatmesh-bench"}},
  "buffers": [{{"byteLength": {total_bytes}}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": {pos_bytes}, "byteStride": 12, "target": 34962}},
    {{"buffer": 0, "byteOffset": {pos_bytes}, "byteLength": {norm_bytes}, "byteStride": 12, "target": 34962}},
    {{"buffer": 0, "byteOffset": {}, "byteLength": {idx_bytes}, "target": 34963}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": {count}, "type": "VEC3", "max": [100.0, 100.0, 100.0], "min": [0.0, 0.0, 0.0]}},
    {{"bufferView": 1, "componentType": 5126, "count": {count}, "type": "VEC3", "max": [1.0, 1.0, 1.0], "min": [-1.0, -1.0, -1.0]}},
    {{"bufferView": 2, "componentType": 5123, "count": {count}, "type": "SCALAR", "max": [{max_idx}], "min": [0]}}
  ],
  "meshes": [
    {{
      "name": "benchmark_mesh",
      "primitives": [
        {{
          "attributes": {{"POSITION": 0, "NORMAL": 1}},
          "indices": 2,
          "mode": 4
        }}
      ]
    }}
  ],
  "nodes": [{{"mesh": 0, "name": "Node0"}}],
  "scenes": [{{"nodes": [0]}}],
  "scene": 0
}}"#,
        pos_bytes + norm_bytes,
        max_idx = count.saturating_sub(1)
    );
    json.into_bytes()
}

fn create_aos_vertex_buffer(count: usize) -> Vec<AoSVertex48> {
    let mut vertices = Vec::with_capacity(count);
    for i in 0..count {
        vertices.push(AoSVertex48 {
            position: [i as f32 * 0.1, (i * 2) as f32 * 0.1, (i * 3) as f32 * 0.1],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            tex_coord: [0.5, 0.5],
        });
    }
    vertices
}

const DATASET_SCALES: [(&str, usize); 3] = [
    ("100_verts_prop", 100),
    ("5000_verts_char", 5000),
    ("50000_verts_env", 50000),
];

fn bench_cold_mount_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cold_Mount_Latency");

    for &(scale_name, count) in &DATASET_SCALES {
        let (mesh_buf, total_size) = create_test_flatmesh(count);
        let mesh_slice = mesh_buf.slice(total_size);
        let gltf_json = create_gltf_json_buffer(count);

        group.bench_with_input(
            BenchmarkId::new("FlatMeshView_mount_zero_alloc", scale_name),
            &mesh_slice,
            |b, &slice| {
                b.iter(|| {
                    let view = FlatMeshView::new(black_box(slice)).unwrap();
                    black_box(view.header().signature);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gltf_json_parse", scale_name),
            &gltf_json,
            |b, json_bytes| {
                b.iter(|| {
                    let root = gltf::json::Root::from_slice(black_box(json_bytes)).unwrap();
                    black_box(root.asset.version);
                });
            },
        );
    }

    group.finish();
}

fn bench_stream0_extraction_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stream0_Extraction_Throughput");

    for &(scale_name, count) in &DATASET_SCALES {
        let (mesh_buf, total_size) = create_test_flatmesh(count);
        let mesh_slice = mesh_buf.slice(total_size);
        let view = FlatMeshView::new(mesh_slice).unwrap();
        let payload_bytes = (count * core::mem::size_of::<FlatPositionVertex>()) as u64;

        group.throughput(Throughput::Bytes(payload_bytes));
        group.bench_with_input(
            BenchmarkId::new("FlatMesh_Stream0_Positions", scale_name),
            &view,
            |b, v| {
                b.iter(|| {
                    let positions = v.positions().unwrap();
                    let mut sum = [0.0f32; 3];
                    for p in positions {
                        sum[0] += p.position[0];
                        sum[1] += p.position[1];
                        sum[2] += p.position[2];
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

fn bench_z_prepass_memory_bus_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Z_Prepass_Memory_Bus_Simulation");

    for &(scale_name, count) in &DATASET_SCALES {
        let (mesh_buf, total_size) = create_test_flatmesh(count);
        let mesh_slice = mesh_buf.slice(total_size);
        let view = FlatMeshView::new(mesh_slice).unwrap();
        let aos_buffer = create_aos_vertex_buffer(count);

        let useful_bytes = (count * 12) as u64;
        group.throughput(Throughput::Bytes(useful_bytes));

        // 1. SoA Stream 0 (FlatMesh): 16 bytes stride per vertex
        group.bench_with_input(
            BenchmarkId::new("SoA_Stream0_16B_Stride", scale_name),
            &view,
            |b, v| {
                b.iter(|| {
                    let positions = v.positions().unwrap();
                    let mut sum = [0.0f32; 3];
                    for p in positions {
                        sum[0] += p.position[0];
                        sum[1] += p.position[1];
                        sum[2] += p.position[2];
                    }
                    black_box(sum)
                });
            },
        );

        // 2. AoS Interleaved: 48 bytes stride per vertex (wastes 32B cache per vertex on normals/tangents/uvs)
        group.bench_with_input(
            BenchmarkId::new("AoS_Interleaved_48B_Stride", scale_name),
            &aos_buffer,
            |b, aos| {
                b.iter(|| {
                    let mut sum = [0.0f32; 3];
                    for v in aos.iter() {
                        sum[0] += v.position[0];
                        sum[1] += v.position[1];
                        sum[2] += v.position[2];
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cold_mount_latency,
    bench_stream0_extraction_throughput,
    bench_z_prepass_memory_bus_simulation,
);
criterion_main!(benches);
