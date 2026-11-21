use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flatasset_world::{FlatWorldView, QBvh4Node, QBVH_EMPTY_CHILD, QBVH_LEAF_MASK};
use std::fs;

const LUMP_VIS: usize = 4;
const LUMP_PLANES: usize = 1;
const LUMP_NODES: usize = 5;
const LUMP_CLIPNODES: usize = 9;
const LUMP_LEAVES: usize = 10;
const LUMP_MODELS: usize = 14;

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn i32_at(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn i16_at(b: &[u8], off: usize) -> i16 {
    i16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}

fn bsp_lump(b: &[u8], index: usize) -> (usize, usize) {
    let off = u32_at(b, 4 + index * 8) as usize;
    let len = u32_at(b, 8 + index * 8) as usize;
    assert!(off + len <= b.len(), "lump {index} out of bounds");
    (off, len)
}

fn flw_lump_sizes(b: &[u8]) -> Vec<(u32, u64, u64)> {
    let count = u32_at(b, 12) as usize;
    (0..count)
        .map(|i| {
            let base = 64 + i * 64;
            (
                u32_at(b, base),
                u64::from_le_bytes(b[base + 16..base + 24].try_into().unwrap()),
                u64::from_le_bytes(b[base + 24..base + 32].try_into().unwrap()),
            )
        })
        .collect()
}

fn flw_lump_name(t: u32) -> &'static str {
    match t {
        0 => "Meshlets",
        1 => "QBvh4",
        2 => "CollisionHulls",
        3 => "VisibilityBitmask",
        4 => "EntityGraph",
        5 => "MaterialManifest",
        6 => "Lightmaps",
        7 => "NavMesh",
        8 => "Script",
        _ => "Unknown",
    }
}

/// Slab test against an integer BSP node AABB.
fn slab_hit(origin: [f32; 3], inv_dir: [f32; 3], mins: [f32; 3], maxs: [f32; 3]) -> bool {
    let mut tmin = 0.0f32;
    let mut tmax = f32::MAX;
    for axis in 0..3 {
        let t1 = (mins[axis] - origin[axis]) * inv_dir[axis];
        let t2 = (maxs[axis] - origin[axis]) * inv_dir[axis];
        let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        tmin = tmin.max(lo);
        tmax = tmax.min(hi);
        if tmin > tmax {
            return false;
        }
    }
    true
}

/// AABB-clipped traversal of the BSP30 node tree, counting visited leaves.
fn bsp_raycast(b: &[u8], origin: [f32; 3], inv_dir: [f32; 3]) -> (u32, u32) {
    let (node_off, node_len) = bsp_lump(b, LUMP_NODES);
    let _ = bsp_lump(b, LUMP_LEAVES);
    let (model_off, _) = bsp_lump(b, LUMP_MODELS);
    let root = i32_at(b, model_off + 36);
    assert!(root >= 0 && (root as usize) < node_len / 24);

    let mut stack = [0i32; 256];
    let mut sp = 1usize;
    stack[0] = root;
    let mut leaves = 0u32;
    let mut tested = 0u32;
    while sp > 0 {
        sp -= 1;
        let node = stack[sp];
        if node < 0 {
            leaves += 1;
            continue;
        }
        tested += 1;
        let base = node_off + node as usize * 24;
        let mins = [
            f32::from(i16_at(b, base + 8)),
            f32::from(i16_at(b, base + 10)),
            f32::from(i16_at(b, base + 12)),
        ];
        let maxs = [
            f32::from(i16_at(b, base + 14)),
            f32::from(i16_at(b, base + 16)),
            f32::from(i16_at(b, base + 18)),
        ];
        if !slab_hit(origin, inv_dir, mins, maxs) {
            continue;
        }
        let left = i16_at(b, base + 4) as i32;
        let right = i16_at(b, base + 6) as i32;
        if sp + 2 <= stack.len() {
            stack[sp] = left;
            stack[sp + 1] = right;
            sp += 2;
        }
    }
    (tested, leaves)
}

/// Full structural validation of the BSP (what a loader must do before use).
/// Same scope as FLW mount: header + lump-table bounds checks only.
fn bsp_mount_table_only(b: &[u8]) -> u32 {
    let mut acc = u32_at(b, 0);
    for i in 0..15 {
        let (off, len) = bsp_lump(b, i);
        acc = acc.wrapping_add((off + len) as u32);
    }
    acc
}

fn bsp_mount_validate(b: &[u8]) -> u32 {
    let version = u32_at(b, 0);
    let mut acc = version;
    for i in 0..15 {
        let (off, len) = bsp_lump(b, i);
        acc = acc.wrapping_add((off + len) as u32);
    }
    let (node_off, node_len) = bsp_lump(b, LUMP_NODES);
    let node_count = node_len / 24;
    for i in 0..node_count {
        let base = node_off + i * 24;
        for slot in 0..2 {
            let child = i16_at(b, base + 4 + slot * 2);
            if child >= 0 {
                acc = acc.wrapping_add(child as u32 % (node_count as u32 + 1));
            }
            acc = acc.wrapping_add(i16_at(b, base + 8) as u32);
        }
    }
    let (leaf_off, leaf_len) = bsp_lump(b, LUMP_LEAVES);
    for i in 0..leaf_len / 28 {
        acc = acc.wrapping_add(i32_at(b, leaf_off + i * 28 + 4) as u32);
        acc = acc.wrapping_add(i16_at(b, leaf_off + i * 28 + 8) as u32);
    }
    acc
}

/// Quake PVS RLE row decode (high bit = zero run, else copy `n` bytes).
fn decode_pvs_row(src: &[u8], row: &mut [u8]) {
    let mut in_pos = 0;
    let mut out_pos = 0;
    while in_pos < src.len() && out_pos < row.len() {
        let b = src[in_pos];
        in_pos += 1;
        if b & 0x80 != 0 {
            out_pos += (b & 0x7F) as usize;
        } else {
            let count = b as usize;
            let take = count.min(row.len() - out_pos).min(src.len() - in_pos);
            row[out_pos..out_pos + take].copy_from_slice(&src[in_pos..in_pos + take]);
            out_pos += count;
            in_pos += count;
        }
    }
}

fn pvs_row(b: &[u8], leaf: usize, row: &mut [u8]) {
    row.fill(0);
    let (leaf_off, _) = bsp_lump(b, LUMP_LEAVES);
    let (vis_off, vis_len) = bsp_lump(b, LUMP_VIS);
    let vis = i32_at(b, leaf_off + leaf * 28 + 4);
    if vis >= 0 && (vis as usize) < vis_len {
        decode_pvs_row(&b[vis_off + vis as usize..vis_off + vis_len], row);
    }
    row[leaf / 8] |= 1 << (leaf % 8);
}

/// Full AABB-clipped descent of the Q-BVH4 tree, counting visited leaves.
fn qbvh_raycast(
    nodes: &[QBvh4Node],
    origin: [f32; 3],
    inv_dir: [f32; 3],
    world_min: [f32; 3],
    world_max: [f32; 3],
) -> (u32, u32) {
    let mut stack = [0u32; 256];
    let mut sp = 1usize;
    let mut leaves = 0u32;
    let mut tested = 0u32;
    while sp > 0 {
        sp -= 1;
        tested += 1;
        let node = &nodes[stack[sp] as usize];
        let (mask, _) = node.intersect_ray(origin, inv_dir, 0.0, 1e30, world_min, world_max);
        for (slot, &child) in node.children.iter().enumerate() {
            if mask & (1 << slot) == 0 {
                continue;
            }
            if child == QBVH_EMPTY_CHILD {
                continue;
            }
            if child & QBVH_LEAF_MASK != 0 {
                leaves += 1;
            } else if sp < stack.len() {
                stack[sp] = child;
                sp += 1;
            }
        }
    }
    (tested, leaves)
}

fn bench_real_map(c: &mut Criterion) {
    let maps = [
        (
            "de_inferno",
            "D:/Repo/CS16/hlmsl/test_maps/de_inferno.bsp",
            "D:/Repo/flatworld-rs/de_inferno.flw",
        ),
        (
            "de_mirage",
            "D:/Repo/CS16/hlmsl/test_maps/de_mirage.bsp",
            "D:/Repo/flatworld-rs/de_mirage.flw",
        ),
        (
            "regamedll_test",
            "D:/Repo/GoldSrc.rs/references/regamedll/regamedll/extra/Test Maps/regamedll_test_map_v8.bsp",
            "D:/Repo/flatworld-rs/regamedll_test.flw",
        ),
    ];

    eprintln!("\n=============================================================================");
    eprintln!("     CS 1.6 MAP STORAGE: BSP30 vs FLATWORLD (same-semantics subsets)");
    eprintln!("=============================================================================");
    eprintln!(
        "{:<16} {:<24} {:<22} {:<22}",
        "Map", "PVS (BSP RLE / FLW dense)", "Spatial (BSP / FLW QBVH)", "FLW total / BSP total"
    );
    eprintln!("-----------------------------------------------------------------------------");

    for (name, bsp_path, flw_path) in maps {
        let (Ok(bsp_bytes), Ok(flw_bytes)) = (fs::read(bsp_path), fs::read(flw_path)) else {
            eprintln!("{:<16} (assets missing, skipped)", name);
            continue;
        };
        let (_, vis_len) = bsp_lump(&bsp_bytes, LUMP_VIS);
        let spatial_bsp: usize = [
            LUMP_NODES,
            LUMP_PLANES,
            LUMP_CLIPNODES,
            LUMP_LEAVES,
            LUMP_MODELS,
        ]
        .iter()
        .map(|&i| bsp_lump(&bsp_bytes, i).1)
        .sum();
        let lumps = flw_lump_sizes(&flw_bytes);
        let flw_pvs = lumps.iter().find(|l| l.0 == 3).map(|l| l.2).unwrap_or(0);
        let flw_qbvh = lumps.iter().find(|l| l.0 == 1).map(|l| l.2).unwrap_or(0);
        eprintln!(
            "{:<16} {:<24} {:<22} {:<22}",
            name,
            format!(
                "{} KB / {} KB ({:.1}x)",
                vis_len / 1024,
                flw_pvs / 1024,
                vis_len as f64 / flw_pvs.max(1) as f64
            ),
            format!(
                "{} KB / {} KB ({:.2}x)",
                spatial_bsp / 1024,
                flw_qbvh / 1024,
                spatial_bsp as f64 / flw_qbvh.max(1) as f64
            ),
            format!(
                "{} KB / {} KB ({:.2}x)",
                flw_bytes.len() / 1024,
                bsp_bytes.len() / 1024,
                bsp_bytes.len() as f64 / flw_bytes.len() as f64
            ),
        );
        eprintln!(
            "  FLW contains only: {}",
            lumps
                .iter()
                .map(|l| format!("{} {} KB", flw_lump_name(l.0), l.2 / 1024))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    eprintln!("  NOTE: textures/lightmaps/geometry/entities have no FLW counterpart yet,");
    eprintln!("        so the total ratio is NOT a full-format comparison.\n");

    let Ok(bsp_bytes) = fs::read("D:/Repo/CS16/hlmsl/test_maps/de_inferno.bsp") else {
        eprintln!("Bench map files not present on disk, skipping detailed criterion benchmarks.");
        return;
    };
    let Ok(flw_bytes) = fs::read("D:/Repo/flatworld-rs/de_inferno.flw") else {
        eprintln!("Bench flw file not present on disk, skipping detailed criterion benchmarks.");
        return;
    };

    // 1. Raw Container Mount / Ingestion Benchmark
    c.bench_function("inferno_flw_mount_zero_alloc", |b| {
        b.iter(|| black_box(FlatWorldView::new(black_box(&flw_bytes)).unwrap()));
    });

    c.bench_function("inferno_bsp_mount_table_only", |b| {
        b.iter(|| black_box(bsp_mount_table_only(black_box(&bsp_bytes))));
    });

    c.bench_function("inferno_bsp_mount_validate", |b| {
        b.iter(|| black_box(bsp_mount_validate(black_box(&bsp_bytes))));
    });

    // 2. Raycast: FLW QBVH4 vs BSP node-tree traversal
    let flw_view = FlatWorldView::new(&flw_bytes).unwrap();
    let qbvh_nodes = flw_view.qbvh_nodes().unwrap();
    let pvs = flw_view.world_pvs().unwrap();
    let header = flw_view.header();

    let origin = [0.0, 0.0, 100.0];
    let inv_dir = [1e30, 1e30, -1.0];

    c.bench_function("inferno_qbvh_raycast_4children", |b| {
        b.iter(|| {
            black_box(qbvh_nodes[0].intersect_ray(
                black_box(origin),
                black_box(inv_dir),
                black_box(0.0),
                black_box(1000.0),
                black_box(header.bounds_min),
                black_box(header.bounds_max),
            ))
        });
    });

    let qc = qbvh_raycast(
        qbvh_nodes,
        origin,
        inv_dir,
        header.bounds_min,
        header.bounds_max,
    );
    let bc = bsp_raycast(&bsp_bytes, origin, inv_dir);
    eprintln!("ray origin {origin:?} dir {:?}: QBVH tested {} nodes / {} leaves; BSP tested {} nodes / {} leaves",
        [1.0, 1.0, -1.0], qc.0, qc.1, bc.0, bc.1);

    c.bench_function("inferno_qbvh_raycast_traversal", |b| {
        b.iter(|| {
            black_box(qbvh_raycast(
                black_box(qbvh_nodes),
                black_box(origin),
                black_box(inv_dir),
                black_box(header.bounds_min),
                black_box(header.bounds_max),
            ))
        });
    });

    c.bench_function("inferno_bsp_raycast_traversal", |b| {
        b.iter(|| {
            black_box(bsp_raycast(
                black_box(&bsp_bytes),
                black_box(origin),
                black_box(inv_dir),
            ))
        });
    });

    let (_, leaf_len) = bsp_lump(&bsp_bytes, LUMP_LEAVES);
    let leaf_count = leaf_len / 28;
    let row_bytes = leaf_count.div_ceil(8);
    let mut row = vec![0u8; row_bytes];

    // 3. PVS: FLW dense lookup vs BSP RLE row decode + bit test
    let clusters = pvs.cluster_count();
    c.bench_function("inferno_pvs_lookup_1000_pairs", |b| {
        b.iter(|| {
            let mut visible_count = 0;
            for i in 0..1000 {
                let src = (i * 3) % clusters;
                let dst = (i * 7) % clusters;
                if pvs.is_visible(src, dst) {
                    visible_count += 1;
                }
            }
            black_box(visible_count)
        });
    });

    // Worst case for RLE: every lookup decodes its row.
    c.bench_function("inferno_bsp_pvs_lookup_1000_fresh_rows", |b| {
        b.iter(|| {
            let mut visible_count = 0;
            for i in 0..1000 {
                let src = (i * 3) % leaf_count;
                let dst = (i * 7) % leaf_count;
                pvs_row(&bsp_bytes, src, &mut row);
                if row[dst / 8] & (1 << (dst % 8)) != 0 {
                    visible_count += 1;
                }
            }
            black_box(visible_count)
        });
    });

    // Realistic BSP usage: decode a row once, query 32 targets (Quake's leafvisible cache).
    c.bench_function("inferno_bsp_pvs_lookup_1024_grouped", |b| {
        b.iter(|| {
            let mut visible_count = 0;
            for src in 0..32 {
                pvs_row(&bsp_bytes, src % leaf_count, &mut row);
                for t in 0..32 {
                    let dst = (src * 32 + t) % leaf_count;
                    if row[dst / 8] & (1 << (dst % 8)) != 0 {
                        visible_count += 1;
                    }
                }
            }
            black_box(visible_count)
        });
    });
}

criterion_group!(benches, bench_real_map);
criterion_main!(benches);
