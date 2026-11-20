//! Converts a GoldSrc BSP30 map into a FlatWorld `.flw` with a Q-BVH4 built
//! from the BSP node tree and a PVS matrix decoded from the visibility lump.
//!
//! Usage: `cargo run --release --example bsp_to_flw -- <in.bsp> <out.flw>`

use core::mem::size_of;
use std::{env, fs};

use flatasset_world::header::LumpType;
use flatasset_world::qbvh::{QBvh4Node, QBVH_LEAF_MASK};
use flatasset_world::FlatWorldBuilder;
use zerocopy::IntoBytes;

const LUMP_VISIBILITY: usize = 4;
const LUMP_NODES: usize = 5;
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

fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn lump(b: &[u8], index: usize) -> (usize, usize) {
    let off = u32_at(b, 4 + index * 8) as usize;
    let len = u32_at(b, 8 + index * 8) as usize;
    assert!(off + len <= b.len(), "lump {index} out of file bounds");
    (off, len)
}

fn leaf_bounds(b: &[u8], leaf_off: usize, index: usize) -> ([f32; 3], [f32; 3]) {
    let base = leaf_off + index * 28;
    (
        [
            f32::from(i16_at(b, base + 8)),
            f32::from(i16_at(b, base + 10)),
            f32::from(i16_at(b, base + 12)),
        ],
        [
            f32::from(i16_at(b, base + 14)),
            f32::from(i16_at(b, base + 16)),
            f32::from(i16_at(b, base + 18)),
        ],
    )
}

fn node_bounds(b: &[u8], node_off: usize, index: usize) -> ([f32; 3], [f32; 3]) {
    let base = node_off + index * 24;
    (
        [
            f32::from(i16_at(b, base + 8)),
            f32::from(i16_at(b, base + 10)),
            f32::from(i16_at(b, base + 12)),
        ],
        [
            f32::from(i16_at(b, base + 14)),
            f32::from(i16_at(b, base + 16)),
            f32::from(i16_at(b, base + 18)),
        ],
    )
}

/// Quake PVS RLE: high bit = zero run, otherwise copy the next `n` bytes.
fn decode_pvs(src: &[u8], row: &mut [u8]) {
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let (bsp_path, out_path) = (&args[1], &args[2]);
    let b = fs::read(bsp_path).expect("read BSP");

    assert_eq!(u32_at(&b, 0), 30, "expected BSP30");

    let (model_off, _) = lump(&b, LUMP_MODELS);
    let world_min = [
        f32_at(&b, model_off),
        f32_at(&b, model_off + 4),
        f32_at(&b, model_off + 8),
    ];
    let world_max = [
        f32_at(&b, model_off + 12),
        f32_at(&b, model_off + 16),
        f32_at(&b, model_off + 20),
    ];
    let root = i32_at(&b, model_off + 36) as usize;

    let (node_off, node_len) = lump(&b, LUMP_NODES);
    let node_count = node_len / 24;
    let (leaf_off, leaf_len) = lump(&b, LUMP_LEAVES);
    let leaf_count = leaf_len / 28;

    // Collapse two BSP levels per Q-BVH4 node so all four child slots carry geometry.
    let bounds_of = |child: i16| {
        if child >= 0 {
            node_bounds(&b, node_off, child as usize)
        } else {
            leaf_bounds(&b, leaf_off, (-child - 1) as usize)
        }
    };

    let mut nodes: Vec<QBvh4Node> = Vec::with_capacity(node_count / 2);
    nodes.push(QBvh4Node::empty());
    let mut work: Vec<(usize, usize)> = vec![(0, root)];
    let mut cursor = 0;
    while cursor < work.len() {
        let (slot_idx, bsp_node) = work[cursor];
        cursor += 1;
        let mut grandchildren: Vec<i16> = Vec::with_capacity(4);
        for slot in 0..2 {
            let child = i16_at(&b, node_off + bsp_node * 24 + 4 + slot * 2);
            if child >= 0 {
                grandchildren.push(i16_at(&b, node_off + child as usize * 24 + 4));
                grandchildren.push(i16_at(&b, node_off + child as usize * 24 + 6));
            } else {
                grandchildren.push(child);
            }
        }
        for (slot, &g) in grandchildren.iter().enumerate() {
            let (cmin, cmax) = bounds_of(g);
            let (qmin, qmax) = flatasset_world::quantize_aabb(cmin, cmax, world_min, world_max);
            nodes[slot_idx].min_x[slot] = qmin[0];
            nodes[slot_idx].min_y[slot] = qmin[1];
            nodes[slot_idx].min_z[slot] = qmin[2];
            nodes[slot_idx].max_x[slot] = qmax[0];
            nodes[slot_idx].max_y[slot] = qmax[1];
            nodes[slot_idx].max_z[slot] = qmax[2];
            nodes[slot_idx].children[slot] = if g >= 0 {
                let idx = nodes.len();
                nodes.push(QBvh4Node::empty());
                work.push((idx, g as usize));
                idx as u32
            } else {
                QBVH_LEAF_MASK | ((-g - 1) as u32)
            };
        }
    }

    let row_bytes = leaf_count.div_ceil(8);
    let mut pvs = vec![0u8; leaf_count * row_bytes];
    let (vis_off, vis_len) = lump(&b, LUMP_VISIBILITY);
    if vis_len > 0 {
        for leaf in 0..leaf_count {
            let vis = i32_at(&b, leaf_off + leaf * 28 + 4);
            let row = &mut pvs[leaf * row_bytes..(leaf + 1) * row_bytes];
            if vis >= 0 && (vis as usize) < vis_len {
                decode_pvs(&b[vis_off + vis as usize..vis_off + vis_len], row);
            }
            row[leaf / 8] |= 1 << (leaf % 8);
        }
    } else {
        pvs.fill(0xFF);
    }

    let word_sparse_pvs = flatasset_world::pvs::encode_word_sparse_pvs(leaf_count, &pvs);
    eprintln!(
        "PVS: dense {} KB -> word-sparse {} KB ({:.2}x reduction)",
        pvs.len() / 1024,
        word_sparse_pvs.len() / 1024,
        pvs.len() as f64 / word_sparse_pvs.len() as f64,
    );

    let flw = FlatWorldBuilder::new(world_min, world_max)
        .add_lump(
            LumpType::QBvh4,
            nodes.len() as u64,
            size_of::<QBvh4Node>() as u32,
            nodes.as_bytes().to_vec(),
        )
        .add_lump(
            LumpType::VisibilityBitmask,
            leaf_count as u64,
            1,
            word_sparse_pvs,
        )
        .build();

    if let Some(parent) = std::path::Path::new(out_path).parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    fs::write(out_path, &flw).expect("write FLW");

    eprintln!(
        "{bsp_path}: root={root}, nodes={}, leaves={}, {:.1} MB -> {out_path} ({:.1} MB)",
        nodes.len(),
        leaf_count,
        b.len() as f64 / 1e6,
        flw.len() as f64 / 1e6,
    );
}
