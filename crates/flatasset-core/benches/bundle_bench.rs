//! Benchmark comparing monolithic GoldSrc BSP30 vs modular Zero-Copy FlatBundle (`.flb`).
//!
//! Evaluates:
//! 1. Full Container Mount Latency (Zero-Copy Mmap & Bounds Validation).
//! 2. Sub-Asset Discovery and Header Decoding (World / Textures).
//! 3. Cross-Format Storage Footprint Breakdown.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flatasset_core::bundle::FlatBundleView;
use flatasset_texture::FlatTextureView;
use flatasset_world::FlatWorldView;
use std::fs;

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn bsp_lump(b: &[u8], index: usize) -> (usize, usize) {
    let off = u32_at(b, 4 + index * 8) as usize;
    let len = u32_at(b, 8 + index * 8) as usize;
    assert!(off + len <= b.len(), "lump {index} out of bounds");
    (off, len)
}

/// BSP30 full structural validation (the minimum an engine must do to validate lump offsets and counts).
fn bsp_mount_and_validate(b: &[u8]) -> u32 {
    let version = u32_at(b, 0);
    let mut acc = version;
    for i in 0..15 {
        let (off, len) = bsp_lump(b, i);
        acc = acc.wrapping_add((off + len) as u32);
    }
    // Validate nodes lump (lump 5)
    let (node_off, node_len) = bsp_lump(b, 5);
    let node_count = node_len / 24;
    for i in 0..node_count {
        let base = node_off + i * 24;
        let left = i16::from_le_bytes(b[base + 4..base + 6].try_into().unwrap());
        acc = acc.wrapping_add(left as u32);
    }
    // Validate textures lump (lump 2)
    let (tex_off, tex_len) = bsp_lump(b, 2);
    if tex_len >= 4 {
        let num_miptex = u32_at(b, tex_off) as usize;
        for i in 0..num_miptex {
            let m_off = u32_at(b, tex_off + 4 + i * 4) as usize;
            acc = acc.wrapping_add(m_off as u32);
        }
    }
    acc
}

pub fn bench_bundle_vs_bsp(c: &mut Criterion) {
    let bsp_path = "D:/Repo/CS16/hlmsl/test_maps/de_inferno.bsp";
    let flb_raw_path = "d:/Repo/flatworld-rs/de_inferno.flb";
    let flb_bc1_path = "d:/Repo/flatworld-rs/de_inferno_bc1.flb";
    let flb_bc1_zstd_path = "d:/Repo/flatworld-rs/de_inferno_bc1_compressed.flb";

    let Ok(bsp_bytes) = fs::read(bsp_path) else {
        eprintln!("Bench map files not present on disk, skipping bundle_bench.");
        return;
    };
    let Ok(flb_raw_bytes) = fs::read(flb_raw_path) else {
        eprintln!("Bench raw flb file not present on disk, skipping bundle_bench.");
        return;
    };
    let Ok(flb_bc1_bytes) = fs::read(flb_bc1_path) else {
        eprintln!("Bench bc1 flb file not present on disk, skipping bundle_bench.");
        return;
    };
    let Ok(flb_bc1_zstd_bytes) = fs::read(flb_bc1_zstd_path) else {
        eprintln!("Bench compressed flb file not present on disk, skipping bundle_bench.");
        return;
    };

    println!("\n=============================================================================");
    println!("     CS 1.6 FULL MAP CONTAINER: BSP30 vs FLATBUNDLE (.flb)");
    println!("=============================================================================");
    println!(
        "BSP30 Monolith       : {:8.1} KB (Textures + Lighting + QBVH/PVS + Geometry)",
        bsp_bytes.len() as f64 / 1024.0
    );
    println!(
        "FlatBundle (Raw Palette): {:5.1} KB (Direct-DMA Zero-Copy, mmap-ready, 0 alloc)",
        flb_raw_bytes.len() as f64 / 1024.0
    );
    println!(
        "FlatBundle (BC1 GPU Raw): {:5.1} KB ({:.2}x smaller than BSP, Direct-to-VRAM)",
        flb_bc1_bytes.len() as f64 / 1024.0,
        bsp_bytes.len() as f64 / flb_bc1_bytes.len() as f64
    );
    println!(
        "FlatBundle (BC1 + Zstd) : {:5.1} KB ({:.2}x smaller than BSP!)",
        flb_bc1_zstd_bytes.len() as f64 / 1024.0,
        bsp_bytes.len() as f64 / flb_bc1_zstd_bytes.len() as f64
    );
    println!("-----------------------------------------------------------------------------\n");

    // 1. Mount & Validate Full Map Container
    c.bench_function("flb_bc1_mount_zero_alloc", |b| {
        b.iter(|| {
            let view = FlatBundleView::new(black_box(&flb_bc1_bytes)).unwrap();
            black_box(view.items().len())
        });
    });

    c.bench_function("bsp_mount_and_validate_15_lumps", |b| {
        b.iter(|| {
            let acc = bsp_mount_and_validate(black_box(&bsp_bytes));
            black_box(acc)
        });
    });

    // 2. Sub-Asset Discovery and Reader Mount
    c.bench_function("flb_mount_world_and_tex_views", |b| {
        b.iter(|| {
            let bundle = FlatBundleView::new(black_box(&flb_raw_bytes)).unwrap();
            let (_, world_bytes) = bundle.get_item(0).unwrap();
            let (_, tex_bytes) = bundle.get_item(1).unwrap();

            let world_view = FlatWorldView::new(world_bytes).unwrap();
            let tex_view = FlatTextureView::new(tex_bytes).unwrap();

            black_box((world_view.header().lump_count, tex_view.header().mip_levels))
        });
    });

    c.bench_function("bsp_decode_texture_directory", |b| {
        b.iter(|| {
            let (tex_off, _) = bsp_lump(black_box(&bsp_bytes), 2);
            let num_miptex = u32_at(&bsp_bytes, tex_off) as usize;
            let mut valid_offsets = 0;
            for i in 0..num_miptex {
                let off = u32_at(&bsp_bytes, tex_off + 4 + i * 4);
                if off != 0xFFFFFFFF && off > 0 {
                    valid_offsets += 1;
                }
            }
            black_box(valid_offsets)
        });
    });
}

criterion_group!(benches, bench_bundle_vs_bsp);
criterion_main!(benches);
