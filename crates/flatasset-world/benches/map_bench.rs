use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flatasset_world::FlatWorldView;
use std::fs;

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

    eprintln!(
        "\n========================================================================================="
    );
    eprintln!("                  REAL CS 1.6 MAP STORAGE BENCHMARK: BSP30 vs FLATWORLD");
    eprintln!(
        "========================================================================================="
    );
    eprintln!(
        "{:<16} {:<12} {:<12} {:<10} {:<12} {:<12}",
        "Map", "Raw BSP", "Raw FLW", "FLW Ratio", "Zstd BSP", "Zstd FLW"
    );
    eprintln!(
        "-----------------------------------------------------------------------------------------"
    );

    for (name, bsp_path, flw_path) in maps {
        let Ok(bsp_bytes) = fs::read(bsp_path) else {
            eprintln!("{:<16} (files not found, skipping benchmark)", name);
            continue;
        };
        let Ok(flw_bytes) = fs::read(flw_path) else {
            eprintln!("{:<16} (flw not found, skipping benchmark)", name);
            continue;
        };
        let bsp_zst = zstd::encode_all(&bsp_bytes[..], 3).unwrap();
        let flw_zst = zstd::encode_all(&flw_bytes[..], 3).unwrap();

        eprintln!(
            "{:<16} {:<12} {:<12} {:<10.2}x {:<12} {:<12}",
            name,
            format!("{} KB", bsp_bytes.len() / 1024),
            format!("{} KB", flw_bytes.len() / 1024),
            bsp_bytes.len() as f64 / flw_bytes.len() as f64,
            format!("{} KB", bsp_zst.len() / 1024),
            format!("{} KB", flw_zst.len() / 1024),
        );
    }
    eprintln!(
        "=========================================================================================\n"
    );

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

    c.bench_function("inferno_bsp_header_parse", |b| {
        b.iter(|| {
            // Read 544 bytes of BSP header + 15 lump descriptors
            let version = u32::from_le_bytes(bsp_bytes[0..4].try_into().unwrap());
            let mut lump_offsets = [0u32; 15];
            for (i, lump_off) in lump_offsets.iter_mut().enumerate() {
                let off = 4 + i * 8;
                *lump_off = u32::from_le_bytes(bsp_bytes[off..off + 4].try_into().unwrap());
            }
            black_box((version, lump_offsets))
        });
    });

    // 2. Q-BVH4 Raycast vs Map Traversal
    let flw_view = FlatWorldView::new(&flw_bytes).unwrap();
    let qbvh_nodes = flw_view.qbvh_nodes().unwrap();
    let pvs = flw_view.pvs_view().unwrap();
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

    // 3. PVS Visibility: 1000 random cluster lookups
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
}

criterion_group!(benches, bench_real_map);
criterion_main!(benches);
