use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flatasset_world::{
    header::LumpType, pvs::PvsView, qbvh::quantize_aabb, FlatWorldBuilder, FlatWorldView,
    QBvh4Node, QBVH_LEAF_MASK,
};
use zerocopy::IntoBytes;

fn bench_qbvh(c: &mut Criterion) {
    let world_min = [-1024.0, -1024.0, -512.0];
    let world_max = [1024.0, 1024.0, 512.0];

    let mut node = QBvh4Node::empty();
    for i in 0..4 {
        let offset = i as f32 * 50.0;
        let (q_min, q_max) = quantize_aabb(
            [-100.0 + offset, -100.0, -10.0],
            [100.0 + offset, 100.0, 10.0],
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

    let origin = [0.0, 0.0, 100.0];
    let inv_dir = [1e30, 1e30, -1.0];

    c.bench_function("qbvh4_raycast_4children", |b| {
        b.iter(|| {
            black_box(node.intersect_ray(
                black_box(origin),
                black_box(inv_dir),
                black_box(0.0),
                black_box(1000.0),
                black_box(world_min),
                black_box(world_max),
            ))
        });
    });

    // PVS benchmark
    let cluster_count = 1024usize;
    let row_stride = cluster_count.div_ceil(8);
    let pvs_data = vec![0x55u8; cluster_count * row_stride];
    let pvs = PvsView::new(cluster_count, &pvs_data).unwrap();

    c.bench_function("pvs_is_visible_1024_clusters", |b| {
        b.iter(|| black_box(pvs.is_visible(black_box(128), black_box(512))));
    });

    // FlatWorldView mount benchmark
    let qbvh_nodes = [node];
    let flw_bytes = FlatWorldBuilder::new(world_min, world_max)
        .add_lump(
            LumpType::QBvh4,
            qbvh_nodes.len() as u64,
            size_of::<QBvh4Node>() as u32,
            qbvh_nodes.as_bytes().to_vec(),
        )
        .build();

    c.bench_function("flatworld_view_mount", |b| {
        b.iter(|| black_box(FlatWorldView::new(black_box(&flw_bytes)).unwrap()));
    });
}

criterion_group!(benches, bench_qbvh);
criterion_main!(benches);
