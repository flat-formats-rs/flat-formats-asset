use flatasset_core::{
    bundle::{FlatBundleBuilder, FlatBundleView},
    identify_flat_asset, FlatAssetKind,
};

#[test]
fn test_flatasset_identification() {
    let dummy_flm = b"FLTM\x01\x00\x00\x00data";
    let dummy_flt = b"FLTT\x01\x00\x00\x00data";
    let dummy_flw = b"FLW\x00\x01\x00\x00\x00data";
    let dummy_fld = b"FLD\x00\x01\x00\x00\x00data";
    let dummy_flb = b"FLTB\x01\x00\x00\x00data";
    let dummy_bin = b"RIFF\x00\x00\x00\x00WAVE";

    assert_eq!(identify_flat_asset(dummy_flm), FlatAssetKind::Mesh);
    assert_eq!(identify_flat_asset(dummy_flt), FlatAssetKind::Texture);
    assert_eq!(identify_flat_asset(dummy_flw), FlatAssetKind::World);
    assert_eq!(identify_flat_asset(dummy_fld), FlatAssetKind::Demo);
    assert_eq!(identify_flat_asset(dummy_flb), FlatAssetKind::Bundle);
    assert_eq!(identify_flat_asset(dummy_bin), FlatAssetKind::Unknown);
    assert_eq!(identify_flat_asset(b""), FlatAssetKind::Unknown);
}

#[test]
fn test_flatbundle_build_and_mount() {
    let mut builder = FlatBundleBuilder::new();

    let asset1 = b"FLTM_test_mesh_data_with_some_bytes_12345";
    let asset2 = b"FLTT_test_texture_dds_dma_bytes_67890";
    let asset3 = b"FLW\x00_test_world_geometry_meshlets";

    builder.add_asset(101, asset1, false);
    builder.add_asset(102, asset2, true);
    builder.add_asset(103, asset3, false);

    let bundle_bytes = builder.build();
    let view = FlatBundleView::new(&bundle_bytes).expect("Failed to mount bundle view");

    assert_eq!(view.header().version, 1);
    assert_eq!(view.header().item_count, 3);
    assert_eq!(view.items().len(), 3);

    // Item 0
    let (item0, bytes0) = view.get_item(0).expect("Item 0 missing");
    assert_eq!(item0.name_hash, 101);
    assert_eq!(item0.compression, 0);
    assert_eq!(bytes0, asset1);

    // Item 1 (compressed)
    let (item1, bytes1) = view.get_item(1).expect("Item 1 missing");
    assert_eq!(item1.name_hash, 102);
    assert_eq!(item1.compression, 1);
    assert_eq!(item1.uncompressed_size, asset2.len() as u64);
    assert!(!bytes1.is_empty());

    // Item 2
    let (item2, bytes2) = view.get_item(2).expect("Item 2 missing");
    assert_eq!(item2.name_hash, 103);
    assert_eq!(item2.compression, 0);
    assert_eq!(bytes2, asset3);

    // Out of bounds
    assert!(view.get_item(3).is_err());
}

#[test]
fn test_flatbundle_rejects_out_of_bounds_item_payload() {
    use flatasset_core::bundle::{FlatBundleHeader, FlatBundleItemDesc};
    use zerocopy::FromBytes;

    let mut builder = FlatBundleBuilder::new();
    builder.add_asset(1, b"abcd", false);
    let mut bytes = builder.build();
    let index_offset = {
        let (header, _) = FlatBundleHeader::ref_from_prefix(&bytes).unwrap();
        header.index_offset as usize
    };
    {
        let (item, _) = <FlatBundleItemDesc>::mut_from_prefix(&mut bytes[index_offset..]).unwrap();
        item.file_offset = u64::MAX;
    }
    assert!(matches!(
        FlatBundleView::new(&bytes),
        Err(flatasset_core::bundle::FlatBundleError::ItemRangeOutOfBounds { .. })
    ));
}

#[test]
fn test_cas_reference_and_pool_deduplication() {
    use flatasset_core::{FlatBundleBuilder, FlatBundleView, FlatCasPool};

    let shared_texture_data = b"FLTT_shared_brick_wall_texture_direct_dma_4096";
    let hash = blake3::hash(shared_texture_data);
    let hash_bytes = *hash.as_bytes();

    // 1. Create a shared pool bundle containing the shared texture
    let mut pool_builder = FlatBundleBuilder::new();
    let mut hash_prefix = [0u8; 8];
    hash_prefix.copy_from_slice(&hash_bytes[..8]);
    let texture_hash_id = u64::from_le_bytes(hash_prefix);
    pool_builder.add_asset(texture_hash_id, shared_texture_data, false);
    let pool_bytes = pool_builder.build();

    let pool_view = FlatBundleView::new(&pool_bytes).expect("mount pool bundle");
    let mut cas_pool = FlatCasPool::new();
    let registered = cas_pool.register_bundle(&pool_view).expect("register pool");
    assert_eq!(registered, 1);
    assert_eq!(cas_pool.len(), 1);

    // 2. Create map_a.flb with CAS reference instead of duplicate inline texture
    let mut map_a_builder = FlatBundleBuilder::new();
    map_a_builder.add_asset(1, b"FLW\x00_map_a_geometry", false);
    map_a_builder.add_cas_reference(hash_bytes, 0x54544C46, shared_texture_data.len() as u64);
    let map_a_bytes = map_a_builder.build();

    let map_a_view = FlatBundleView::new(&map_a_bytes).expect("mount map_a");
    assert_eq!(map_a_view.items().len(), 2);

    let cas_item = &map_a_view.items()[1];
    assert!(cas_item.is_cas_reference());
    assert_eq!(cas_item.byte_length, 0); // 0 bytes inline on disk!
    assert_eq!(cas_item.name_hash, texture_hash_id);

    // 3. Resolve CAS reference from the pool
    let resolved = cas_pool
        .resolve_item(cas_item, &map_a_view)
        .expect("resolve from CAS pool");
    assert_eq!(resolved, shared_texture_data);
}
