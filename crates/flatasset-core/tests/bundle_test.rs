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
