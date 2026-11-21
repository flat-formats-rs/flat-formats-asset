use flatasset_mesh::{FlatMeshView, StudioMdlParser};
use std::fs;
use std::path::Path;

fn align_to_64(bytes: &[u8]) -> (Vec<u8>, usize) {
    let mut aligned = Vec::with_capacity(bytes.len() + 64);
    let offset = (64 - (aligned.as_ptr() as usize % 64)) % 64;
    aligned.resize(offset, 0);
    aligned.extend_from_slice(bytes);
    (aligned, offset)
}

#[test]
fn test_studio_mdl_w_ak47_conversion() {
    let mdl_path = Path::new(r"C:\Games\CS 1.6 GoldClient\cstrike\models\w_ak47.mdl");
    if !mdl_path.exists() {
        eprintln!("Skipping test: w_ak47.mdl not found on host");
        return;
    }

    let bytes = fs::read(mdl_path).expect("Failed to read w_ak47.mdl");
    let parser = StudioMdlParser::new(&bytes).expect("Failed to initialize StudioMdlParser");

    let model_data = parser.parse(None).expect("Failed to parse w_ak47.mdl");

    assert!(
        !model_data.positions.is_empty(),
        "Positions should not be empty"
    );
    assert!(
        !model_data.indices.is_empty(),
        "Indices should not be empty"
    );
    assert_eq!(model_data.bones.len(), 1, "w_ak47 has 1 bone");
    assert_eq!(model_data.textures.len(), 1, "w_ak47 has 1 texture");
    assert_eq!(model_data.textures[0].name, "ak47_skin.BMP");

    let flm_bytes = model_data
        .build_flatmesh()
        .expect("Failed to build FlatMesh");
    assert!(!flm_bytes.is_empty());

    let (aligned_buf, offset) = align_to_64(&flm_bytes);
    let aligned_slice = &aligned_buf[offset..offset + flm_bytes.len()];

    let view = FlatMeshView::new(aligned_slice).expect("Failed to mount FlatMeshView");
    let positions = view.positions().expect("Failed to read positions");
    assert_eq!(positions.len(), model_data.positions.len());

    let attributes = view.attributes().expect("Failed to read attributes");
    assert_eq!(attributes.len(), model_data.attributes.len());

    let skin = view
        .skin_weights()
        .expect("Failed to query skin")
        .expect("Skin lump missing");
    assert_eq!(skin.len(), model_data.skin_weights.len());

    let indices = view.indices_u16().expect("Failed to read indices");
    assert_eq!(indices.len(), model_data.indices.len());

    let meshlets = view
        .meshlets()
        .expect("Failed to query meshlets")
        .expect("Meshlets missing");
    assert!(!meshlets.is_empty());
}

#[test]
fn test_studio_mdl_leet_player_conversion() {
    let mdl_path = Path::new(r"C:\Games\CS 1.6 GoldClient\cstrike\models\player\leet\leet.mdl");
    if !mdl_path.exists() {
        eprintln!("Skipping test: leet.mdl not found on host");
        return;
    }

    let bytes = fs::read(mdl_path).expect("Failed to read leet.mdl");
    let parser = StudioMdlParser::new(&bytes).expect("Failed to initialize StudioMdlParser");

    let model_data = parser.parse(None).expect("Failed to parse leet.mdl");

    assert!(!model_data.positions.is_empty());
    assert!(!model_data.indices.is_empty());
    assert_eq!(model_data.bones.len(), 23, "leet has 23 bones");
    assert_eq!(model_data.textures.len(), 13, "leet has 13 textures");

    let flm_bytes = model_data
        .build_flatmesh()
        .expect("Failed to build FlatMesh");
    assert!(!flm_bytes.is_empty());

    let (aligned_buf, offset) = align_to_64(&flm_bytes);
    let aligned_slice = &aligned_buf[offset..offset + flm_bytes.len()];

    let view = FlatMeshView::new(aligned_slice).expect("Failed to mount FlatMeshView");
    assert_eq!(view.positions().unwrap().len(), model_data.positions.len());
    assert_eq!(
        view.attributes().unwrap().len(),
        model_data.attributes.len()
    );
    assert_eq!(
        view.skin_weights().unwrap().unwrap().len(),
        model_data.skin_weights.len()
    );
    assert_eq!(view.indices_u16().unwrap().len(), model_data.indices.len());
    assert!(!view.meshlets().unwrap().unwrap().is_empty());
}

#[test]
fn test_studio_mdl_external_texture_t_mdl() {
    let mdl_path = Path::new(r"C:\Games\CS 1.6 GoldClient\cstrike\models\arcticorange.mdl");
    let t_mdl_path = Path::new(r"C:\Games\CS 1.6 GoldClient\cstrike\models\arcticorangeT.mdl");
    if !mdl_path.exists() || !t_mdl_path.exists() {
        eprintln!("Skipping test: arcticorange*.mdl not found on host");
        return;
    }

    let bytes = fs::read(mdl_path).expect("Failed to read arcticorange.mdl");
    let t_bytes = fs::read(t_mdl_path).expect("Failed to read arcticorangeT.mdl");

    let parser = StudioMdlParser::new(&bytes).expect("Failed to initialize StudioMdlParser");
    let model_data = parser
        .parse(Some(&t_bytes))
        .expect("Failed to parse arcticorange.mdl with T.mdl");

    assert_eq!(
        model_data.textures.len(),
        3,
        "arcticorangeT.mdl has 3 textures"
    );
    assert!(!model_data.positions.is_empty());
    assert!(!model_data.indices.is_empty());

    let flm_bytes = model_data
        .build_flatmesh()
        .expect("Failed to build FlatMesh");
    let (aligned_buf, offset) = align_to_64(&flm_bytes);
    let aligned_slice = &aligned_buf[offset..offset + flm_bytes.len()];

    let view = FlatMeshView::new(aligned_slice).expect("Failed to mount FlatMeshView");
    assert_eq!(view.positions().unwrap().len(), model_data.positions.len());
}
