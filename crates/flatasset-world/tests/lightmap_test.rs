use flatasset_world::header::LumpType;
use flatasset_world::lightmap::{
    FaceLightmapInput, LightmapAtlasView, LightmapPacker, DEFAULT_ATLAS_WIDTH, DEFAULT_LIGHTGAMMA,
    DEFAULT_LIGHTSCALE, LIGHTMAP_MAGIC,
};
use flatasset_world::{FlatWorldBuilder, FlatWorldView};
use std::fs;

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}

fn i32_at(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn lump(b: &[u8], index: usize) -> (usize, usize) {
    let off = u32_at(b, 4 + index * 8) as usize;
    let len = u32_at(b, 8 + index * 8) as usize;
    assert!(off + len <= b.len(), "lump {index} out of bounds");
    (off, len)
}

#[test]
fn test_inferno_bsp_lightmap_extraction_and_roundtrip() {
    let bsp_path = "D:/Repo/CS16/hlmsl/test_maps/de_inferno.bsp";
    let Ok(b) = fs::read(bsp_path) else {
        eprintln!("de_inferno.bsp not found, skipping real BSP lightmap test");
        return;
    };

    let (face_off, face_len) = lump(&b, 7);
    let (texinfo_off, _) = lump(&b, 6);
    let (vert_off, _) = lump(&b, 3);
    let (edge_off, _) = lump(&b, 12);
    let (surfedge_off, _) = lump(&b, 13);
    let (light_off, light_len) = lump(&b, 8);

    assert!(light_len > 0, "inferno should have lighting lump");
    let face_count = face_len / 20;

    let lighting = &b[light_off..light_off + light_len];
    let mut inputs = Vec::new();

    for face_idx in 0..face_count {
        let base = face_off + face_idx * 20;
        let lightofs = i32_at(&b, base + 16);
        if lightofs < 0 || (lightofs as usize) >= light_len {
            continue;
        }

        let firstedge = u32_at(&b, base + 4) as usize;
        let numedges = u16_at(&b, base + 8) as usize;
        let texinfo_id = u16_at(&b, base + 10) as usize;

        let ti_base = texinfo_off + texinfo_id * 40;
        let s_vec = [
            f32_at(&b, ti_base),
            f32_at(&b, ti_base + 4),
            f32_at(&b, ti_base + 8),
        ];
        let s_shift = f32_at(&b, ti_base + 12);
        let t_vec = [
            f32_at(&b, ti_base + 16),
            f32_at(&b, ti_base + 20),
            f32_at(&b, ti_base + 24),
        ];
        let t_shift = f32_at(&b, ti_base + 28);

        let mut s_min = f32::MAX;
        let mut s_max = f32::MIN;
        let mut t_min = f32::MAX;
        let mut t_max = f32::MIN;

        for e in 0..numedges {
            let se = i32_at(&b, surfedge_off + (firstedge + e) * 4);
            let vert_idx = if se >= 0 {
                u16_at(&b, edge_off + (se as usize) * 4) as usize
            } else {
                u16_at(&b, edge_off + ((-se) as usize) * 4 + 2) as usize
            };

            let px = f32_at(&b, vert_off + vert_idx * 12);
            let py = f32_at(&b, vert_off + vert_idx * 12 + 4);
            let pz = f32_at(&b, vert_off + vert_idx * 12 + 8);

            let s = px * s_vec[0] + py * s_vec[1] + pz * s_vec[2] + s_shift;
            let t = px * t_vec[0] + py * t_vec[1] + pz * t_vec[2] + t_shift;

            s_min = s_min.min(s);
            s_max = s_max.max(s);
            t_min = t_min.min(t);
            t_max = t_max.max(t);
        }

        let b_s_min = (s_min / 16.0).floor();
        let b_s_max = (s_max / 16.0).ceil();
        let b_t_min = (t_min / 16.0).floor();
        let b_t_max = (t_max / 16.0).ceil();

        let w = ((b_s_max - b_s_min) as i32 + 1).max(1) as u16;
        let h = ((b_t_max - b_t_min) as i32 + 1).max(1) as u16;

        let start = lightofs as usize;
        let req_len = (w as usize) * (h as usize) * 3;
        if start + req_len <= lighting.len() {
            inputs.push(FaceLightmapInput {
                face_idx,
                s_min: b_s_min * 16.0,
                t_min: b_t_min * 16.0,
                width: w,
                height: h,
                rgb_luxels: &lighting[start..start + req_len],
            });
        }
    }

    assert!(!inputs.is_empty(), "expected parsed lightmap faces");
    println!(
        "Parsed {} lightmapped faces from de_inferno.bsp",
        inputs.len()
    );

    let packer = LightmapPacker::new(DEFAULT_ATLAS_WIDTH, DEFAULT_LIGHTGAMMA, DEFAULT_LIGHTSCALE);
    let payload = packer.build_lump_payload(&inputs);

    println!(
        "Packed Lightmap Atlas: total lump payload = {} KB",
        payload.len() / 1024
    );

    let view = LightmapAtlasView::new(&payload).expect("parse view");
    assert_eq!(view.header().magic, LIGHTMAP_MAGIC);
    assert_eq!(view.header().width, DEFAULT_ATLAS_WIDTH);
    assert_eq!(view.header().face_count as usize, inputs.len());
    assert_eq!(view.header().row_pitch_bytes % 256, 0);
    assert_eq!(view.header().data_offset % 256, 0);

    // Test FlatWorld container integration
    let flw = FlatWorldBuilder::new([-4096.0; 3], [4096.0; 3])
        .add_lump(LumpType::Lightmaps, inputs.len() as u64, 1, payload)
        .build();

    let world_view = FlatWorldView::new(&flw).expect("world view");
    let lm_opt = world_view.lightmap_view().expect("retrieve lightmap lump");
    assert!(lm_opt.is_some());
    let lm = lm_opt.unwrap();
    assert_eq!(lm.header().face_count as usize, inputs.len());
    assert_eq!(lm.face_descs().len(), inputs.len());
}
