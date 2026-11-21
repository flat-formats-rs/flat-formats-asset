//! GoldSrc StudioMDL v10 legacy adapter for FlatMesh.
//!
//! Extracts skeletal hierarchy, vertex streams, de-strips triangle meshes,
//! quantizes normals to A2B10G10R10, converts UV to FP16, and extracts embedded textures.

use anyhow::{bail, Result};
use flatasset_mesh::{
    clusterize_triangles, FlatAttributeVertex, FlatDrawIndexedIndirect, FlatMeshBuilder,
    FlatPackedNormal, FlatPositionVertex, FlatSkinVertex,
};

/// Extracted 8-bit paletted texture converted to RGBA8.
#[derive(Clone, Debug)]
pub struct MdlTexture {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub _flags: u32,
    pub rgba_data: Vec<u8>,
}

/// Extracted bone descriptor from StudioMDL.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C, align(32))]
pub struct MdlBone {
    pub name_bytes: [u8; 32],
    pub parent: i32,
    pub flags: i32,
    pub pos: [f32; 3],
    pub rot: [f32; 3],
}
const _: () = assert!(core::mem::size_of::<MdlBone>() == 64);

/// Complete parsed StudioMDL model geometry and assets.
pub struct MdlModelData {
    pub positions: Vec<FlatPositionVertex>,
    pub attributes: Vec<FlatAttributeVertex>,
    pub skin_weights: Vec<FlatSkinVertex>,
    pub indices: Vec<u16>,
    pub bones: Vec<MdlBone>,
    pub textures: Vec<MdlTexture>,
    pub draw_commands: Vec<FlatDrawIndexedIndirect>,
}

impl MdlModelData {
    /// Convert parsed geometry into an ultra-fast FlatMesh container (.flm).
    pub fn build_flatmesh(&self) -> Result<Vec<u8>> {
        let clusters = clusterize_triangles(&self.positions, &self.indices);

        let mut builder = FlatMeshBuilder::new()
            .with_positions(self.positions.clone())
            .with_attributes(self.attributes.clone())
            .with_skin_rig(self.skin_weights.clone())
            .with_indices_u16(&self.indices)
            .with_meshlets(clusters.meshlets, clusters.micro_indices);

        if !self.draw_commands.is_empty() {
            builder = builder.with_indirect_draw(self.draw_commands.clone());
        }

        if !self.bones.is_empty() {
            let bone_bytes: &[u8] = unsafe {
                core::slice::from_raw_parts(
                    self.bones.as_ptr() as *const u8,
                    self.bones.len() * core::mem::size_of::<MdlBone>(),
                )
            };
            builder = builder.with_skeleton_hierarchy(
                bone_bytes.to_vec(),
                core::mem::size_of::<MdlBone>() as u8,
                self.bones.len() as u32,
            );
        }

        builder
            .build()
            .map_err(|e| anyhow::anyhow!("FlatMesh build error: {:?}", e))
    }
}

/// StudioMDL v10 Parser.
pub struct StudioMdlParser<'a> {
    data: &'a [u8],
}

impl<'a> StudioMdlParser<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 244 {
            bail!(
                "Buffer too small for StudioMDL header: {} bytes",
                data.len()
            );
        }

        let magic = &data[0..4];
        if magic != b"IDST" {
            bail!("Invalid magic signature: {:?}", magic);
        }

        let version = i32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 10 {
            bail!("Unsupported StudioMDL version: {}", version);
        }

        Ok(Self { data })
    }

    /// Parse complete StudioMDL geometry and optional external texture MDL (`<name>T.mdl`).
    pub fn parse(&self, external_texture_data: Option<&'a [u8]>) -> Result<MdlModelData> {
        let bones = self.parse_bones()?;
        let textures = self.parse_textures(external_texture_data)?;

        let numbodyparts = i32::from_le_bytes(self.data[204..208].try_into().unwrap()) as usize;
        let bodypartindex = i32::from_le_bytes(self.data[208..212].try_into().unwrap()) as usize;

        let mut positions = Vec::new();
        let mut attributes = Vec::new();
        let mut skin_weights = Vec::new();
        let mut indices = Vec::new();
        let mut draw_commands = Vec::new();

        for bp in 0..numbodyparts {
            let bp_base = bodypartindex + bp * 76;
            if bp_base + 76 > self.data.len() {
                bail!("Bodypart {} out of bounds", bp);
            }

            let nummodels =
                i32::from_le_bytes(self.data[bp_base + 64..bp_base + 68].try_into().unwrap())
                    as usize;
            let modelindex =
                i32::from_le_bytes(self.data[bp_base + 72..bp_base + 76].try_into().unwrap())
                    as usize;

            for m in 0..nummodels {
                let m_base = modelindex + m * 112;
                if m_base + 112 > self.data.len() {
                    bail!("Model {} in bodypart {} out of bounds", m, bp);
                }

                let nummesh =
                    i32::from_le_bytes(self.data[m_base + 72..m_base + 76].try_into().unwrap())
                        as usize;
                let meshindex =
                    i32::from_le_bytes(self.data[m_base + 76..m_base + 80].try_into().unwrap())
                        as usize;
                let numverts =
                    i32::from_le_bytes(self.data[m_base + 80..m_base + 84].try_into().unwrap())
                        as usize;
                let vertinfoindex =
                    i32::from_le_bytes(self.data[m_base + 84..m_base + 88].try_into().unwrap())
                        as usize;
                let vertindex =
                    i32::from_le_bytes(self.data[m_base + 88..m_base + 92].try_into().unwrap())
                        as usize;
                let numnorms =
                    i32::from_le_bytes(self.data[m_base + 92..m_base + 96].try_into().unwrap())
                        as usize;
                let normindex =
                    i32::from_le_bytes(self.data[m_base + 100..m_base + 104].try_into().unwrap())
                        as usize;

                if numverts == 0 || nummesh == 0 {
                    continue;
                }

                if vertindex + numverts * 12 > self.data.len()
                    || vertinfoindex + numverts > self.data.len()
                    || normindex + numnorms * 12 > self.data.len()
                {
                    bail!("Vertex data in model {} exceeds file buffer", m);
                }

                // Read vertex positions [f32; 3]
                let mdl_positions: &[[f32; 3]] = unsafe {
                    core::slice::from_raw_parts(
                        self.data[vertindex..].as_ptr() as *const [f32; 3],
                        numverts,
                    )
                };

                // Read vertex bone influences
                let mdl_vertinfo = &self.data[vertinfoindex..vertinfoindex + numverts];

                // Read vertex normals [f32; 3]
                let mdl_normals: &[[f32; 3]] = unsafe {
                    core::slice::from_raw_parts(
                        self.data[normindex..].as_ptr() as *const [f32; 3],
                        numnorms,
                    )
                };

                // Parse each mesh in this model
                for mi in 0..nummesh {
                    let mesh_base = meshindex + mi * 20;
                    if mesh_base + 20 > self.data.len() {
                        bail!("Mesh {} in model {} out of bounds", mi, m);
                    }

                    let triindex = i32::from_le_bytes(
                        self.data[mesh_base + 4..mesh_base + 8].try_into().unwrap(),
                    ) as usize;
                    let skinref = i32::from_le_bytes(
                        self.data[mesh_base + 8..mesh_base + 12].try_into().unwrap(),
                    ) as usize;

                    let (tex_w, tex_h) = if skinref < textures.len() {
                        (
                            textures[skinref].width as f32,
                            textures[skinref].height as f32,
                        )
                    } else {
                        (1.0f32, 1.0f32)
                    };

                    let inv_u = if tex_w > 0.0 { 1.0 / tex_w } else { 0.0 };
                    let inv_v = if tex_h > 0.0 { 1.0 / tex_h } else { 0.0 };

                    let first_index = indices.len() as u32;

                    // De-strip and de-fan triangles
                    let mut pos = triindex;
                    while pos + 2 <= self.data.len() {
                        let cmd = i16::from_le_bytes(self.data[pos..pos + 2].try_into().unwrap());
                        pos += 2;
                        if cmd == 0 {
                            break;
                        }

                        let is_strip = cmd > 0;
                        let count = cmd.unsigned_abs() as usize;

                        let mut run_indices = Vec::with_capacity(count);
                        for _ in 0..count {
                            if pos + 8 > self.data.len() {
                                bail!("Triangle command run exceeded data buffer");
                            }

                            let v_idx =
                                i16::from_le_bytes(self.data[pos..pos + 2].try_into().unwrap())
                                    as usize;
                            let n_idx =
                                i16::from_le_bytes(self.data[pos + 2..pos + 4].try_into().unwrap())
                                    as usize;
                            let s =
                                i16::from_le_bytes(self.data[pos + 4..pos + 6].try_into().unwrap());
                            let t =
                                i16::from_le_bytes(self.data[pos + 6..pos + 8].try_into().unwrap());
                            pos += 8;

                            if v_idx >= numverts || n_idx >= numnorms {
                                continue;
                            }

                            let p = mdl_positions[v_idx];
                            let n = mdl_normals[n_idx];
                            let bone = mdl_vertinfo[v_idx];

                            let u_norm = (s as f32) * inv_u;
                            let v_norm = (t as f32) * inv_v;

                            let u16_u = f32_to_f16(u_norm);
                            let u16_v = f32_to_f16(v_norm);

                            let packed_norm = FlatPackedNormal::pack([n[0], n[1], n[2]], 1.0);

                            let vert_id = positions.len() as u16;
                            positions.push(FlatPositionVertex {
                                position: p,
                                cluster_id: 0,
                            });
                            attributes.push(FlatAttributeVertex {
                                normal_tangent: packed_norm,
                                uv: [u16_u, u16_v],
                            });
                            skin_weights.push(FlatSkinVertex {
                                joint_indices: [bone, 0, 0, 0],
                                joint_weights: [255, 0, 0, 0],
                            });

                            run_indices.push(vert_id);
                        }

                        if run_indices.len() >= 3 {
                            if is_strip {
                                for i in 2..run_indices.len() {
                                    if i % 2 == 0 {
                                        indices.push(run_indices[i - 2]);
                                        indices.push(run_indices[i - 1]);
                                        indices.push(run_indices[i]);
                                    } else {
                                        indices.push(run_indices[i - 1]);
                                        indices.push(run_indices[i - 2]);
                                        indices.push(run_indices[i]);
                                    }
                                }
                            } else {
                                // Triangle Fan
                                for i in 2..run_indices.len() {
                                    indices.push(run_indices[0]);
                                    indices.push(run_indices[i - 1]);
                                    indices.push(run_indices[i]);
                                }
                            }
                        }
                    }

                    let index_count = (indices.len() as u32) - first_index;
                    if index_count > 0 {
                        draw_commands.push(FlatDrawIndexedIndirect {
                            index_count,
                            instance_count: 1,
                            first_index,
                            vertex_offset: 0,
                            first_instance: 0,
                        });
                    }
                }
            }
        }

        Ok(MdlModelData {
            positions,
            attributes,
            skin_weights,
            indices,
            bones,
            textures,
            draw_commands,
        })
    }

    /// Extract bones hierarchy.
    pub fn parse_bones(&self) -> Result<Vec<MdlBone>> {
        let numbones = i32::from_le_bytes(self.data[140..144].try_into().unwrap()) as usize;
        let boneindex = i32::from_le_bytes(self.data[144..148].try_into().unwrap()) as usize;

        if boneindex + numbones * 112 > self.data.len() {
            bail!("Bone table exceeds file buffer");
        }

        let mut bones = Vec::with_capacity(numbones);
        for b in 0..numbones {
            let b_off = boneindex + b * 112;
            let mut name_bytes = [0u8; 32];
            name_bytes.copy_from_slice(&self.data[b_off..b_off + 32]);

            let parent = i32::from_le_bytes(self.data[b_off + 32..b_off + 36].try_into().unwrap());
            let flags = i32::from_le_bytes(self.data[b_off + 36..b_off + 40].try_into().unwrap());

            let pos_raw: &[f32; 3] =
                unsafe { &*(self.data[b_off + 64..b_off + 76].as_ptr() as *const [f32; 3]) };
            let rot_raw: &[f32; 3] =
                unsafe { &*(self.data[b_off + 76..b_off + 88].as_ptr() as *const [f32; 3]) };

            bones.push(MdlBone {
                name_bytes,
                parent,
                flags,
                pos: *pos_raw,
                rot: *rot_raw,
            });
        }

        Ok(bones)
    }

    /// Extract embedded or external textures and palettes into RGBA8.
    pub fn parse_textures(&self, external_data: Option<&'a [u8]>) -> Result<Vec<MdlTexture>> {
        let mut source_data = self.data;
        let mut numtextures = i32::from_le_bytes(self.data[180..184].try_into().unwrap()) as usize;
        let mut textureindex = i32::from_le_bytes(self.data[184..188].try_into().unwrap()) as usize;

        if numtextures == 0 {
            if let Some(ext) = external_data {
                if ext.len() >= 188 && &ext[0..4] == b"IDST" {
                    source_data = ext;
                    numtextures = i32::from_le_bytes(ext[180..184].try_into().unwrap()) as usize;
                    textureindex = i32::from_le_bytes(ext[184..188].try_into().unwrap()) as usize;
                }
            }
        }

        if numtextures == 0 {
            return Ok(Vec::new());
        }

        let mut textures = Vec::with_capacity(numtextures);
        for t in 0..numtextures {
            let t_base = textureindex + t * 80;
            if t_base + 80 > source_data.len() {
                break;
            }

            let name_slice = &source_data[t_base..t_base + 64];
            let name_len = name_slice
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_slice.len());
            let name = String::from_utf8_lossy(&name_slice[..name_len]).into_owned();

            let flags =
                u32::from_le_bytes(source_data[t_base + 64..t_base + 68].try_into().unwrap());
            let width =
                u32::from_le_bytes(source_data[t_base + 68..t_base + 72].try_into().unwrap());
            let height =
                u32::from_le_bytes(source_data[t_base + 72..t_base + 76].try_into().unwrap());
            let index =
                u32::from_le_bytes(source_data[t_base + 76..t_base + 80].try_into().unwrap())
                    as usize;

            let pix_len = (width * height) as usize;
            let pal_off = index + pix_len;

            if pal_off + 768 > source_data.len() {
                continue;
            }

            let pixels = &source_data[index..index + pix_len];
            let palette = &source_data[pal_off..pal_off + 768];

            let mut rgba_data = Vec::with_capacity(pix_len * 4);
            for &p in pixels {
                let pal_idx = (p as usize) * 3;
                let r = palette[pal_idx];
                let g = palette[pal_idx + 1];
                let b = palette[pal_idx + 2];
                let a = if (flags & 0x02) != 0 && p == 255 {
                    0u8 // GoldSrc masked transparency
                } else {
                    255u8
                };
                rgba_data.extend_from_slice(&[r, g, b, a]);
            }

            textures.push(MdlTexture {
                name,
                width,
                height,
                _flags: flags,
                rgba_data,
            });
        }

        Ok(textures)
    }
}

/// Convert standard IEEE 754 f32 to IEEE 754 binary16 (half-precision).
#[inline(always)]
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;

    if exp <= 0 {
        0
    } else if exp >= 31 {
        (sign | 0x7C00) as u16
    } else {
        (sign | ((exp as u32) << 10) | mant) as u16
    }
}
