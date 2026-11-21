use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use flatasset_core::{
    bundle::{FlatBundleBuilder, FlatBundleView},
    cas::FlatCasPool,
    identify_flat_asset,
};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "fla")]
#[command(about = "FlatAsset Universal Ecosystem CLI & Multi-Asset Bundler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detects and inspects the kind and metadata of any Flat container file
    Identify {
        /// Path to the binary asset
        file: PathBuf,
    },
    /// Inspects an `.flb` FlatBundle container and lists its contents
    InspectBundle {
        /// Path to the .flb file
        file: PathBuf,
    },
    /// Bundles multiple flat assets into a single .flb package
    Bundle {
        /// Output .flb file
        #[arg(short, long)]
        output: PathBuf,
        /// Input files to bundle
        inputs: Vec<PathBuf>,
        /// Enable in-container Zstd compression for items
        #[arg(short, long)]
        compress: bool,
    },
    /// Content-Addressable Storage (CAS) Deduplicator across bundles and standalone assets
    Deduplicate {
        /// Input bundles or flat asset files to analyze and deduplicate
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output shared pool bundle path (e.g., `shared/pool.flb`)
        #[arg(short, long)]
        pool_output: PathBuf,
        /// Output directory for rewriting deduplicated bundles with CAS references
        #[arg(short, long)]
        out_dir: Option<PathBuf>,
    },
    /// Inspects and verifies the integrity of a CAS shared pool
    InspectPool {
        /// Path to the shared pool .flb file
        pool: PathBuf,
    },
    /// Converts textures from a GoldSrc WAD3 archive into FlatTex (.flt) Direct-DMA packages
    ConvertWad {
        /// Path to the .wad file
        wad: PathBuf,
        /// Output directory for .flt files
        #[arg(short, long)]
        out_dir: PathBuf,
    },
    /// Converts a 6-face skybox (TGA/BMP files: *bk, *dn, *ft, *lf, *rt, *up) into a FlatTex Cubemap
    ConvertSkybox {
        /// Path prefix to the 6 skybox faces (e.g. `cstrike/gfx/env/desert`)
        prefix: PathBuf,
        /// Output path for the .flt cubemap
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Microbenchmark for zero-copy inspection across assets
    BenchIdentify {
        /// Path to asset file
        file: PathBuf,
        /// Number of iterations
        #[arg(default_value_t = 100000)]
        iterations: u32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Identify { file } => {
            let data =
                std::fs::read(&file).with_context(|| format!("Failed to read {:?}", file))?;
            let kind = identify_flat_asset(&data);

            println!("=== FlatAsset Identification ===");
            println!("File: {:?}", file);
            println!("Kind: {}", kind.display_name());
            println!("File Extension: .{}", kind.extension());
            println!("Magic: 0x{:08X}", kind as u32);
            println!("Size: {} bytes", data.len());
        }
        Commands::InspectBundle { file } => {
            let f = File::open(&file).with_context(|| format!("Failed to open {:?}", file))?;
            let mmap = unsafe { memmap2::Mmap::map(&f)? };
            let view = FlatBundleView::new(&mmap)
                .map_err(|e| anyhow::anyhow!("FlatBundle validation error: {:?}", e))?;

            let header = view.header();
            println!("=== FlatBundle Container Inspector ===");
            println!("File: {:?}", file);
            println!("Version: {}", header.version);
            println!("Total Size: {} bytes", header.total_size);
            println!("Total Bundled Assets: {}", header.item_count);
            println!("Index Table Offset: 0x{:X}", header.index_offset);
            println!("BLAKE3 Checksum: {:02x?}", header.checksum);
            println!("\nBundled Assets:");
            println!(
                "--------------------------------------------------------------------------------"
            );
            println!("Index  Kind       Disk Offset   Disk Size    Mem Size     Compressed");
            println!(
                "--------------------------------------------------------------------------------"
            );

            for (i, item) in view.items().iter().enumerate() {
                let kind = match item.asset_kind {
                    0x4D544C46 => "FlatMesh ",
                    0x54544C46 => "FlatTex  ",
                    0x00574C46 => "FlatWorld",
                    0x00444C46 => "FlatDemo ",
                    0x42544C46 => "FlatBundle",
                    _ => "Unknown   ",
                };
                println!(
                    "{:<6} {:<10} 0x{:<10X} {:<12} {:<12} {}",
                    i,
                    kind,
                    item.file_offset,
                    item.byte_length,
                    item.uncompressed_size,
                    if item.is_cas_reference() {
                        "CAS Ref (Shared Pool)"
                    } else if item.compression != 0 {
                        "Yes (Zstd)"
                    } else {
                        "No (Raw DMA)"
                    }
                );
            }
        }
        Commands::Bundle {
            output,
            inputs,
            compress,
        } => {
            let mut builder = FlatBundleBuilder::new();
            println!(
                "Bundling {} assets into {:?} (compressed: {})...",
                inputs.len(),
                output,
                compress
            );

            for (i, input_path) in inputs.iter().enumerate() {
                let bytes = std::fs::read(input_path)
                    .with_context(|| format!("Failed to read asset {:?}", input_path))?;
                let name_hash = i as u64; // In real engine, hash path or ID
                builder.add_asset(name_hash, &bytes, compress);
                println!("  Added {:?} ({} bytes)", input_path, bytes.len());
            }

            let bundle_data = builder.build();
            std::fs::write(&output, &bundle_data)
                .with_context(|| format!("Failed to write bundle {:?}", output))?;

            println!(
                "Successfully generated bundle {:?} ({} bytes)!",
                output,
                bundle_data.len()
            );
        }
        Commands::Deduplicate {
            inputs,
            pool_output,
            out_dir,
        } => {
            println!("=== Content-Addressable Storage (CAS) Deduplicator ===");
            println!("Scanning {} input bundle(s)...", inputs.len());

            #[derive(Clone)]
            struct AssetItem {
                hash: [u8; 32],
                hash_id: u64,
                _kind: u32,
                uncompressed_size: u64,
                bytes: Vec<u8>,
                sources: Vec<PathBuf>,
            }

            type ManifestEntry = ([u8; 32], u32, u64);
            type BundleManifest = (PathBuf, Vec<ManifestEntry>);

            let mut asset_map: HashMap<[u8; 32], AssetItem> = HashMap::new();
            let mut bundle_manifests: Vec<BundleManifest> = Vec::new();

            for input_path in &inputs {
                let f = File::open(input_path)
                    .with_context(|| format!("Failed to open {:?}", input_path))?;
                let mmap = unsafe { memmap2::Mmap::map(&f)? };
                let view = FlatBundleView::new(&mmap)
                    .with_context(|| format!("Failed to parse bundle {:?}", input_path))?;

                let mut bundle_items = Vec::new();

                for item in view.items() {
                    let bytes = if item.is_cas_reference() && item.byte_length == 0 {
                        Vec::new()
                    } else {
                        view.item_bytes(item)?.to_vec()
                    };

                    if bytes.is_empty() {
                        continue;
                    }

                    let hash = *blake3::hash(&bytes).as_bytes();
                    let mut prefix = [0u8; 8];
                    prefix.copy_from_slice(&hash[..8]);
                    let hash_id = u64::from_le_bytes(prefix);

                    bundle_items.push((hash, item.asset_kind, item.uncompressed_size));

                    let entry = asset_map.entry(hash).or_insert_with(|| AssetItem {
                        hash,
                        hash_id,
                        _kind: item.asset_kind,
                        uncompressed_size: item.uncompressed_size,
                        bytes,
                        sources: Vec::new(),
                    });
                    entry.sources.push(input_path.clone());
                }

                bundle_manifests.push((input_path.clone(), bundle_items));
            }

            // Identify assets shared across multiple bundles or explicitly deduplicated
            let mut shared_assets: Vec<AssetItem> = Vec::new();
            let mut duplicate_wasted_bytes = 0u64;

            for item in asset_map.values() {
                if item.sources.len() > 1 {
                    duplicate_wasted_bytes +=
                        item.uncompressed_size * (item.sources.len() as u64 - 1);
                    shared_assets.push(item.clone());
                }
            }

            println!(
                "Discovered {} total unique assets across all packages.",
                asset_map.len()
            );
            println!(
                "Shared reusable assets: {} items (duplication savings: {:.2} MB)",
                shared_assets.len(),
                duplicate_wasted_bytes as f64 / (1024.0 * 1024.0)
            );

            // 1. Build and serialize the shared pool
            let mut pool_builder = FlatBundleBuilder::new();
            for asset in &shared_assets {
                pool_builder.add_asset(asset.hash_id, &asset.bytes, false);
            }
            let pool_bytes = pool_builder.build();

            if let Some(parent) = pool_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&pool_output, &pool_bytes)
                .with_context(|| format!("Failed to write pool {:?}", pool_output))?;

            println!(
                "Generated CAS Shared Pool {:?} ({} KB)",
                pool_output,
                pool_bytes.len() / 1024
            );

            // 2. Optionally rewrite deduplicated bundles with CAS references
            if let Some(target_dir) = out_dir {
                std::fs::create_dir_all(&target_dir)?;
                let shared_set: HashSet<[u8; 32]> = shared_assets.iter().map(|a| a.hash).collect();

                for (src_path, manifest) in &bundle_manifests {
                    let mut rewritten_builder = FlatBundleBuilder::new();
                    let file_name = src_path.file_name().unwrap_or_default().to_string_lossy();
                    let out_path = target_dir.join(file_name.as_ref());

                    for (hash, kind, uncompressed_size) in manifest {
                        if shared_set.contains(hash) {
                            // Replace with 64-byte CAS reference descriptor
                            rewritten_builder.add_cas_reference(*hash, *kind, *uncompressed_size);
                        } else if let Some(asset) = asset_map.get(hash) {
                            // Unique asset stays inline
                            rewritten_builder.add_asset(asset.hash_id, &asset.bytes, false);
                        }
                    }

                    let rewritten_data = rewritten_builder.build();
                    std::fs::write(&out_path, &rewritten_data)?;
                    println!(
                        "Rewrote bundle {:?} -> {:?} (size: {} KB -> {} KB)",
                        src_path.file_name().unwrap(),
                        out_path,
                        std::fs::metadata(src_path)?.len() / 1024,
                        rewritten_data.len() / 1024
                    );
                }
            }

            println!("Deduplication finished successfully.");
        }
        Commands::InspectPool { pool } => {
            let f = File::open(&pool).with_context(|| format!("Failed to open {:?}", pool))?;
            let mmap = unsafe { memmap2::Mmap::map(&f)? };
            let view = FlatBundleView::new(&mmap)
                .map_err(|e| anyhow::anyhow!("FlatBundle validation error: {:?}", e))?;

            let mut cas_pool = FlatCasPool::new();
            let count = cas_pool.register_bundle(&view)?;

            println!("=== CAS Shared Pool Inspector ===");
            println!("File: {:?}", pool);
            println!("Total Pool Size: {} KB", mmap.len() / 1024);
            println!("Indexed Unique Items: {}", count);
            println!("\nRegistered Assets:");
            println!(
                "--------------------------------------------------------------------------------"
            );
            println!("Index  Name Hash (BLAKE3-64)   Asset Kind    Disk Size     Uncompressed");
            println!(
                "--------------------------------------------------------------------------------"
            );

            for (i, item) in view.items().iter().enumerate() {
                let kind = match item.asset_kind {
                    0x4D544C46 => "FlatMesh ",
                    0x54544C46 => "FlatTex  ",
                    0x00574C46 => "FlatWorld",
                    0x00444C46 => "FlatDemo ",
                    _ => "Unknown  ",
                };
                println!(
                    "{:<6} 0x{:<20X} {:<12} {:<12} {:<12}",
                    i, item.name_hash, kind, item.byte_length, item.uncompressed_size
                );
            }
        }
        Commands::ConvertWad { wad, out_dir } => {
            println!("=== GoldSrc WAD3 to FlatTex Converter ===");
            let data = std::fs::read(&wad).with_context(|| format!("Failed to read {:?}", wad))?;
            if data.len() < 12 || &data[0..4] != b"WAD3" {
                anyhow::bail!("Invalid WAD3 signature: {:?}", &data[0..4.min(data.len())]);
            }

            let num_lumps = u32::from_le_bytes(data[4..8].try_into()?) as usize;
            let dir_offset = u32::from_le_bytes(data[8..12].try_into()?) as usize;

            println!("WAD: {:?} (Lumps: {})", wad, num_lumps);
            std::fs::create_dir_all(&out_dir)?;

            let mut converted = 0;
            for i in 0..num_lumps {
                let base = dir_offset + i * 32;
                if base + 32 > data.len() {
                    break;
                }
                let file_offset = u32::from_le_bytes(data[base..base + 4].try_into()?) as usize;
                let disk_size = u32::from_le_bytes(data[base + 4..base + 8].try_into()?) as usize;
                let lump_type = data[base + 12];
                let name = data[base + 16..base + 32]
                    .split(|&b| b == 0)
                    .next()
                    .unwrap_or_default();
                let name_str = String::from_utf8_lossy(name).to_string();

                // 0x43 is GoldSrc MIP texture
                if lump_type == 0x43 && file_offset + disk_size <= data.len() {
                    let tex_data = &data[file_offset..file_offset + disk_size];
                    if tex_data.len() < 40 {
                        continue;
                    }
                    let width = u32::from_le_bytes(tex_data[16..20].try_into()?);
                    let height = u32::from_le_bytes(tex_data[20..24].try_into()?);
                    let mip0_off = u32::from_le_bytes(tex_data[24..28].try_into()?) as usize;

                    if width == 0 || height == 0 || mip0_off >= tex_data.len() {
                        continue;
                    }

                    // Palette is at end of MIP texture: 2 bytes num_colors + 256*3 RGB
                    let total_pixels = (width * height) as usize;
                    let mip1_pixels = (width / 2 * height / 2) as usize;
                    let mip2_pixels = (width / 4 * height / 4) as usize;
                    let mip3_pixels = (width / 8 * height / 8) as usize;
                    let pal_offset =
                        mip0_off + total_pixels + mip1_pixels + mip2_pixels + mip3_pixels + 2;

                    if pal_offset + 768 > tex_data.len() {
                        continue;
                    }

                    let palette = &tex_data[pal_offset..pal_offset + 768];
                    let indices = &tex_data[mip0_off..mip0_off + total_pixels];

                    // Decode to RGBA8
                    let mut rgba = vec![255u8; total_pixels * 4];
                    for (px_idx, &pal_idx) in indices.iter().enumerate() {
                        let c_idx = (pal_idx as usize) * 3;
                        let r = palette[c_idx];
                        let g = palette[c_idx + 1];
                        let b = palette[c_idx + 2];
                        let out_idx = px_idx * 4;
                        rgba[out_idx] = r;
                        rgba[out_idx + 1] = g;
                        rgba[out_idx + 2] = b;
                        // Blue-screen transparency for '{' prefixed textures
                        if name_str.starts_with('{') && r == 0 && g == 0 && b == 255 {
                            rgba[out_idx + 3] = 0;
                        }
                    }

                    // Encode to Direct-DMA BC1
                    let (dma_bytes, aligned_pitch, rows) =
                        flatasset_tex::encode_rgba_to_bc1_dma(width, height, &rgba);
                    let mut builder = flatasset_tex::FlatTexBuilder::new_2d(
                        width as u16,
                        height as u16,
                        flatasset_tex::VkFormat::Bc1RgbUnormBlock,
                    );
                    builder.add_mip(flatasset_tex::MipInput {
                        width,
                        height,
                        dma_bytes,
                        aligned_row_pitch: aligned_pitch,
                        rows_per_image: rows,
                    });

                    if let Ok(flt_bytes) = builder.build() {
                        let clean_name = name_str.replace(['/', '\\'], "_");
                        let out_file = out_dir.join(format!("{}.flt", clean_name));
                        std::fs::write(&out_file, &flt_bytes)?;
                        converted += 1;
                    }
                }
            }
            println!(
                "Converted {} MIP textures to Direct-DMA FlatTex (.flt)!",
                converted
            );
        }
        Commands::ConvertSkybox { prefix, output } => {
            println!("=== Skybox 6-Face to FlatTex Cubemap Converter ===");
            let suffixes = ["ft", "bk", "up", "dn", "rt", "lf"];
            let mut face_images = Vec::new();
            let mut cubemap_w = 0u32;
            let mut cubemap_h = 0u32;

            for suffix in &suffixes {
                let prefix_str = prefix.to_string_lossy();
                let path_tga = PathBuf::from(format!("{}{}.tga", prefix_str, suffix));
                let path_bmp = PathBuf::from(format!("{}{}.bmp", prefix_str, suffix));

                let path_to_load = if path_tga.exists() {
                    path_tga
                } else if path_bmp.exists() {
                    path_bmp
                } else {
                    anyhow::bail!(
                        "Missing skybox face: neither {:?} nor {:?} found",
                        path_tga,
                        path_bmp
                    );
                };

                let img = image::open(&path_to_load)
                    .with_context(|| format!("Failed to load skybox face {:?}", path_to_load))?
                    .to_rgba8();

                let (w, h) = img.dimensions();
                if cubemap_w == 0 {
                    cubemap_w = w;
                    cubemap_h = h;
                } else if cubemap_w != w || cubemap_h != h {
                    anyhow::bail!("Cubemap face dimension mismatch in {:?}", path_to_load);
                }

                face_images.push(img.into_raw());
            }

            let mut builder = flatasset_tex::FlatTexBuilder::new_cubemap(
                cubemap_w as u16,
                cubemap_h as u16,
                flatasset_tex::VkFormat::Bc1RgbUnormBlock,
            );

            for (face_idx, rgba) in face_images.iter().enumerate() {
                let (dma_bytes, aligned_pitch, rows) =
                    flatasset_tex::encode_rgba_to_bc1_dma(cubemap_w, cubemap_h, rgba);
                builder.add_mip(flatasset_tex::MipInput {
                    width: cubemap_w,
                    height: cubemap_h,
                    dma_bytes,
                    aligned_row_pitch: aligned_pitch,
                    rows_per_image: rows,
                });
                println!(
                    "  Face {}: {} (aligned pitch: {}B)",
                    face_idx, suffixes[face_idx], aligned_pitch
                );
            }

            let flt_bytes = builder.build().map_err(|e| anyhow::anyhow!("{:?}", e))?;
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, &flt_bytes)?;
            println!(
                "Successfully generated Cubemap FlatTex {:?} ({} KB, 6 faces)!",
                output,
                flt_bytes.len() / 1024
            );
        }
        Commands::BenchIdentify { file, iterations } => {
            let data = std::fs::read(&file)?;
            let start = std::time::Instant::now();
            let mut sink = 0u32;

            for _ in 0..iterations {
                let kind = identify_flat_asset(&data);
                sink = sink.wrapping_add(kind as u32);
            }

            let elapsed = start.elapsed();
            let per_op = elapsed.as_secs_f64() * 1e9 / (iterations as f64);
            println!("=== Asset Identification Benchmark ===");
            println!("Iterations: {}", iterations);
            println!("Total Time: {:?}", elapsed);
            println!("Latency per check: {:.2} ns", per_op);
            println!("Dummy sink: {}", sink);
        }
    }

    Ok(())
}
