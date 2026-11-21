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
