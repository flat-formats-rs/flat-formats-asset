use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use flatasset_core::{
    bundle::{FlatBundleBuilder, FlatBundleView},
    identify_flat_asset,
};
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
                    if item.compression != 0 {
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
