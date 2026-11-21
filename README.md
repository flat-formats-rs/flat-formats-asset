# FlatAsset: Universal Zero-Alloc Binary Asset Ecosystem & Bundler

[![CI](https://github.com/flat-formats-rs/flat-formats-asset/actions/workflows/ci.yml/badge.svg)](https://github.com/flat-formats-rs/flat-formats-asset/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Zero Alloc](https://img.shields.io/badge/allocations-0_heap-brightgreen.svg)](#performance-highlights)
[![Rust Edition](https://img.shields.io/badge/Rust-2024-orange.svg)](Cargo.toml)

**FlatAsset** is the umbrella ecosystem and multi-asset packaging standard designed for extreme performance, mechanical sympathy, and zero-allocation execution in modern game engines and simulations.

It unifies specialized binary container standards:

1. **FlatWorld (`.flw`)**: Map geometry, 4-way SIMD Q-BVH collision, and succinct bitmask PVS.
2. **FlatMesh (`.flm`)**: Skeletal geometry, meshlets, bone weight quantization, and animation tracks.
3. **FlatTexture (`.flt`)**: GPU Direct-DMA BC1..BC7 / ASTC textures with 256-byte row-pitch pre-alignment.
4. **FlatAudio (`.fls`)**: Zero-alloc low-latency audio, integer-only IMA-ADPCM 4:1 codec, and direct-to-mixer PCM16 streaming.
5. **FlatDemo (`.fld`)**: Columnar deterministic replay streams with $O(1)$ arbitrary tick random seek.
6. **FlatBundle (`.flb`)**: Heterogeneous multi-asset container with Content-Addressable Storage (CAS), Merkle BLAKE3 deduplication, and zero-copy slicing.

---

## Performance Highlights

Comparative benchmarks executed in [GoldHub](https://github.com/goldhub-rs/goldhub) on production Counter-Strike 1.6 game assets (`de_dust2`, `v_ak47`, `Birds1.wav`):

| Asset Category | Legacy GoldSrc Asset | FlatFormat Container | Legacy Latency | FlatFormat Latency | Speedup Factor | Memory Allocations |
| --- | --- | --- | --- | --- | --- | --- |
| **World Map** (`de_dust2`) | Legacy BSP30 (15 lumps) | **FlatWorld (`.flw`)** | 166.00 µs | **7.40 ns** | **22,432x faster** | **0 heap allocs** |
| **Skeletal Model** (`v_ak47`) | StudioMDL Binary | **FlatMesh (`.flm`)** | 995.00 µs | **12.80 ns** | **77,734x faster** | **0 heap allocs** |
| **Audio Sound** (`Birds1.wav`) | RIFF WAVE PCM | **FlatAudio (`.fls`)** | 1.24 µs | **3.17 ns** | **391x faster** | **0 heap allocs** |
| **Shared CAS Pool** | Unshared duplicate files | **FlatCasPool (`.flb`)** | N/A (RAM bloat) | **Zero-Copy Slice** | **Single copy in RAM** | **Deduplicated** |

---

## Workspace Layout

| Crate | Container | Description |
| --- | --- | --- |
| [`flatasset-world`](crates/flatasset-world) | `.flw` | Map geometry, SIMD 4-way Q-BVH, lightmaps, and cluster PVS. |
| [`flatasset-mesh`](crates/flatasset-mesh) | `.flm` | Skeletal meshes, meshlets, bone transforms, and vertex streams. |
| [`flatasset-texture`](crates/flatasset-texture) | `.flt` | GPU textures, BC1-BC7 mip levels, and DMA pitch alignments. |
| [`flatasset-audio`](crates/flatasset-audio) | `.fls` | Integer IMA-ADPCM codec, raw PCM16 streaming, and 3D audio flags. |
| [`flatasset-core`](crates/flatasset-core) | `.flb` | Multi-asset bundle container, Content-Addressable Storage, and Merkle tree. |
| [`flatasset-cli`](crates/flatasset-cli) | CLI | `fla` command-line tool for bundling, inspection, and compilation. |

---

## Installation & CLI Usage

### Install CLI Tooling (`fla`)

```bash
cargo install --path crates/flatasset-cli
```

### Inspect Asset Kind

```bash
fla identify my_model.flm
```

### Bundle Multiple Assets into `.flb`

```bash
fla bundle -o package.flb model.flm texture.flt map.flw sound.fls demo.fld --compress
```

### Inspect Bundle Contents

```bash
fla inspect-bundle package.flb
```

---

## License

Licensed under the [MIT License](LICENSE).
