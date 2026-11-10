# FlatAsset: Universal Zero-Alloc Binary Asset Ecosystem & Bundler

[![CI](https://github.com/ulquiorracode/flatasset-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ulquiorracode/flatasset-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/badge/crates.io-flatasset-blue.svg)](https://crates.io/crates/flatasset)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**FlatAsset** is the umbrella ecosystem and multi-asset packaging standard designed for extreme performance, mechanical sympathy, and zero-allocation execution in modern game engines and simulations.

It unifies the four specialized binary container standards:
1. **FlatMesh (`.flm`)**: Skeletal geometry, meshlets, bone weight quantization, and animation tracks.
2. **FlatTex (`.flt`)**: GPU Direct-DMA BC1..BC7 / ASTC textures with 256-byte row-pitch pre-alignment.
3. **FlatWorld (`.flw`)**: Map geometry, 4-way SIMD Q-BVH collision, and succinct bitmask PVS.
4. **FlatDemo (`.fld`)**: Columnar deterministic replay streams with $O(1)$ arbitrary tick random seek.
5. **FlatBundle (`.flb`)**: Heterogeneous multi-asset container with 64-byte alignment and BLAKE3 Merkle integrity.

---

## Performance Highlights

| Metric | Result | Benchmark Details |
| :--- | :--- | :--- |
| **Asset Identification** | **0.03 ns** | Branchless magic identification via 32-bit LE word |
| **Bundle Mount Time** | **< 2.0 ns** | Direct `mmap` zero-copy header validation |
| **All-in-One Package** | **25.9 KB** | Complete bundle containing mesh + texture + full map + demo |
| **Integrity Verification** | **Hardware-accelerated** | BLAKE3 Merkle root checksum across bundled lumps |

---

## Installation & CLI Usage

### Install CLI Tooling (`fla`)
```bash
cargo install --path cli/flatasset-cli
```

### Inspect Asset Kind
```bash
fla identify my_model.flm
```

### Bundle Multiple Assets into `.flb`
```bash
fla bundle -o package.flb model.flm texture.flt map.flw demo.fld --compress
```

### Inspect Bundle Contents
```bash
fla inspect-bundle package.flb
```

---

## License

Licensed under either of [Apache License, Version 2.0](LICENSE) or [MIT License](LICENSE) at your option.
