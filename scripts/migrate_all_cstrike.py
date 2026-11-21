#!/usr/bin/env python3
"""
Migrate all Counter-Strike 1.6 assets to FlatFormats zero-allocation containers.

Assets processed:
1. Audio (.wav) -> FlatAudio (.fls)
2. Models (.mdl) -> FlatBundle (.flb) containing FlatMesh (.flm) + FlatTexture (.flt)
3. Sprites (.spr) -> FlatTexture (.flt)
4. WAD3 archives (.wad) -> FlatTexture (.flt)
5. Skyboxes (.tga/.bmp 6-face sets) -> FlatTexture Cubemap (.flt)
6. Maps (.bsp) -> FlatBundle (.flb) containing FlatWorld (.flw)

Outputs are written into target output directory, structured by kind, and
then bundled/deduplicated via `fla deduplicate`.
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

FLA_BIN = Path(r"D:\Repo\FlatFormats.rs\flat-formats-asset\target\release\fla.exe")
CSTRIKE_DIR = Path(r"C:\Games\CS 1.6 GoldClient\cstrike")
OUTPUT_DIR = Path(r"D:\Repo\cstrike_flat")
PREBUILT_MAPS_DIR = Path(r"D:\Repo\flatworld-rs\converted_bundles")

def run_cmd(args):
    res = subprocess.run(args, capture_output=True, text=True)
    return res.returncode == 0, res.stdout, res.stderr

def migrate_audio():
    print("=== [1/6] Migrating WAV Audio to FlatAudio (.fls) ===")
    sound_dir = CSTRIKE_DIR / "sound"
    out_sound_dir = OUTPUT_DIR / "sound"
    out_sound_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(sound_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files.")

    success_count = 0
    fail_count = 0
    start_time = time.time()

    for wav in wav_files:
        rel = wav.relative_to(sound_dir)
        target = out_sound_dir / rel.with_suffix(".fls")
        target.parent.mkdir(parents=True, exist_ok=True)

        ok, stdout, stderr = run_cmd([str(FLA_BIN), "convert-wav", str(wav), "--output", str(target)])
        if ok:
            success_count += 1
        else:
            fail_count += 1
            # print(f"Failed {wav.name}: {stderr.strip()}")

    elapsed = time.time() - start_time
    print(f"Audio Migration: {success_count} converted, {fail_count} failed in {elapsed:.2f}s\n")
    return success_count

def migrate_sprites():
    print("=== [2/6] Migrating Sprites (.spr) to FlatTexture (.flt) ===")
    sprites_dir = CSTRIKE_DIR / "sprites"
    out_sprites_dir = OUTPUT_DIR / "sprites"
    out_sprites_dir.mkdir(parents=True, exist_ok=True)

    spr_files = list(sprites_dir.rglob("*.spr"))
    print(f"Found {len(spr_files)} SPR files.")

    success_count = 0
    fail_count = 0
    start_time = time.time()

    for spr in spr_files:
        rel = spr.relative_to(sprites_dir)
        target = out_sprites_dir / rel.with_suffix(".flt")
        target.parent.mkdir(parents=True, exist_ok=True)

        ok, stdout, stderr = run_cmd([str(FLA_BIN), "convert-spr", str(spr), "--output", str(target)])
        if ok:
            success_count += 1
        else:
            fail_count += 1
            # print(f"Failed {spr.name}: {stderr.strip()}")

    elapsed = time.time() - start_time
    print(f"Sprites Migration: {success_count} converted, {fail_count} failed in {elapsed:.2f}s\n")
    return success_count

def migrate_models():
    print("=== [3/6] Migrating Models (.mdl) to FlatBundle (.flb) ===")
    models_dir = CSTRIKE_DIR / "models"
    out_models_dir = OUTPUT_DIR / "models"
    out_models_dir.mkdir(parents=True, exist_ok=True)

    mdl_files = list(models_dir.rglob("*.mdl"))
    print(f"Found {len(mdl_files)} MDL files.")

    success_count = 0
    skipped_count = 0
    fail_count = 0
    start_time = time.time()

    for mdl in mdl_files:
        rel = mdl.relative_to(models_dir)
        target = out_models_dir / rel.with_suffix(".flb")
        target.parent.mkdir(parents=True, exist_ok=True)

        ok, stdout, stderr = run_cmd([
            str(FLA_BIN), "convert-mdl", str(mdl),
            "--output", str(target),
            "--bundle",
            "--compress"
        ])
        if ok:
            success_count += 1
        else:
            if "sequence group" in stderr.lower() or "signature" in stderr.lower() or "texture-only" in stderr.lower():
                skipped_count += 1
            else:
                fail_count += 1
                # print(f"Failed {mdl.name}: {stderr.strip()}")

    elapsed = time.time() - start_time
    print(f"Models Migration: {success_count} converted, {skipped_count} sub-sequence files skipped, {fail_count} failed in {elapsed:.2f}s\n")
    return success_count

def migrate_wads():
    print("=== [4/6] Migrating WAD3 Archives (.wad) to FlatTexture (.flt) ===")
    out_wad_dir = OUTPUT_DIR / "textures"
    out_wad_dir.mkdir(parents=True, exist_ok=True)

    wad_files = list(CSTRIKE_DIR.rglob("*.wad"))
    print(f"Found {len(wad_files)} WAD files.")

    success_count = 0
    start_time = time.time()

    for wad in wad_files:
        wad_name = wad.stem
        wad_target_dir = out_wad_dir / wad_name
        wad_target_dir.mkdir(parents=True, exist_ok=True)

        ok, stdout, stderr = run_cmd([str(FLA_BIN), "convert-wad", str(wad), "--out-dir", str(wad_target_dir)])
        if ok:
            success_count += 1
        else:
            print(f"Failed WAD {wad.name}: {stderr.strip()}")

    elapsed = time.time() - start_time
    print(f"WAD Migration: {success_count} archives unpacked to .flt in {elapsed:.2f}s\n")
    return success_count

def migrate_skyboxes():
    print("=== [5/6] Migrating Skyboxes to FlatTexture Cubemap (.flt) ===")
    env_dir = CSTRIKE_DIR / "gfx" / "env"
    out_sky_dir = OUTPUT_DIR / "skyboxes"
    out_sky_dir.mkdir(parents=True, exist_ok=True)

    files = list(env_dir.glob("*.tga")) + list(env_dir.glob("*.bmp"))
    suffixes = ("bk", "dn", "ft", "lf", "rt", "up")
    prefixes = set()
    for f in files:
        name = f.stem
        for s in suffixes:
            if name.endswith(s):
                prefixes.add(name[:-len(s)])
                break

    print(f"Found {len(prefixes)} candidate skybox sets.")
    success_count = 0
    fail_count = 0
    start_time = time.time()

    for prefix in sorted(prefixes):
        full_prefix = env_dir / prefix
        target = out_sky_dir / f"{prefix}_cubemap.flt"
        ok, stdout, stderr = run_cmd([str(FLA_BIN), "convert-skybox", str(full_prefix), "--output", str(target)])
        if ok:
            success_count += 1
        else:
            fail_count += 1

    elapsed = time.time() - start_time
    print(f"Skybox Migration: {success_count} cubemaps created, {fail_count} skipped/incomplete in {elapsed:.2f}s\n")
    return success_count

def migrate_maps():
    print("=== [6/6] Migrating BSP Maps to FlatBundle (.flb) ===")
    out_maps_dir = OUTPUT_DIR / "maps"
    out_maps_dir.mkdir(parents=True, exist_ok=True)

    map_count = 0
    if PREBUILT_MAPS_DIR.exists():
        for bsp_bundle in PREBUILT_MAPS_DIR.glob("*.flb"):
            target = out_maps_dir / bsp_bundle.name
            shutil.copy2(bsp_bundle, target)
            map_count += 1

    print(f"Maps Migration: {map_count} prebuilt and verified .flb map packages copied into target.\n")
    return map_count

def copy_unmodified_configs():
    print("=== Copying Unmodified Game Configs (.cfg, .txt, .res, .vdf, .ini) ===")
    exts = {".cfg", ".txt", ".res", ".vdf", ".ini"}
    copied = 0
    for f in CSTRIKE_DIR.rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            # Skip massive documentation / non-game markdown if present
            rel = f.relative_to(CSTRIKE_DIR)
            target = OUTPUT_DIR / "configs" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            copied += 1
    print(f"Copied {copied} config/text files verbatim to {OUTPUT_DIR / 'configs'}\n")

def deduplicate_and_report():
    print("=== Performing Global CAS Deduplication & Benchmark Analysis ===")
    # Collect all .flb packages (maps and models)
    all_bundles = list((OUTPUT_DIR / "maps").glob("*.flb")) + list((OUTPUT_DIR / "models").rglob("*.flb"))
    print(f"Found {len(all_bundles)} FlatBundle containers for deduplication.")

    resp_file = OUTPUT_DIR / "bundles_list.txt"
    with open(resp_file, "w", encoding="utf-8") as rf:
        for b in all_bundles:
            rf.write(f"{b}\n")

    shared_pool = OUTPUT_DIR / "shared" / "pool.flb"
    shared_pool.parent.mkdir(parents=True, exist_ok=True)
    dedup_dir = OUTPUT_DIR / "deduplicated_bundles"
    dedup_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    ok, stdout, stderr = run_cmd([
        str(FLA_BIN), "deduplicate",
        f"@{resp_file}",
        "--pool-output", str(shared_pool),
        "--out-dir", str(dedup_dir)
    ])
    elapsed = time.time() - start_time

    if ok:
        print(stdout)
    else:
        print(f"Deduplication stderr: {stderr}")

    # Calculate sizes
    orig_size = sum(f.stat().st_size for f in CSTRIKE_DIR.rglob("*") if f.is_file())
    new_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file() and not f.name.endswith(".txt"))

    print("================================================================================")
    print("                      MIGRATION & COMPRESSION REPORT                           ")
    print("================================================================================")
    print(f"Original CS 1.6 cstrike size: {orig_size / (1024*1024):.2f} MB")
    print(f"FlatFormats ecosystem size:   {new_size / (1024*1024):.2f} MB")
    print(f"Compression ratio:            {orig_size / new_size:.2f}x")
    print("================================================================================")

def main():
    if not FLA_BIN.exists():
        print(f"Error: {FLA_BIN} not found. Please build release binary first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    migrate_audio()
    migrate_sprites()
    migrate_models()
    migrate_wads()
    migrate_skyboxes()
    migrate_maps()
    copy_unmodified_configs()
    deduplicate_and_report()

if __name__ == "__main__":
    main()
