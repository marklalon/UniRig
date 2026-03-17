#!/usr/bin/env python
"""
UniRig Model Downloader

This script downloads all required model checkpoints and configs for offline use.
Run this once after setup to ensure all models are available locally.

Usage:
    d:/AI/ComfyUI/tools/.venv/Scripts/python.exe download_models.py
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# Base directory for all downloads (repo root)
REPO_ROOT = Path(__file__).parent.resolve()

# UniRig checkpoints from VAST-AI/UniRig
# NOTE: skin/skeleton/model.ckpt does not exist in the HF repo (404)
UNIRIG_CHECKPOINTS = {
    "experiments/skeleton/articulation-xl_quantization_256/model.ckpt": {
        "repo_id": "VAST-AI/UniRig",
        "filename": "skeleton/articulation-xl_quantization_256/model.ckpt",
    },
    "experiments/skin/articulation-xl/model.ckpt": {
        "repo_id": "VAST-AI/UniRig",
        "filename": "skin/articulation-xl/model.ckpt",
    },
}

# Base transformer config (OPT-350M) for offline loading
OPT_CONFIG = {
    "repo_id": "facebook/opt-350m",
    "local_dir": "models/facebook/opt-350m",
}


def download_unirig_checkpoints():
    """Download UniRig checkpoints to local experiments/ directory."""
    print("=" * 60)
    print("Downloading UniRig checkpoints...")
    print("=" * 60)
    
    for local_path, info in UNIRIG_CHECKPOINTS.items():
        target_path = REPO_ROOT / local_path
        
        if target_path.exists():
            print(f"[SKIP] {local_path} already exists")
            continue
        
        print(f"[DOWNLOAD] {info['repo_id']} -> {local_path}")
        
        # Create parent directories
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download to a temp location first
        downloaded_path = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
        )
        
        # Copy to target location
        import shutil
        shutil.copy2(downloaded_path, target_path)
        print(f"[DONE] {local_path}")
    
    print()


def download_opt_config():
    """Download OPT-350M config for offline loading."""
    print("=" * 60)
    print("Downloading OPT-350M config...")
    print("=" * 60)
    
    local_dir = REPO_ROOT / OPT_CONFIG["local_dir"]
    
    if local_dir.exists() and (local_dir / "config.json").exists():
        print(f"[SKIP] {OPT_CONFIG['local_dir']} already exists")
        return
    
    print(f"[DOWNLOAD] {OPT_CONFIG['repo_id']} -> {OPT_CONFIG['local_dir']}")
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Download only config files, not the full model weights
    # We just need the config for AutoConfig.from_pretrained
    snapshot_download(
        repo_id=OPT_CONFIG["repo_id"],
        local_dir=str(local_dir),
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"],
    )
    
    print(f"[DONE] {OPT_CONFIG['local_dir']}")
    print()


def verify_downloads():
    """Verify all required files exist."""
    print("=" * 60)
    print("Verifying downloads...")
    print("=" * 60)
    
    all_ok = True
    
    for local_path in UNIRIG_CHECKPOINTS.keys():
        target_path = REPO_ROOT / local_path
        if target_path.exists():
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"[OK] {local_path} ({size_mb:.1f} MB)")
        else:
            print(f"[MISSING] {local_path}")
            all_ok = False
    
    opt_dir = REPO_ROOT / OPT_CONFIG["local_dir"]
    if (opt_dir / "config.json").exists():
        print(f"[OK] {OPT_CONFIG['local_dir']}/config.json")
    else:
        print(f"[MISSING] {OPT_CONFIG['local_dir']}/config.json")
        all_ok = False
    
    print()
    if all_ok:
        print("All models verified successfully!")
    else:
        print("Some models are missing. Please check the download process.")
    
    return all_ok


def main():
    print("UniRig Model Downloader")
    print(f"Target directory: {REPO_ROOT}")
    print()
    
    download_unirig_checkpoints()
    download_opt_config()
    verify_downloads()
    
    print()
    print("Download complete. You can now run UniRig offline.")


if __name__ == "__main__":
    main()
