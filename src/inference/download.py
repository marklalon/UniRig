import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Repo root directory (where download_models.py saves checkpoints)
_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()

def download(ckpt_name: str) -> str:
    """
    Return local checkpoint path, downloading from HuggingFace if needed.
    Prioritizes repo-local files for offline use.
    """
    # NOTE: experiments/skin/skeleton/model.ckpt does not exist in HF repo
    MAP = {
        'experiments/skeleton/articulation-xl_quantization_256/model.ckpt': 'skeleton/articulation-xl_quantization_256/model.ckpt',
        'experiments/skin/articulation-xl/model.ckpt': 'skin/articulation-xl/model.ckpt',
    }
    
    # If empty or None, return as-is
    if not ckpt_name:
        return ckpt_name
    
    # Check if local file already exists (prioritize offline)
    local_path = _REPO_ROOT / ckpt_name
    if local_path.exists():
        print(f"[local] using checkpoint: {ckpt_name}")
        return str(local_path)
    
    # If not in map, return original path (may fail later if file doesn't exist)
    if ckpt_name not in MAP:
        print(f"not found in map: {ckpt_name}")
        return ckpt_name
    
    # Download from HuggingFace and save to local path
    try:
        print(f"[download] fetching {ckpt_name} from HuggingFace...")
        downloaded_path = hf_hub_download(
            repo_id='VAST-AI/UniRig',
            filename=MAP[ckpt_name],
        )
        
        # Ensure target directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy to local path for future offline use
        import shutil
        shutil.copy2(downloaded_path, local_path)
        print(f"[download] saved to: {local_path}")
        
        return str(local_path)
    except Exception as e:
        print(f"Failed to download {ckpt_name}: {e}")
        return ckpt_name