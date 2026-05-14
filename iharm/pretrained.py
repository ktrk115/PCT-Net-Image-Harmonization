import os
from pathlib import Path
from typing import Optional, Union

from torch.hub import download_url_to_file, get_dir

from iharm.inference.utils import load_model

DEFAULT_REPO = os.environ.get("IHARM_REPO", "ktrk115/PCT-Net-Image-Harmonization")
DEFAULT_BRANCH = os.environ.get("IHARM_BRANCH", "main")
DEFAULT_SUBDIR = "pretrained_models"

PRETRAINED_MODELS = {
    "PCTNet_CNN": {
        "model_type": "CNN_pct",
        "filename": "PCTNet_CNN.pth",
    },
    "PCTNet_ViT": {
        "model_type": "ViT_pct",
        "filename": "PCTNet_ViT.pth",
    },
}


def _default_cache_dir() -> Path:
    return Path(get_dir()) / "checkpoints" / "iharm"


def _build_url(filename: str, repo: str, branch: str, subdir: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{subdir}/{filename}"


def list_pretrained() -> list:
    return sorted(PRETRAINED_MODELS.keys())


def download_pretrained(
    name: str,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
    subdir: str = DEFAULT_SUBDIR,
    cache_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
    progress: bool = True,
) -> Path:
    if name not in PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown pretrained model: {name!r}. "
            f"Available: {list_pretrained()}"
        )

    repo = repo or DEFAULT_REPO
    branch = branch or DEFAULT_BRANCH
    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)

    filename = PRETRAINED_MODELS[name]["filename"]
    dest = cache_root / filename
    if force_reload or not dest.exists():
        url = _build_url(filename, repo, branch, subdir)
        download_url_to_file(url, str(dest), progress=progress)
    return dest


def load_pretrained(
    name: str,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
    subdir: str = DEFAULT_SUBDIR,
    cache_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
    progress: bool = True,
    verbose: bool = False,
):
    """Download (if needed) and load a pretrained PCT-Net model.

    Returns the instantiated ``torch.nn.Module`` with weights loaded.
    """
    weights_path = download_pretrained(
        name,
        repo=repo,
        branch=branch,
        subdir=subdir,
        cache_dir=cache_dir,
        force_reload=force_reload,
        progress=progress,
    )
    model_type = PRETRAINED_MODELS[name]["model_type"]
    return load_model(model_type, str(weights_path), verbose=verbose)
