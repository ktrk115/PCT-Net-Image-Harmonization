# PCT-Net (package usage)

Fork of PCT-Net packaged for use as a library. See [README.md](../README.md) for the original project.

## Install

```sh
uv add git+https://github.com/ktrk115/PCT-Net-Image-Harmonization.git
```

`torch` and other runtime deps are not declared (CUDA-specific) — install them in the consumer project.

## Load a pretrained model

```python
from iharm.pretrained import load_pretrained, list_pretrained

list_pretrained()           # ['PCTNet_CNN', 'PCTNet_ViT']
model = load_pretrained("PCTNet_ViT").eval()
```

Weights are cached under `torch.hub.get_dir()/checkpoints/iharm/`.
Override the source repo via `repo=` / `branch=` or the `IHARM_REPO` / `IHARM_BRANCH` env vars.
