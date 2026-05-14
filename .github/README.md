# PCT-Net (package usage)

Fork of PCT-Net packaged for use as a library. See [README.md](../README.md) for the original project.

## Install

```sh
uv add git+https://github.com/ktrk115/PCT-Net-Image-Harmonization.git
```

Inference dependencies (`torch>=1.12`, `torchvision>=0.13`, `einops`,
`opencv-python`) are installed automatically. If you need a specific
CUDA build of torch, install it yourself first.

`kornia` is only required for HSV/YUV color-space variants and
`Predictor(hsv=True)`; the released `PCTNet_CNN` / `PCTNet_ViT` checkpoints
are RGB so it is not pulled in. Install it manually if you need those paths.
Training/eval extras (`tensorboard`, `albumentations`, `pandas`,
`pytorch_msssim`, `tqdm`, etc.) are not pulled in — see `requirements.txt`.

## Load a pretrained model

```python
from iharm.pretrained import load_pretrained, list_pretrained

list_pretrained()                       # ['PCTNet_CNN', 'PCTNet_ViT']
model = load_pretrained("PCTNet_ViT").eval()
```

Both `PCTNet_CNN` (~8 MB, 2.2 M params) and `PCTNet_ViT` (~18 MB, 4.8 M params)
have been smoke-tested end-to-end: download → state-dict load → forward pass
on `(1, 3, 256, 256)` low-res + `(1, 3, 1024, 1024)` high-res inputs.

Weights are cached under `torch.hub.get_dir()/checkpoints/iharm/`.
Override the source repo via `repo=` / `branch=` arguments or the
`IHARM_REPO` / `IHARM_BRANCH` env vars.
