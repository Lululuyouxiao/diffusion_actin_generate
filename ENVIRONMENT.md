# Environment

`environment.yml` describes the modern Linux environment used on the RTX
4090 D server. It targets Python 3.9, PyTorch 2.3.1 with CUDA 12.1,
torchvision 0.18.1, NumPy 1.26.4, and the package versions recorded from the
working `actin-generate` environment.

The original upstream README declared Python 3.9, CUDA 11.7, PyTorch 1.11.0,
and torchvision 0.12.0. Those are historical upstream versions and are no
longer the versions configured by this file. The installed NVIDIA driver
provides the system interface; `nvcc` is not required for the prebuilt
PyTorch CUDA wheels.

Update an existing environment without recreating it:

```bash
conda env update -n actin-generate -f environment.yml
conda activate actin-generate
```

The PyTorch CUDA wheels install their required `nvidia-*-cu12` runtime
packages. These are intentionally not listed separately in the YAML.

## Current package pins

```text
torch==2.3.1+cu121
torchvision==0.18.1+cu121
numpy==1.26.4
einops==0.8.2
tqdm==4.70.1
matplotlib==3.9.4
scipy==1.13.1
scikit-image==0.24.0
opencv-python-headless==4.11.0.86
pillow==11.3.0
```

The upstream `requirements.txt` also declares `pylsd2`. It is not present in
the current server environment; install it separately only when using code
that imports the LSD geometry evaluator.

## Paper environment note

The paper states that the implementation uses PyTorch and was accelerated
by NVIDIA RTX A4000 and NVIDIA Titan RTX GPUs. It does not specify additional
Python, PyTorch, or CUDA versions for the diffusion model. The current server
environment is selected for RTX 4090 D compatibility and is not an exact
recreation of the upstream software environment.

## Data

The original README requires the dataset archive to be extracted before
training. The training implementation expects extracted paired image
directories; it does not read `.7z` archives directly.
