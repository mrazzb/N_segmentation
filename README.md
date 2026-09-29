# Cellpose Nested Layout Pipeline

This repository runs the Cellpose morphology and C0/C1 signal analysis pipeline on TIFF files arranged in nested treatment and well folders.

Expected layout:

```text
17032026_AnnV_PI_01_Split Scenes/
  CsA/
    A1/
    B1/
    E1/
    E2/
  DTX/
    A1/
```

Each final-level folder may contain the three channel files for many images. Files are grouped by their parent folder and microscope image stem, for example `C=0`, `C=1`, and `C=2`.

## Install with uv

Install uv on Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
After installing, close and reopen the terminal, then check that `uv` works:

```powershell
uv --version
uv sync --extra gpu
uv run python scripts/check_gpu.py
```

The `gpu` extra installs the PyTorch CUDA 12.8 wheel. The Cellpose script uses `use_gpu = True` and `model = "cellsam"` by default.

## Run

Edit `DATA_ROOT` in `scripts/batch_nested_layout_morphology_signal_analysis.py`, then run:

```powershell
uv run python scripts/batch_nested_layout_morphology_signal_analysis.py
```

Results are written beside the input folder in a separate `*_morphology_signal_results` directory. The output includes a layout manifest, per-image counts, per-cell signal details, masks, and overlays.

Set `MAX_IMAGES = 1` in the script for a test run. Existing masks are reused when present.

## GPU notes

The GPU model matters indirectly through its NVIDIA compute capability and the installed driver. The Python code does not need a GPU-specific target, but PyTorch must be installed with a CUDA wheel compatible with the driver and GPU. A current NVIDIA driver can generally run an older CUDA runtime wheel; a very old GPU may not be supported by newer PyTorch wheels.

Check the driver and GPU with:

```powershell
nvidia-smi
```

If CUDA 12.8 is not appropriate for the target machine, use the PyTorch installation selector and change the `torch` index in `pyproject.toml` to a supported CUDA wheel, such as `cu126`. Do not install a random CPU `torch` wheel over the CUDA wheel.

Official references:

- https://pytorch.org/get-started/locally/
- https://docs.astral.sh/uv/concepts/projects/dependencies/
- https://cellpose.readthedocs.io/
