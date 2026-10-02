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

Use Python 3.11 on Windows:

```powershell
uv python install 3.11
uv venv --python 3.11
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

## Custom all-images batch script

Use `scripts/batch_custom_cellpose_signal_analysis.py` to process all complete C0/C1/C2 TIFF groups under a folder. It supports nested treatment and well folders and uses the custom preprocessing and Cellpose parameters defined at the top of the script.

Edit these settings:

```python
DATA_ROOT = Path(r'E:\N_segmentation\path\to\input_folder')
OUTPUT_DIR = Path(r'E:\N_segmentation\path\to\output_folder')

C0_SIGNAL_THRESHOLD = 300
C1_SIGNAL_THRESHOLD = 300
```

The thresholds use raw TIFF intensities. A cell is counted as positive when at least one pixel inside its mask is strictly greater than the corresponding channel threshold.

Run:

```powershell
uv run python scripts/batch_custom_cellpose_signal_analysis.py
```

You can also override the main paths from the command line:

```powershell
uv run python scripts/batch_custom_cellpose_signal_analysis.py `
  --data-root "E:\N_segmentation\path\to\input_folder" `
  --output-dir "E:\N_segmentation\path\to\output_folder" `
  --max-images 1
```

The output contains Cellpose masks, processed brightfield images, overlays, per-image counts, per-cell details, and `processing_errors.csv`. Overlay colors are cyan Cellpose boundaries, red C0 signal, green C1 signal, and yellow simultaneous C0+C1 signal.

## Regenerate signal results without Cellpose

Use `scripts/regenerate_signal_threshold_results.py` when masks already exist and only the C0/C1 thresholds need to change. It reads the previous run's per-image and per-cell CSV files, reuses the saved masks, regenerates the overlays, and recalculates signal counts. Cellpose is not run again, and previous morphology labels are preserved.

Run it with:

```powershell
uv run python scripts/regenerate_signal_threshold_results.py `
  --results-dir "E:\N_segmentation\path\to\previous_results" `
  --c0-threshold 300 `
  --c1-threshold 500 `
  --output-dir "E:\N_segmentation\path\to\threshold_regenerated"
```

The script also has editable defaults at the top:

```python
RESULTS_DIR = Path(r'E:\N_segmentation\path\to\previous_results')
PER_IMAGE_CSV = RESULTS_DIR / 'per_image_morphology_signal_counts.csv'
PER_CELL_CSV = RESULTS_DIR / 'per_cell_morphology_signal_details.csv'
OUTPUT_DIR = RESULTS_DIR / 'threshold_regenerated'

C0_SIGNAL_THRESHOLD = 300
C1_SIGNAL_THRESHOLD = 500
```

The regenerated output contains replacement per-image and per-cell CSV tables, updated overlays, and `processing_errors.csv`.
