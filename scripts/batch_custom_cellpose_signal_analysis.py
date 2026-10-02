from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from cellpose import models
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from tifffile import imwrite

# Reuse the existing TIFF grouping and image-processing helpers.
import batch_morphology_signal_analysis as pipeline


# ----------------------------- Editable settings -----------------------------
# The folder can contain treatment/well subfolders. All TIFF groups below it
# with C0, C1, and C2 are processed.
DATA_ROOT = Path(r'E:\N_segmentation\Nouveau dossier (1)\Nouveau dossier\CsA')
OUTPUT_DIR = DATA_ROOT.parent / f'{DATA_ROOT.name}_custom_cellpose_results'

CHANNEL_INDEX = 2  # Brightfield / segmentation channel.
REQUIRED_CHANNELS = (0, 1, 2)
MAX_IMAGES = None  # Set an integer for a small test run.

# Raw channel-specific thresholds. Signal is present when any masked pixel is > its threshold.
C0_SIGNAL_THRESHOLD = 300
C1_SIGNAL_THRESHOLD = 300

PREPROCESSING = {
    'clahe': False,
    'clahe_clip': 0.01,
    'clahe_tile': 4,
    'sharpen': True,
    'sharpen_amount': 3.2,
    'sharpen_radius': 0.05,
    'dog': False,
    'dog_low': 0.7,
    'dog_high': 4.0,
}

CELLPOSE = {
    'model': 'cellsam',
    'use_gpu': True,
    'flow': 0.55,
    'cellprob': -1.0,
    'min_size': 10,
}

SAVE_MASKS_AND_OVERLAYS = True
# -----------------------------------------------------------------------------


def safe_name(value: object) -> str:
    return re.sub(r'[^A-Za-z0-9._+-]+', '_', str(value)).strip('_') or 'unnamed'


def output_group_name(group: dict, index: int) -> str:
    raw = f"{group['relative_folder']}__{group['base']}{group['suffix']}"
    token = safe_name(raw)
    if len(token) > 100:
        token = f'{token[:84]}_{hashlib.sha1(raw.encode()).hexdigest()[:10]}'
    return f'{index:04d}_{token}'


def relative_folder(folder: Path, data_root: Path) -> str:
    try:
        return str(folder.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(folder.resolve())


def discover_groups(data_root: Path) -> list[dict]:
    groups = []
    for key, channels in pipeline.find_channel_groups(data_root):
        if not set(REQUIRED_CHANNELS).issubset(channels):
            continue
        folder, base, suffix = key
        groups.append({
            'folder': folder,
            'base': base,
            'suffix': suffix,
            'channels': channels,
            'image_name': channels[CHANNEL_INDEX].stem,
            'relative_folder': relative_folder(folder, data_root),
        })
    return groups


def save_signal_overlay(
    c2: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    masks: np.ndarray,
    path: Path,
) -> None:
    base = pipeline.percentile_rgb(c2)
    valid = masks > 0
    c0_signal = (c0 > C0_SIGNAL_THRESHOLD) & valid
    c1_signal = (c1 > C1_SIGNAL_THRESHOLD) & valid
    both = c0_signal & c1_signal

    # Every cell keeps a cyan boundary. Signal-positive cells also receive a
    # translucent red, green, or yellow cell highlight, with stronger color
    # on the thresholded signal pixels themselves.
    overlay = base.copy()
    c0_color = np.array((1.0, 0.05, 0.05))
    c1_color = np.array((0.05, 1.0, 0.05))
    both_color = np.array((1.0, 0.85, 0.05))

    for label in np.unique(masks):
        if label == 0:
            continue
        cell = masks == label
        has_c0 = bool(np.any(c0_signal[cell]))
        has_c1 = bool(np.any(c1_signal[cell]))
        if has_c0 and has_c1:
            color = both_color
        elif has_c0:
            color = c0_color
        elif has_c1:
            color = c1_color
        else:
            continue
        overlay[cell] = 0.78 * overlay[cell] + 0.22 * color

    c0_only = c0_signal & ~both
    c1_only = c1_signal & ~both
    overlay[c0_only] = 0.15 * overlay[c0_only] + 0.85 * c0_color
    overlay[c1_only] = 0.15 * overlay[c1_only] + 0.85 * c1_color
    overlay[both] = 0.15 * overlay[both] + 0.85 * both_color

    # Draw cyan boundaries last so the Cellpose outline stays visible.
    overlay[find_boundaries(masks, mode='outer')] = np.array((0.05, 0.75, 1.0))

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.imshow(np.clip(overlay, 0, 1))
    axis.axis('off')
    axis.legend(
        handles=[
            Patch(facecolor=(0.05, 0.75, 1.0), edgecolor='none', label='Cellpose boundary (cyan)'),
            Patch(facecolor=(1.0, 0.05, 0.05), edgecolor='none', label='C0-positive cell (cyan + red)'),
            Patch(facecolor=(0.05, 1.0, 0.05), edgecolor='none', label='C1-positive cell (cyan + green)'),
            Patch(facecolor=(1.0, 0.85, 0.05), edgecolor='none', label='C0+C1-positive cell (cyan + yellow)'),
        ],
        loc='upper right',
        framealpha=0.75,
        fontsize=8,
    )
    figure.savefig(path, dpi=160, bbox_inches='tight', pad_inches=0.05)
    plt.close(figure)


def make_summary(group: dict, records: list[dict], overlay_path: Path) -> dict:
    def count(predicate):
        return sum(bool(predicate(record)) for record in records)

    return {
        'image_name': group['image_name'],
        'relative_folder': group['relative_folder'],
        'source_c0': str(group['channels'][0]),
        'source_c1': str(group['channels'][1]),
        'source_c2': str(group['channels'][CHANNEL_INDEX]),
        'overlay_path': str(overlay_path),
        'total_cells': len(records),
        'spindle_cells': count(lambda record: record['morphology_label'] == 'spindle'),
        'spheroid_cells': count(lambda record: record['morphology_label'] == 'spheroid'),
        'spindle_cells_with_c0_signal': count(lambda record: record['morphology_label'] == 'spindle' and record['signal_c0']),
        'spindle_cells_with_c1_signal': count(lambda record: record['morphology_label'] == 'spindle' and record['signal_c1']),
        'spindle_cells_with_c0_and_c1_signal': count(lambda record: record['morphology_label'] == 'spindle' and record['signal_c0'] and record['signal_c1']),
        'spheroid_cells_with_c0_signal': count(lambda record: record['morphology_label'] == 'spheroid' and record['signal_c0']),
        'spheroid_cells_with_c1_signal': count(lambda record: record['morphology_label'] == 'spheroid' and record['signal_c1']),
        'spheroid_cells_with_c0_and_c1_signal': count(lambda record: record['morphology_label'] == 'spheroid' and record['signal_c0'] and record['signal_c1']),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def process(data_root: Path, output_dir: Path, max_images: int | None) -> None:
    data_root = data_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f'Data folder was not found: {data_root}')

    groups = discover_groups(data_root)
    if max_images is not None:
        groups = groups[:max_images]
    print(f'Found {len(groups)} complete C0/C1/C2 image groups in {data_root}')
    if not groups:
        print('No complete image groups found. Check the filename pattern and channels.')
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.PREPROCESSING = PREPROCESSING
    pipeline.CELLPOSE = CELLPOSE
    pipeline.CHANNEL_INDEX = CHANNEL_INDEX

    model = None
    all_records = []
    completed = []
    errors = []

    for index, group in enumerate(groups, start=1):
        channels = group['channels']
        group_name = output_group_name(group, index)
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        mask_path = group_dir / 'cellpose_mask.tif'
        processed_path = group_dir / 'processed_brightfield.tif'
        overlay_path = group_dir / 'cellpose_signal_overlay.png'

        print(f'[{index}/{len(groups)}] {group["relative_folder"]} / {group["image_name"]}')
        try:
            c0 = pipeline.read_tiff_2d(channels[0])
            c1 = pipeline.read_tiff_2d(channels[1])
            c2 = pipeline.read_tiff_2d(channels[CHANNEL_INDEX])
            if not (c0.shape == c1.shape == c2.shape):
                raise ValueError(
                    f'Channel shape mismatch: C0={c0.shape}, C1={c1.shape}, C2={c2.shape}'
                )

            if model is None:
                model = models.CellposeModel(
                    gpu=CELLPOSE['use_gpu'],
                    model_type=CELLPOSE['model'],
                )
            processed, masks = pipeline.run_cellpose(c2, model)
            masks = np.asarray(masks, dtype=np.int32)

            if SAVE_MASKS_AND_OVERLAYS:
                imwrite(mask_path, masks)
                imwrite(processed_path, np.asarray(processed, dtype=np.float32))
                save_signal_overlay(c2, c0, c1, masks, overlay_path)

            records = []
            for region in regionprops(masks):
                if region.area < CELLPOSE['min_size']:
                    continue
                y0, x0, y1, x1 = region.bbox
                mask_crop = masks[y0:y1, x0:x1] == region.label
                c0_crop = c0[y0:y1, x0:x1]
                c1_crop = c1[y0:y1, x0:x1]
                records.append({
                    'image_name': group['image_name'],
                    'relative_folder': group['relative_folder'],
                    'source_c2': str(channels[CHANNEL_INDEX]),
                    'label': int(region.label),
                    'area_pixels': int(region.area),
                    **pipeline.shape_features(region),
                    'signal_c0': bool(np.any(c0_crop[mask_crop] > C0_SIGNAL_THRESHOLD)),
                    'signal_c1': bool(np.any(c1_crop[mask_crop] > C1_SIGNAL_THRESHOLD)),
                })

            completed.append((group, records, overlay_path))
            all_records.extend(records)
            print(f'    Cellpose objects retained: {len(records)}')
        except Exception as error:
            errors.append({
                'image_name': group['image_name'],
                'relative_folder': group['relative_folder'],
                'source_c2': str(channels.get(CHANNEL_INDEX, '')),
                'error': f'{type(error).__name__}: {error}',
            })
            print(f'    FAILED: {type(error).__name__}: {error}')

    # Morphology labels are assigned consistently across all successfully
    # processed images in this batch.
    pipeline.assign_morphology(all_records)

    summaries = [
        make_summary(group, records, overlay_path)
        for group, records, overlay_path in completed
    ]
    cell_rows = []
    for group, records, overlay_path in completed:
        for record in records:
            cell_rows.append({
                'image_name': record['image_name'],
                'relative_folder': record['relative_folder'],
                'source_c2': record['source_c2'],
                'overlay_path': str(overlay_path),
                'cellpose_label': record['label'],
                'morphology': record['morphology_label'],
                'c0_signal': record['signal_c0'],
                'c1_signal': record['signal_c1'],
                'c0_and_c1_signal': record['signal_c0'] and record['signal_c1'],
                'area_pixels': record['area_pixels'],
                'morphology_confidence': record['morphology_confidence'],
            })

    summary_fields = [
        'image_name', 'relative_folder', 'source_c0', 'source_c1', 'source_c2',
        'overlay_path', 'total_cells', 'spindle_cells', 'spheroid_cells',
        'spindle_cells_with_c0_signal', 'spindle_cells_with_c1_signal',
        'spindle_cells_with_c0_and_c1_signal', 'spheroid_cells_with_c0_signal',
        'spheroid_cells_with_c1_signal', 'spheroid_cells_with_c0_and_c1_signal',
    ]
    detail_fields = [
        'image_name', 'relative_folder', 'source_c2', 'overlay_path',
        'cellpose_label', 'morphology', 'c0_signal', 'c1_signal',
        'c0_and_c1_signal', 'area_pixels', 'morphology_confidence',
    ]
    error_fields = ['image_name', 'relative_folder', 'source_c2', 'error']

    write_csv(output_dir / 'per_image_morphology_signal_counts.csv', summaries, summary_fields)
    write_csv(output_dir / 'per_cell_morphology_signal_details.csv', cell_rows, detail_fields)
    write_csv(output_dir / 'processing_errors.csv', errors, error_fields)

    print(f'Finished {len(completed)} images and {len(all_records)} cells.')
    print(f'Per-image counts: {output_dir / "per_image_morphology_signal_counts.csv"}')
    print(f'Per-cell details: {output_dir / "per_cell_morphology_signal_details.csv"}')
    print(f'Overlays and masks: {output_dir}')
    if errors:
        print(f'Failures: {len(errors)}; see {output_dir / "processing_errors.csv"}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run custom Cellpose preprocessing and C0/C1 signal overlays on all TIFF groups.'
    )
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--max-images', type=int, default=MAX_IMAGES)
    args = parser.parse_args()
    process(args.data_root, args.output_dir, args.max_images)


if __name__ == '__main__':
    main()
