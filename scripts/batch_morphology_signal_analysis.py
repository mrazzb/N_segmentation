from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cellpose import models
from skimage import exposure, filters, util
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from tifffile import TiffFile, imread, imwrite


# ----------------------------- Editable settings -----------------------------
DATA_DIR = Path(r'E:\N_segmentation\Nouveau dossier (1)\Nouveau dossier')
OUTPUT_DIR = DATA_DIR / 'cellpose_morphology_signal_results'
MAX_IMAGES = None  # Set an integer for a small test run.

CHANNEL_INDEX = 2  # C2 is the brightfield/segmentation channel.
SIGNAL_THRESHOLD = 400  # Raw TIFF intensity; signal is strictly greater than this.

PREPROCESSING = {
    'clahe': True,
    'clahe_clip': 0.01,
    'clahe_tile': 8,
    'sharpen': True,
    'sharpen_amount': 2.8,
    'sharpen_radius': 1.1,
    'dog': False,
    'dog_low': 0.7,
    'dog_high': 4.0,
}

CELLPOSE = {
    'model': 'cellsam',
    'use_gpu': True,
    'flow': 0.6,
    'cellprob': -1.0,
    'min_size': 10,
}

MORPHOLOGY_FEATURES = (
    'shape_aspect_ratio',
    'shape_eccentricity',
    'shape_solidity',
    'shape_extent',
    'shape_circularity',
)
LOW_CONFIDENCE_THRESHOLD = 0.10
SAVE_MASKS_AND_OVERLAYS = True
REUSE_EXISTING_MASKS = True  # Avoid rerunning Cellpose when masks already exist.
# -----------------------------------------------------------------------------

CHANNEL_PATTERN = re.compile(
    r'^(?P<base>.+?) - C=(?P<channel>\d+)(?P<suffix>_[^.]+)?$',
    re.IGNORECASE,
)


def safe_name(value: object) -> str:
    return re.sub(r'[^A-Za-z0-9._+-]+', '_', str(value)).strip('_') or 'unnamed'


def find_channel_groups(root: Path):
    groups = {}
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in {'.tif', '.tiff'}:
            continue
        match = CHANNEL_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        key = (path.parent.resolve(), match.group('base'), match.group('suffix') or '')
        groups.setdefault(key, {})[int(match.group('channel'))] = path.resolve()
    return sorted(groups.items(), key=lambda item: str(item[0]).lower())


def read_tiff_2d(path: Path) -> np.ndarray:
    with TiffFile(str(path)) as tiff:
        data = np.asarray(tiff.asarray())
        axes = list(tiff.series[0].axes)
    for axis_index in reversed(range(len(axes))):
        if axes[axis_index] not in {'Y', 'X'}:
            data = np.take(data, 0, axis=axis_index)
            axes.pop(axis_index)
    if set(axes) != {'Y', 'X'}:
        raise ValueError(f'Expected X and Y dimensions for {path.name}; got {axes!r}.')
    if axes != ['Y', 'X']:
        data = np.transpose(data, (axes.index('Y'), axes.index('X')))
    return np.asarray(data)


def normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, copy=False)
    minimum, maximum = float(image.min()), float(image.max())
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - minimum) / (maximum - minimum)).astype(np.float32, copy=False)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    result = util.img_as_float(image)
    if PREPROCESSING['clahe']:
        result = exposure.equalize_adapthist(
            result,
            kernel_size=PREPROCESSING['clahe_tile'],
            clip_limit=PREPROCESSING['clahe_clip'],
        )
    if PREPROCESSING['sharpen']:
        result = filters.unsharp_mask(
            result,
            radius=PREPROCESSING['sharpen_radius'],
            amount=PREPROCESSING['sharpen_amount'],
            preserve_range=True,
        )
    if PREPROCESSING['dog']:
        if PREPROCESSING['dog_high'] <= PREPROCESSING['dog_low']:
            raise ValueError('DoG high sigma must be greater than low sigma.')
        result = filters.difference_of_gaussians(
            result,
            low_sigma=PREPROCESSING['dog_low'],
            high_sigma=PREPROCESSING['dog_high'],
        )
    return normalize(result)


def run_cellpose(image: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    processed = preprocess_image(image)
    result = model.eval(
        processed,
        channels=[0, 0],
        channel_axis=None,
        normalize=True,
        flow_threshold=CELLPOSE['flow'],
        cellprob_threshold=CELLPOSE['cellprob'],
        min_size=CELLPOSE['min_size'],
        do_3D=False,
    )
    masks = np.asarray(result[0], dtype=np.int32)
    if masks.shape != processed.shape:
        raise ValueError(f'Cellpose returned {masks.shape}; expected {processed.shape}.')
    return processed, masks


def percentile_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        scaled = np.zeros_like(image, dtype=np.float32)
    else:
        scaled = np.clip((image - low) / (high - low), 0, 1)
    return np.repeat(scaled[..., None], 3, axis=2)


def save_mask_overlay(image: np.ndarray, masks: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
    axis.imshow(np.ma.masked_where(masks == 0, masks), cmap='nipy_spectral', alpha=0.45)
    axis.axis('off')
    figure.savefig(path, dpi=160, bbox_inches='tight', pad_inches=0)
    plt.close(figure)


def shape_features(region) -> dict[str, float]:
    major = max(float(region.axis_major_length), 1.0)
    minor = max(float(region.axis_minor_length), 1.0)
    perimeter = float(region.perimeter)
    circularity = 4.0 * np.pi * float(region.area) / max(perimeter * perimeter, 1.0)
    return {
        'shape_area': float(region.area),
        'shape_aspect_ratio': major / minor,
        'shape_eccentricity': float(region.eccentricity),
        'shape_solidity': float(region.solidity),
        'shape_extent': float(region.extent),
        'shape_circularity': circularity,
    }


def cluster_shape_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray, int | None]:
    if len(records) < 2:
        return np.zeros(len(records), dtype=int), np.zeros(len(records)), None
    values = np.asarray([[record[column] for column in MORPHOLOGY_FEATURES] for record in records], dtype=float)
    standardized = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-8)
    aspect_column = 0
    centroids = np.array([
        standardized[np.argmin(standardized[:, aspect_column])],
        standardized[np.argmax(standardized[:, aspect_column])],
    ])
    for _ in range(50):
        distances = ((standardized[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        updated = np.array([
            standardized[assignments == cluster].mean(axis=0)
            if np.any(assignments == cluster) else centroids[cluster]
            for cluster in (0, 1)
        ])
        if np.allclose(updated, centroids):
            break
        centroids = updated
    distances = ((standardized[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    assignments = distances.argmin(axis=1)
    confidence = np.abs(distances[:, 0] - distances[:, 1]) / np.maximum(distances.sum(axis=1), 1e-8)
    spheroid_cluster = int(np.argmin(centroids[:, aspect_column]))
    return assignments, confidence, spheroid_cluster


def assign_morphology(records: list[dict]) -> None:
    assignments, confidence, spheroid_cluster = cluster_shape_features(records)
    for index, record in enumerate(records):
        if spheroid_cluster is None:
            label = 'unknown'
        else:
            label = 'spheroid' if int(assignments[index]) == spheroid_cluster else 'spindle'
        record['morphology_label'] = label
        record['morphology_cluster'] = int(assignments[index])
        record['morphology_confidence'] = float(confidence[index])
        record['manual_review'] = 'yes' if confidence[index] < LOW_CONFIDENCE_THRESHOLD else 'no'


def signal_overlay(image: np.ndarray, masks: np.ndarray, records: list[dict]) -> np.ndarray:
    rgb = percentile_rgb(image)
    colors = {
        'c0': np.array((1.0, 0.05, 0.05)),
        'c1': np.array((0.05, 1.0, 0.05)),
        'both': np.array((1.0, 0.85, 0.05)),
    }
    bbox_by_label = {int(region.label): region.bbox for region in regionprops(masks)}
    for record in records:
        if record['signal_c0'] and record['signal_c1']:
            color = colors['both']
        elif record['signal_c0']:
            color = colors['c0']
        elif record['signal_c1']:
            color = colors['c1']
        else:
            continue
        y0, x0, y1, x1 = bbox_by_label[record['label']]
        mask_crop = masks[y0:y1, x0:x1] == record['label']
        rgb_crop = rgb[y0:y1, x0:x1]
        rgb_crop[mask_crop] = 0.35 * rgb_crop[mask_crop] + 0.65 * color
    rgb[find_boundaries(masks, mode='outer')] = (1.0, 1.0, 1.0)
    return np.clip(rgb, 0, 1)


def make_summary(group: dict, records: list[dict]) -> dict:
    def count(predicate):
        return sum(bool(predicate(record)) for record in records)

    return {
        'image_name': Path(group['channels'][CHANNEL_INDEX]).stem,
        'overlay_path': group.get('overlay_path', ''),
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


def main() -> None:
    data_dir = DATA_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f'Dataset folder was not found: {data_dir}')

    groups = find_channel_groups(data_dir)
    required_channels = {0, 1, CHANNEL_INDEX}
    eligible = [(key, channels) for key, channels in groups if required_channels.issubset(channels)]
    if MAX_IMAGES is not None:
        eligible = eligible[:MAX_IMAGES]
    print(f'Found {len(groups)} image groups; processing {len(eligible)} complete groups.')
    if len(eligible) != len(groups):
        print(f'Skipping {len(groups) - len(eligible)} groups missing one of C0/C1/C2.')

    model = None
    all_records = []
    group_results = []
    errors = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (key, channels) in enumerate(eligible, start=1):
        folder, base, suffix = key
        treatment = str(folder.relative_to(data_dir)) if folder != data_dir else '.'
        group_name = safe_name('__'.join(filter(None, [treatment, base + suffix])))
        group = {
            'treatment': treatment,
            'group': base + suffix,
            'group_name': group_name,
            'channels': channels,
        }
        print(f'[{index}/{len(eligible)}] {treatment} / {base}{suffix}')
        try:
            c0 = read_tiff_2d(channels[0])
            c1 = read_tiff_2d(channels[1])
            c2 = read_tiff_2d(channels[CHANNEL_INDEX])
            if not (c0.shape == c1.shape == c2.shape):
                raise ValueError(f'Channel shape mismatch: C0={c0.shape}, C1={c1.shape}, C2={c2.shape}')

            group_dir = output_dir / group_name
            group_dir.mkdir(parents=True, exist_ok=True)
            mask_path = group_dir / f'{group_name}_cellpose_mask.tif'
            processed_path = group_dir / f'{group_name}_processed.tif'
            mask_overlay_path = group_dir / 'cellpose_overlay.png'
            # Keep the optional overlay filename short; Windows path limits can
            # otherwise be exceeded by the original microscope filename.
            signal_overlay_path = group_dir / 'signal_overlay.png'

            if REUSE_EXISTING_MASKS and mask_path.exists():
                masks = np.asarray(imread(mask_path), dtype=np.int32)
                if masks.shape != c2.shape:
                    raise ValueError(f'Existing mask has shape {masks.shape}; expected {c2.shape}.')
                processed = imread(processed_path) if processed_path.exists() else preprocess_image(c2)
                print('    Reusing existing Cellpose mask')
            else:
                if model is None:
                    model = models.CellposeModel(gpu=CELLPOSE['use_gpu'], model_type=CELLPOSE['model'])
                processed, masks = run_cellpose(c2, model)

            if SAVE_MASKS_AND_OVERLAYS:
                if not mask_path.exists():
                    imwrite(mask_path, masks.astype(np.int32))
                if not processed_path.exists():
                    imwrite(processed_path, np.asarray(processed, dtype=np.float32))
                if not mask_overlay_path.exists():
                    save_mask_overlay(c2, masks, mask_overlay_path)

            records = []
            for region in regionprops(masks):
                if region.area < CELLPOSE['min_size']:
                    continue
                y0, x0, y1, x1 = region.bbox
                mask_crop = masks[y0:y1, x0:x1] == region.label
                c0_crop = c0[y0:y1, x0:x1]
                c1_crop = c1[y0:y1, x0:x1]
                features = shape_features(region)
                records.append({
                    'treatment': treatment,
                    'group': base + suffix,
                    'group_name': group_name,
                    'source_c2': str(channels[CHANNEL_INDEX]),
                    'label': int(region.label),
                    'area_pixels': int(region.area),
                    **features,
                    'signal_c0': bool(np.any(c0_crop[mask_crop] > SIGNAL_THRESHOLD)),
                    'signal_c1': bool(np.any(c1_crop[mask_crop] > SIGNAL_THRESHOLD)),
                })

            # Keep paths, not full-resolution arrays, so large batches remain memory bounded.
            group_results.append((group, records, channels[CHANNEL_INDEX], mask_path, signal_overlay_path if SAVE_MASKS_AND_OVERLAYS else None))
            all_records.extend(records)
            print(f'    Cellpose objects retained: {len(records)}')
        except Exception as error:
            errors.append({
                'treatment': treatment,
                'group': base + suffix,
                'error': f'{type(error).__name__}: {error}',
            })
            print(f'    FAILED: {type(error).__name__}: {error}')

    assign_morphology(all_records)
    summaries = []
    for group, records, c2_path, mask_path, signal_overlay_path in group_results:
        group['overlay_path'] = str(signal_overlay_path) if signal_overlay_path is not None else ''
        summaries.append(make_summary(group, records))
        c2 = None
        masks = None
        if SAVE_MASKS_AND_OVERLAYS:
            c2 = read_tiff_2d(Path(c2_path))
            masks = np.asarray(imread(mask_path), dtype=np.int32)
            for record in records:
                record['overlay_path'] = str(signal_overlay_path)

        if SAVE_MASKS_AND_OVERLAYS and signal_overlay_path is not None:
            figure = None
            try:
                figure, axis = plt.subplots(figsize=(10, 8))
                axis.imshow(signal_overlay(c2, masks, records))
                axis.axis('off')
                figure.savefig(signal_overlay_path, dpi=160, bbox_inches='tight', pad_inches=0)
            except Exception as error:
                errors.append({
                    'treatment': group['treatment'],
                    'group': group['group'],
                    'error': f'signal overlay: {type(error).__name__}: {error}',
                })
                print(f"    Signal overlay failed for {group['group']}: {type(error).__name__}: {error}")
            finally:
                if figure is not None:
                    plt.close(figure)


    summary_fields = [
        'image_name', 'overlay_path', 'total_cells', 'spindle_cells', 'spheroid_cells',
        'spindle_cells_with_c0_signal', 'spindle_cells_with_c1_signal',
        'spindle_cells_with_c0_and_c1_signal', 'spheroid_cells_with_c0_signal',
        'spheroid_cells_with_c1_signal', 'spheroid_cells_with_c0_and_c1_signal',
    ]
    with (output_dir / 'per_image_morphology_signal_counts.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    cell_rows = []
    for record in all_records:
        cell_rows.append({
            'image_name': Path(record['source_c2']).stem,
            'overlay_path': record.get('overlay_path', ''),
            'cellpose_label': record['label'],
            'morphology': record['morphology_label'],
            'c0_signal': record['signal_c0'],
            'c1_signal': record['signal_c1'],
            'c0_and_c1_signal': record['signal_c0'] and record['signal_c1'],
            'area_pixels': record['area_pixels'],
            'morphology_confidence': record['morphology_confidence'],
        })
    detail_fields = [
        'image_name', 'overlay_path', 'cellpose_label', 'morphology',
        'c0_signal', 'c1_signal', 'c0_and_c1_signal', 'area_pixels',
        'morphology_confidence',
    ]
    with (output_dir / 'per_cell_morphology_signal_details.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(cell_rows)

    if errors:
        with (output_dir / 'processing_errors.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['treatment', 'group', 'error'])
            writer.writeheader()
            writer.writerows(errors)

    print(f'Finished {len(group_results)} images and {len(all_records)} cells.')
    print(f'Per-image counts: {output_dir / "per_image_morphology_signal_counts.csv"}')
    print(f'Per-cell details: {output_dir / "per_cell_morphology_signal_details.csv"}')
    if errors:
        print(f'Failures: {len(errors)}; see {output_dir / "processing_errors.csv"}')


if __name__ == '__main__':
    main()