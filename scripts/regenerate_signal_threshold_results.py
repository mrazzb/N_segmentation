from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
from skimage.measure import regionprops
from tifffile import imread

# Reuse the current overlay appearance and TIFF helpers, but never call Cellpose.
import batch_custom_cellpose_signal_analysis as custom
import batch_morphology_signal_analysis as pipeline


# ----------------------------- Editable settings -----------------------------
RESULTS_DIR = Path(
    r'E:\N_segmentation\Nouveau dossier_custom_cellpose_results'
)
PER_IMAGE_CSV = RESULTS_DIR / 'per_image_morphology_signal_counts.csv'
PER_CELL_CSV = RESULTS_DIR / 'per_cell_morphology_signal_details.csv'
OUTPUT_DIR = RESULTS_DIR / 'threshold_regenerated'

C0_SIGNAL_THRESHOLD = 300
C1_SIGNAL_THRESHOLD = 300
# -----------------------------------------------------------------------------


def canonical_path(value: object) -> str:
    text = str(value or '').strip().strip('"')
    if not text:
        return ''
    return str(Path(text).expanduser().resolve(strict=False)).casefold()


def safe_name(value: object) -> str:
    return re.sub(r'[^A-Za-z0-9._+-]+', '_', str(value)).strip('_') or 'unnamed'


def read_csv(path: Path) -> list[dict]:
    with path.open('r', newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f'CSV has no header: {path}')
        return list(reader)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def resolve_source_path(value: object, results_dir: Path) -> Path:
    path = Path(str(value or '').strip().strip('"')).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    candidates = [
        results_dir / path,
        results_dir.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


def find_channels_for_c2(c2_path: Path) -> dict[int, Path]:
    c2_key = canonical_path(c2_path)
    for key, channels in pipeline.find_channel_groups(c2_path.parent):
        if canonical_path(channels.get(2, '')) == c2_key:
            return channels
    raise FileNotFoundError(
        f'Could not find matching C0/C1 files for C2 image: {c2_path}'
    )


def find_mask_path(image_row: dict, results_dir: Path) -> Path:
    overlay_value = image_row.get('overlay_path', '')
    if overlay_value:
        overlay_path = resolve_source_path(overlay_value, results_dir)
        candidate = overlay_path.parent / 'cellpose_mask.tif'
        if candidate.exists():
            return candidate

    matches = list(results_dir.rglob('cellpose_mask.tif'))
    image_name = str(image_row.get('image_name', '')).casefold()
    matching_parent = [
        path for path in matches
        if image_name and image_name in path.parent.name.casefold()
    ]
    if len(matching_parent) == 1:
        return matching_parent[0]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f'Could not locate cellpose_mask.tif for image {image_name!r}. '
        f'Expected it beside the previous overlay.'
    )


def cell_key(source_c2: object, label: object, image_name: object = '') -> tuple[str, str, str]:
    return (
        canonical_path(source_c2),
        str(image_name or '').casefold(),
        str(int(float(label))),
    )


def make_summary(image: dict, records: list[dict], overlay_path: Path) -> dict:
    def count(predicate):
        return sum(bool(predicate(record)) for record in records)

    return {
        'image_name': image['image_name'],
        'relative_folder': image.get('relative_folder', ''),
        'source_c0': image['source_c0'],
        'source_c1': image['source_c1'],
        'source_c2': image['source_c2'],
        'overlay_path': str(overlay_path),
        'c0_signal_threshold': C0_SIGNAL_THRESHOLD,
        'c1_signal_threshold': C1_SIGNAL_THRESHOLD,
        'total_cells': len(records),
        'spindle_cells': count(lambda record: record['morphology'] == 'spindle'),
        'spheroid_cells': count(lambda record: record['morphology'] == 'spheroid'),
        'spindle_cells_with_c0_signal': count(lambda record: record['morphology'] == 'spindle' and record['c0_signal']),
        'spindle_cells_with_c1_signal': count(lambda record: record['morphology'] == 'spindle' and record['c1_signal']),
        'spindle_cells_with_c0_and_c1_signal': count(lambda record: record['morphology'] == 'spindle' and record['c0_signal'] and record['c1_signal']),
        'spheroid_cells_with_c0_signal': count(lambda record: record['morphology'] == 'spheroid' and record['c0_signal']),
        'spheroid_cells_with_c1_signal': count(lambda record: record['morphology'] == 'spheroid' and record['c1_signal']),
        'spheroid_cells_with_c0_and_c1_signal': count(lambda record: record['morphology'] == 'spheroid' and record['c0_signal'] and record['c1_signal']),
    }


def run(results_dir: Path, per_image_csv: Path, per_cell_csv: Path, output_dir: Path) -> None:
    results_dir = results_dir.expanduser().resolve()
    per_image_csv = per_image_csv.expanduser().resolve()
    per_cell_csv = per_cell_csv.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not per_image_csv.exists():
        raise FileNotFoundError(f'Per-image CSV was not found: {per_image_csv}')
    if not per_cell_csv.exists():
        raise FileNotFoundError(f'Per-cell CSV was not found: {per_cell_csv}')

    image_rows = read_csv(per_image_csv)
    old_cell_rows = read_csv(per_cell_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The previous batch output stores morphology labels. Reuse them so changing
    # thresholds does not change the morphology classification.
    cells_by_key: dict[tuple[str, str, str], dict] = {}
    for row in old_cell_rows:
        key = cell_key(row.get('source_c2', ''), row.get('cellpose_label', ''), row.get('image_name', ''))
        cells_by_key[key] = row

    custom.C0_SIGNAL_THRESHOLD = C0_SIGNAL_THRESHOLD
    custom.C1_SIGNAL_THRESHOLD = C1_SIGNAL_THRESHOLD

    summaries = []
    detail_rows = []
    errors = []

    for index, image in enumerate(image_rows, start=1):
        image_name = image.get('image_name', f'image_{index:04d}')
        try:
            c2_path = resolve_source_path(image.get('source_c2', ''), results_dir)
            if not c2_path.exists():
                raise FileNotFoundError(f'C2 source image was not found: {c2_path}')

            channels = {}
            if image.get('source_c0'):
                channels[0] = resolve_source_path(image['source_c0'], results_dir)
            if image.get('source_c1'):
                channels[1] = resolve_source_path(image['source_c1'], results_dir)
            channels[2] = c2_path
            if 0 not in channels or 1 not in channels or not channels[0].exists() or not channels[1].exists():
                discovered = find_channels_for_c2(c2_path)
                channels.update(discovered)

            c0 = pipeline.read_tiff_2d(channels[0])
            c1 = pipeline.read_tiff_2d(channels[1])
            c2 = pipeline.read_tiff_2d(channels[2])
            mask_path = find_mask_path(image, results_dir)
            masks = np.asarray(imread(mask_path), dtype=np.int32)
            if not (c0.shape == c1.shape == c2.shape == masks.shape):
                raise ValueError(
                    f'Shape mismatch: C0={c0.shape}, C1={c1.shape}, '
                    f'C2={c2.shape}, mask={masks.shape}'
                )

            overlay_path = output_dir / (
                f'{index:04d}_{safe_name(image_name)}_signal_overlay.png'
            )
            custom.save_signal_overlay(c2, c0, c1, masks, overlay_path)

            records = []
            for region in regionprops(masks):
                if region.area < custom.CELLPOSE['min_size']:
                    continue
                label = int(region.label)
                key = cell_key(channels[2], label, image_name)
                old_cell = cells_by_key.get(key)
                if old_cell is None:
                    key = cell_key(channels[2], label, '')
                    old_cell = cells_by_key.get(key)
                if old_cell is None:
                    raise KeyError(
                        f'No previous per-cell row for {image_name!r}, label {label}.'
                    )

                cell = masks == label
                c0_signal = bool(np.any(c0[cell] > C0_SIGNAL_THRESHOLD))
                c1_signal = bool(np.any(c1[cell] > C1_SIGNAL_THRESHOLD))
                records.append({
                    'image_name': image_name,
                    'relative_folder': image.get('relative_folder', ''),
                    'source_c2': str(channels[2]),
                    'overlay_path': str(overlay_path),
                    'cellpose_label': label,
                    'morphology': old_cell.get('morphology', 'unknown'),
                    'c0_signal': c0_signal,
                    'c1_signal': c1_signal,
                    'c0_and_c1_signal': c0_signal and c1_signal,
                    'area_pixels': old_cell.get('area_pixels', int(region.area)),
                    'morphology_confidence': old_cell.get('morphology_confidence', ''),
                    'c0_signal_threshold': C0_SIGNAL_THRESHOLD,
                    'c1_signal_threshold': C1_SIGNAL_THRESHOLD,
                })

            summaries.append(make_summary({
                'image_name': image_name,
                'relative_folder': image.get('relative_folder', ''),
                'source_c0': str(channels[0]),
                'source_c1': str(channels[1]),
                'source_c2': str(channels[2]),
            }, records, overlay_path))
            detail_rows.extend(records)
            print(f'[{index}/{len(image_rows)}] {image_name}: {len(records)} cells')
        except Exception as error:
            errors.append({
                'image_name': image_name,
                'relative_folder': image.get('relative_folder', ''),
                'error': f'{type(error).__name__}: {error}',
            })
            print(f'[{index}/{len(image_rows)}] FAILED {image_name}: {type(error).__name__}: {error}')

    summary_fields = [
        'image_name', 'relative_folder', 'source_c0', 'source_c1', 'source_c2',
        'overlay_path', 'c0_signal_threshold', 'c1_signal_threshold',
        'total_cells', 'spindle_cells', 'spheroid_cells',
        'spindle_cells_with_c0_signal', 'spindle_cells_with_c1_signal',
        'spindle_cells_with_c0_and_c1_signal', 'spheroid_cells_with_c0_signal',
        'spheroid_cells_with_c1_signal', 'spheroid_cells_with_c0_and_c1_signal',
    ]
    detail_fields = [
        'image_name', 'relative_folder', 'source_c2', 'overlay_path',
        'cellpose_label', 'morphology', 'c0_signal', 'c1_signal',
        'c0_and_c1_signal', 'area_pixels', 'morphology_confidence',
        'c0_signal_threshold', 'c1_signal_threshold',
    ]
    error_fields = ['image_name', 'relative_folder', 'error']

    write_csv(output_dir / 'per_image_morphology_signal_counts.csv', summaries, summary_fields)
    write_csv(output_dir / 'per_cell_morphology_signal_details.csv', detail_rows, detail_fields)
    write_csv(output_dir / 'processing_errors.csv', errors, error_fields)
    print(f'Regenerated overlays and counts in: {output_dir}')
    if errors:
        print(f'Failures: {len(errors)}; see {output_dir / "processing_errors.csv"}')


def main() -> None:
    global C0_SIGNAL_THRESHOLD, C1_SIGNAL_THRESHOLD
    parser = argparse.ArgumentParser(
        description='Regenerate C0/C1 overlays and counts from existing masks without Cellpose.'
    )
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--per-image-csv', type=Path, default=PER_IMAGE_CSV)
    parser.add_argument('--per-cell-csv', type=Path, default=PER_CELL_CSV)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--c0-threshold', type=float, default=C0_SIGNAL_THRESHOLD)
    parser.add_argument('--c1-threshold', type=float, default=C1_SIGNAL_THRESHOLD)
    args = parser.parse_args()

    C0_SIGNAL_THRESHOLD = args.c0_threshold
    C1_SIGNAL_THRESHOLD = args.c1_threshold
    run(args.results_dir, args.per_image_csv, args.per_cell_csv, args.output_dir)


if __name__ == '__main__':
    main()
