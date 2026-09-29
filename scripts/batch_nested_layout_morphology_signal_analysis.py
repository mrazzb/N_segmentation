from __future__ import annotations

import csv
import sys
from pathlib import Path

# Make the existing pipeline importable when this file is run directly.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import batch_morphology_signal_analysis as pipeline


# ----------------------------- Editable settings -----------------------------
# This is the folder containing treatment folders such as CsA, DMSO, or DTX.
# Each treatment may contain any number of final-level folders such as A1, B1,
# E1, E2, and so on. Every final-level folder is searched recursively.
DATA_ROOT = Path(r'E:\N_segmentation\Nouveau dossier (1)\Nouveau dossier')
OUTPUT_DIR = DATA_ROOT.parent / f'{DATA_ROOT.name}_morphology_signal_results'
MAX_IMAGES = None  # Set an integer for a small test run.
CHANNEL_INDEX = 2
REQUIRED_CHANNELS = (0, 1, 2)
# -----------------------------------------------------------------------------


def layout_metadata(folder: Path, data_root: Path) -> dict[str, str]:
    relative = folder.relative_to(data_root)
    parts = relative.parts
    return {
        'dataset': data_root.name,
        'treatment': parts[-2] if len(parts) >= 2 else '',
        'well': parts[-1] if parts else folder.name,
        'relative_folder': str(relative),
    }


def build_layout_rows(data_root: Path) -> tuple[list[dict], list[dict]]:
    groups = pipeline.find_channel_groups(data_root)
    rows = []
    complete_groups = []
    for (folder, base, suffix), channels in groups:
        metadata = layout_metadata(folder, data_root)
        image_name = channels[CHANNEL_INDEX].stem if CHANNEL_INDEX in channels else f'{base}{suffix}'
        row = {
            **metadata,
            'image_name': image_name,
            'source_folder': str(folder),
            'base_name': base,
            'suffix': suffix,
            'has_c0': 0 in channels,
            'has_c1': 1 in channels,
            'has_c2': 2 in channels,
            'source_c0': str(channels.get(0, '')),
            'source_c1': str(channels.get(1, '')),
            'source_c2': str(channels.get(2, '')),
        }
        rows.append(row)
        if set(REQUIRED_CHANNELS).issubset(channels):
            complete_groups.append(row)
    return rows, complete_groups


def write_layout_manifest(rows: list[dict], path: Path) -> None:
    fields = [
        'dataset', 'treatment', 'well', 'relative_folder', 'image_name',
        'source_folder', 'base_name', 'suffix', 'has_c0', 'has_c1', 'has_c2',
        'source_c0', 'source_c1', 'source_c2',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_layout_to_results(layout_rows: list[dict], output_dir: Path) -> None:
    layout_by_image = {row['image_name']: row for row in layout_rows}
    image_csv = output_dir / 'per_image_morphology_signal_counts.csv'
    cell_csv = output_dir / 'per_cell_morphology_signal_details.csv'

    if image_csv.exists():
        rows = list(csv.DictReader(image_csv.open(encoding='utf-8')))
        if rows:
            fields = [
                'dataset', 'treatment', 'well', 'relative_folder',
                *rows[0].keys(),
            ]
            for row in rows:
                layout = layout_by_image.get(row['image_name'], {})
                row.update({key: layout.get(key, '') for key in fields[:4]})
            with image_csv.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    if cell_csv.exists():
        rows = list(csv.DictReader(cell_csv.open(encoding='utf-8')))
        if rows:
            fields = [
                'dataset', 'treatment', 'well', 'relative_folder',
                *rows[0].keys(),
            ]
            for row in rows:
                layout = layout_by_image.get(row['image_name'], {})
                row.update({key: layout.get(key, '') for key in fields[:4]})
            with cell_csv.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def main() -> None:
    data_root = DATA_ROOT.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f'Data folder was not found: {data_root}')

    layout_rows, complete_groups = build_layout_rows(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_layout_manifest(layout_rows, output_dir / 'layout_manifest.csv')
    print(f'Found {len(layout_rows)} image groups in {data_root}')
    print(f'Complete C0/C1/C2 groups: {len(complete_groups)}')
    print(f'Final-level folders: {len({row["source_folder"] for row in layout_rows})}')

    # Configure the unchanged, tested pipeline for this new input root/output.
    pipeline.DATA_DIR = data_root
    pipeline.OUTPUT_DIR = output_dir
    pipeline.MAX_IMAGES = MAX_IMAGES
    pipeline.CHANNEL_INDEX = CHANNEL_INDEX
    pipeline.REUSE_EXISTING_MASKS = True
    pipeline.SAVE_MASKS_AND_OVERLAYS = True
    pipeline.main()

    add_layout_to_results(layout_rows, output_dir)
    print(f'Layout manifest: {output_dir / "layout_manifest.csv"}')
    print(f'Results: {output_dir}')


if __name__ == '__main__':
    main()