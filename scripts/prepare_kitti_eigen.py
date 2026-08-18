"""Prepare the 652-sample Marigold/ZipDepth KITTI Eigen evaluation layout.

The Marigold split contains 697 RGB entries, but only 652 have improved KITTI
depth-benchmark ground truth.  ZipDepth reports the 652-sample protocol from
Marigold.  This script creates the layout expected by ``scripts/eval.py`` using
symbolic links, without copying the image or depth data.
"""

import argparse
from pathlib import Path


def _existing_file(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _ensure_symlink(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        raise FileExistsError(
            f'Existing symlink points to a different source: {destination}'
        )
    if destination.exists():
        raise FileExistsError(f'Refusing to replace existing path: {destination}')
    destination.symlink_to(source)


def prepare(kitti_root: Path, split_file: Path, output_dir: Path) -> int:
    """Validate all records, then create the ZipDepth-compatible link tree."""
    kitti_root = kitti_root.resolve()
    split_file = split_file.resolve()
    output_dir = output_dir.resolve()
    if not split_file.is_file():
        raise FileNotFoundError(f'Split file not found: {split_file}')

    records = []
    errors = []
    seen = set()
    for line_number, raw_line in enumerate(
        split_file.read_text(encoding='utf-8').splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        columns = line.split()
        if len(columns) < 2:
            errors.append(f'line {line_number}: expected at least two columns')
            continue

        rgb_rel, gt_rel = Path(columns[0]), columns[1]
        # The 45 entries without improved ground truth are intentionally not
        # part of the 652-sample benchmark used by Marigold and ZipDepth.
        if gt_rel == 'None':
            continue
        if len(rgb_rel.parts) < 5:
            errors.append(f'line {line_number}: invalid RGB path: {rgb_rel}')
            continue

        date, drive = rgb_rel.parts[0], rgb_rel.parts[1]
        gt_rel_path = Path(gt_rel)
        rgb_source = (kitti_root / 'raw_data' / rgb_rel).resolve()
        gt_source = _existing_file([
            kitti_root / 'val' / gt_rel_path,
            kitti_root / 'train' / date / gt_rel_path,
            kitti_root / 'train' / gt_rel_path,
        ])

        if not rgb_source.is_file():
            errors.append(f'line {line_number}: RGB not found: {rgb_source}')
            continue
        if gt_source is None:
            errors.append(f'line {line_number}: GT not found: {gt_rel}')
            continue

        sample_key = (date, drive, rgb_rel.name)
        if sample_key in seen:
            errors.append(f'line {line_number}: duplicate sample: {sample_key}')
            continue
        seen.add(sample_key)

        drive_output = output_dir / date / drive
        records.append((
            rgb_source,
            drive_output / 'image_02' / 'data' / rgb_rel.name,
            gt_source,
            drive_output / 'proj_depth' / 'groundtruth' / 'image_02' / rgb_rel.name,
        ))

    if errors:
        preview = '\n  '.join(errors[:10])
        suffix = f'\n  ... and {len(errors) - 10} more' if len(errors) > 10 else ''
        raise RuntimeError(
            f'Dataset validation failed with {len(errors)} error(s):\n  '
            f'{preview}{suffix}'
        )
    if len(records) != 652:
        raise RuntimeError(
            f'Expected exactly 652 samples with improved GT, found {len(records)}'
        )

    for rgb_source, rgb_destination, gt_source, gt_destination in records:
        _ensure_symlink(rgb_destination, rgb_source)
        _ensure_symlink(gt_destination, gt_source)

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Prepare the 652-sample KITTI Eigen evaluation set for ZipDepth'
    )
    parser.add_argument('--kitti-root', type=Path, required=True,
                        help='KITTI root containing raw_data/, train/, and val/')
    parser.add_argument('--split-file', type=Path, required=True,
                        help='Marigold eigen_test_files_with_gt.txt')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('datasets/kitti_eigen'),
                        help='Destination link tree (default: datasets/kitti_eigen)')
    args = parser.parse_args()

    count = prepare(args.kitti_root, args.split_file, args.output_dir)
    print(f'Prepared {count} KITTI Eigen samples at {args.output_dir.resolve()}')
    print(f'Created {count * 2} data symlinks (RGB + improved depth GT)')


if __name__ == '__main__':
    main()
