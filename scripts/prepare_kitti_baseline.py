"""Freeze and validate the data contracts for the KITTI-only baseline.

This script consumes the already-generated Phase10B teacher cache. It never
imports or executes Unified Flow Field code.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


EXPECTED_TRAIN_SAMPLES = 23_158
EXPECTED_EIGEN_SAMPLES = 652


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _fixed_bytes(values: list[str]) -> np.ndarray:
    width = max(len(value.encode('utf-8')) for value in values) + 1
    return np.asarray(values, dtype=f'S{width}')


def _validate_cache_payload(
    entry: dict,
    row: dict[str, str],
    teacher_identity: str,
    cache_version: str,
) -> tuple[float, float]:
    path = Path(entry['cache_path'])
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError(f'cannot load cache payload: {path}') from error
    if not isinstance(payload, dict) or set(payload) != {'metadata', 'depth'}:
        raise ValueError(f'cache schema mismatch: {path}')
    metadata, depth = payload['metadata'], payload['depth']
    expected_shape = [1, 1, int(row['height']), int(row['width'])]
    expected = {
        'sample_id': row['sample_id'],
        'image_sha256': row['sha256'],
        'teacher_identity': teacher_identity,
        'cache_version': cache_version,
        'shape': expected_shape,
        'dtype': 'float32',
        'finite': True,
    }
    for key, value in expected.items():
        if not isinstance(metadata, dict) or metadata.get(key) != value:
            raise ValueError(
                f'{path}: metadata {key}={metadata.get(key)!r} != {value!r}'
            )
    if (
        not isinstance(depth, torch.Tensor)
        or list(depth.shape) != expected_shape
        or depth.dtype != torch.float32
        or depth.requires_grad
        or depth.grad_fn is not None
        or not bool(torch.isfinite(depth).all().item())
    ):
        raise ValueError(f'{path}: tensor shape/dtype/finite validation failed')
    return float(depth.min().item()), float(depth.max().item())


def build_train_index(args) -> dict:
    rows = read_csv(args.resolved_train_manifest)
    cache_document = json.loads(args.cache_manifest.read_text(encoding='utf-8'))
    entries = cache_document.get('entries', [])
    if len(rows) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(f'expected {EXPECTED_TRAIN_SAMPLES} train rows, got {len(rows)}')
    if cache_document.get('entry_count') != EXPECTED_TRAIN_SAMPLES:
        raise ValueError('cache manifest entry_count is not 23,158')
    if len(entries) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(f'expected {EXPECTED_TRAIN_SAMPLES} cache entries, got {len(entries)}')

    row_ids = [row['sample_id'] for row in rows]
    entry_ids = [entry['sample_id'] for entry in entries]
    if len(set(row_ids)) != len(row_ids) or len(set(entry_ids)) != len(entry_ids):
        raise ValueError('duplicate sample_id in train or cache manifest')
    entry_by_id = {entry['sample_id']: entry for entry in entries}
    if set(row_ids) != set(entry_ids):
        missing = sorted(set(row_ids) - set(entry_ids))
        extra = sorted(set(entry_ids) - set(row_ids))
        raise ValueError(f'train/cache ID mismatch: missing={missing[:3]}, extra={extra[:3]}')

    cache_root = args.cache_root.resolve()
    cache_files = {path.resolve() for path in cache_root.rglob('*.pt')}
    manifest_files = {Path(entry['cache_path']).resolve() for entry in entries}
    if cache_files != manifest_files:
        raise ValueError(
            f'cache directory/manifest mismatch: directory={len(cache_files)}, '
            f'manifest={len(manifest_files)}, missing={len(manifest_files-cache_files)}, '
            f'extra={len(cache_files-manifest_files)}'
        )

    teacher_identity = cache_document.get('teacher_identity')
    cache_version = cache_document.get('cache_version')
    rgb_paths, depth_paths, sample_ids, image_shas = [], [], [], []
    global_min, global_max = float('inf'), float('-inf')
    above_uint16_x256_range = 0

    for row in tqdm(rows, desc='Validating 23,158 train/cache pairs', unit='sample'):
        entry = entry_by_id[row['sample_id']]
        rgb_path = Path(row['resolved_source_path']).resolve()
        cache_path = Path(entry['cache_path']).resolve()
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        if not cache_path.is_file() or cache_root not in cache_path.parents:
            raise FileNotFoundError(cache_path)
        if entry['image_sha256'] != row['sha256']:
            raise ValueError(f'image SHA mismatch for {row["sample_id"]}')
        expected_shape = [1, 1, int(row['height']), int(row['width'])]
        if entry.get('shape') != expected_shape:
            raise ValueError(f'cache manifest shape mismatch for {row["sample_id"]}')
        if args.verify_rgb_hashes and sha256_file(rgb_path) != row['sha256']:
            raise ValueError(f'RGB file content SHA mismatch: {rgb_path}')
        if args.verify_payloads:
            item_min, item_max = _validate_cache_payload(
                entry, row, teacher_identity, cache_version
            )
            global_min = min(global_min, item_min)
            global_max = max(global_max, item_max)
            above_uint16_x256_range += int(item_max > (65535.0 / 256.0))

        rgb_paths.append(str(rgb_path))
        depth_paths.append(str(cache_path))
        sample_ids.append(row['sample_id'])
        image_shas.append(row['sha256'])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.output.with_suffix('')
    np.save(f'{prefix}_rgb.npy', _fixed_bytes(rgb_paths))
    np.save(f'{prefix}_depth.npy', _fixed_bytes(depth_paths))
    np.save(f'{prefix}_domain.npy', _fixed_bytes(['kitti'] * len(rows)))
    np.save(f'{prefix}_sample_id.npy', _fixed_bytes(sample_ids))
    np.save(f'{prefix}_image_sha256.npy', _fixed_bytes(image_shas))

    verification = {
        'sample_count': len(rows),
        'unique_sample_id_count': len(set(sample_ids)),
        'cache_file_count': len(cache_files),
        'all_rgb_files_exist': True,
        'all_cache_files_exist': True,
        'all_sample_ids_match': True,
        'all_image_sha256_match': True,
        'all_shapes_match': True,
        'rgb_content_hashes_verified': bool(args.verify_rgb_hashes),
        'all_float32_payloads_verified': bool(args.verify_payloads),
        'pseudo_depth_global_min': global_min if args.verify_payloads else None,
        'pseudo_depth_global_max': global_max if args.verify_payloads else None,
        'samples_exceeding_uint16_x256_range': (
            above_uint16_x256_range if args.verify_payloads else None
        ),
    }
    metadata = {
        'version': 'kitti_only_zipdepth_train_index_v1',
        'total_samples': len(rows),
        'domains': ['kitti'],
        'depth_format': 'phase10b_dav2_l_float32_pt',
        'depth_scale': 1.0,
        'strict_loading': True,
        'cache_version': cache_version,
        'teacher_identity': teacher_identity,
        'source_resolved_train_manifest': str(args.resolved_train_manifest.resolve()),
        'source_resolved_train_manifest_sha256': sha256_file(args.resolved_train_manifest),
        'source_cache_manifest': str(args.cache_manifest.resolve()),
        'source_cache_manifest_sha256': sha256_file(args.cache_manifest),
        'cache_root': str(cache_root),
        'verification': verification,
    }
    encoded = json.dumps(metadata, indent=2, sort_keys=True) + '\n'
    args.output.write_text(encoded, encoding='utf-8')
    Path(f'{prefix}_metadata.json').write_text(encoded, encoding='utf-8')
    print(json.dumps(verification, indent=2, sort_keys=True))
    print(f'Frozen training index: {args.output.resolve()}')
    return metadata


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def build_eigen_manifest(args) -> dict:
    records, seen = [], set()
    source_rows = 0
    excluded_without_improved_gt = 0
    kitti_root = args.kitti_root.resolve()
    for line_number, raw_line in enumerate(
        args.eigen_split.read_text(encoding='utf-8').splitlines(), start=1
    ):
        columns = raw_line.strip().split()
        if not columns or columns[0].startswith('#'):
            continue
        if len(columns) < 2:
            raise ValueError(f'eigen split line {line_number}: fewer than two columns')
        source_rows += 1
        if columns[1] == 'None':
            excluded_without_improved_gt += 1
            continue
        rgb_rel, gt_rel = Path(columns[0]), Path(columns[1])
        if len(rgb_rel.parts) != 5 or rgb_rel.parts[2:4] != ('image_02', 'data'):
            raise ValueError(f'eigen split line {line_number}: malformed RGB path')
        date, drive, frame = rgb_rel.parts[0], rgb_rel.parts[1], rgb_rel.stem
        sample_id = f'{date}/{drive}/{frame}'
        if sample_id in seen:
            raise ValueError(f'duplicate Eigen sample: {sample_id}')
        seen.add(sample_id)
        rgb_path = (kitti_root / 'raw_data' / rgb_rel).resolve()
        depth_path = _first_existing([
            kitti_root / 'val' / gt_rel,
            kitti_root / 'train' / date / gt_rel,
            kitti_root / 'train' / gt_rel,
        ])
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        if depth_path is None:
            raise FileNotFoundError(f'no improved Eigen GT for {gt_rel}')
        records.append({
            'sample_id': sample_id,
            'split_line': line_number,
            'image_relative_path': str(rgb_rel),
            'depth_relative_path': str(gt_rel),
            'image_path': str(rgb_path),
            'depth_path': str(depth_path),
            'image_sha256': sha256_file(rgb_path),
            'depth_sha256': sha256_file(depth_path),
        })
    if len(records) != EXPECTED_EIGEN_SAMPLES:
        raise ValueError(f'expected 652 Eigen samples, got {len(records)}')
    if source_rows != 697 or excluded_without_improved_gt != 45:
        raise ValueError(
            f'expected 697 source rows and 45 without improved GT, got '
            f'{source_rows} and {excluded_without_improved_gt}'
        )
    document = {
        'version': 'kitti_eigen_652_zipdepth_eval_v1',
        'sample_count': len(records),
        'source_row_count': source_rows,
        'excluded_without_improved_gt': excluded_without_improved_gt,
        'source_split': str(args.eigen_split.resolve()),
        'source_split_sha256': sha256_file(args.eigen_split),
        'protocol': {
            'ground_truth': 'KITTI improved depth PNG divided by 256',
            'depth_range_m': [0.001, 80.0],
            'benchmark_crop': [352, 1216],
            'evaluation_mask': 'eigen',
            'prediction_alignment': 'least_square_disparity_scale_shift',
            'alignment_max_resolution': None,
        },
        'records': records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(f'Frozen Eigen manifest: {args.output.resolve()} ({len(records)} samples)')
    print(f'Manifest SHA256: {sha256_file(args.output)}')
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    train = subparsers.add_parser('train-index')
    train.add_argument('--resolved-train-manifest', type=Path, required=True)
    train.add_argument('--cache-manifest', type=Path, required=True)
    train.add_argument('--cache-root', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--verify-rgb-hashes', action='store_true')
    train.add_argument('--verify-payloads', action='store_true')

    eigen = subparsers.add_parser('eigen-manifest')
    eigen.add_argument('--kitti-root', type=Path, required=True)
    eigen.add_argument('--eigen-split', type=Path, required=True)
    eigen.add_argument('--output', type=Path, required=True)

    args = parser.parse_args()
    if args.command == 'train-index':
        build_train_index(args)
    else:
        build_eigen_manifest(args)


if __name__ == '__main__':
    main()
