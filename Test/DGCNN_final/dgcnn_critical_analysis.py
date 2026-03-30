from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dgcnn_inference import load_dgcnn_model


def set_critical_random_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_npy_files(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith('.npy')])


def canonicalize_flake_id(file_name_or_id: str) -> str:
    flake_id = os.path.basename(file_name_or_id)
    if flake_id.lower().endswith('.npy'):
        flake_id = flake_id[:-4]
    suffix = '_normalized_npy'
    while flake_id.endswith(suffix):
        flake_id = flake_id[: -len(suffix)]
    return flake_id


def list_source_npy_by_flake_id(source_npy_dir: str) -> Dict[str, str]:
    """Map canonical flake_id -> source npy filename."""
    mapping: Dict[str, str] = {}
    for npy_file in sorted([f for f in os.listdir(source_npy_dir) if f.lower().endswith('.npy')]):
        flake_id = canonicalize_flake_id(npy_file)
        if flake_id not in mapping:
            mapping[flake_id] = npy_file
    return mapping


def normalize_existing_critical_filenames(critical_points_dir: str) -> None:
    """Rename legacy critical files to canonical <flake_id>_critical.npy format when safe."""
    if not os.path.isdir(critical_points_dir):
        return

    for file_name in sorted(os.listdir(critical_points_dir)):
        if not file_name.endswith('_critical.npy'):
            continue

        raw_flake_id = file_name[: -len('_critical.npy')]
        canonical_flake_id = canonicalize_flake_id(raw_flake_id)
        canonical_name = f'{canonical_flake_id}_critical.npy'

        if canonical_name == file_name:
            continue

        src = os.path.join(critical_points_dir, file_name)
        dst = os.path.join(critical_points_dir, canonical_name)
        if os.path.exists(dst):
            continue
        os.rename(src, dst)


def resolve_npy_source_dir(batch_root: str, tests_root: str, test_name: str) -> Tuple[Optional[str], Optional[str], int]:
    candidates = [
        (os.path.join(tests_root, test_name), 'dataset_root_npy'),
        (os.path.join(batch_root, test_name, 'processed_npy'), 'batch_processed_npy'),
    ]
    for folder, source_type in candidates:
        n_files = count_npy_files(folder)
        if n_files > 0:
            return folder, source_type, n_files
    return None, None, 0


def check_critical_point_status(
    batch_root: str,
    tests_root: str,
    test_names: Iterable[str],
    *,
    skip_existing_critical: bool = True,
) -> pd.DataFrame:
    status_rows: List[Dict] = []
    for test_name in sorted(test_names):
        source_dir, source_type, n_files = resolve_npy_source_dir(batch_root, tests_root, test_name)
        critical_dir = os.path.join(batch_root, test_name, 'critical_points')
        existing_critical = (
            len([f for f in os.listdir(critical_dir) if f.endswith('_critical.npy')])
            if os.path.isdir(critical_dir)
            else 0
        )
        pending = max(0, n_files - existing_critical) if skip_existing_critical else n_files
        status_rows.append(
            {
                'test_set': test_name,
                'source_type': source_type if source_type else 'missing',
                'source_dir': source_dir if source_dir else '',
                'npy_count': n_files,
                'existing_critical': existing_critical,
                'pending': pending,
                'ready': n_files > 0,
            }
        )
    return pd.DataFrame(status_rows)


def load_ensemble_models(
    kfold_results_dir: str,
    folds: Sequence[int],
    *,
    num_classes: int = 3,
    device: Optional[torch.device] = None,
):
    models = []
    available_folds: List[int] = []

    for fold in folds:
        checkpoint = os.path.join(kfold_results_dir, f'fold_{fold}', 'best_model_balanced.pth')
        if not os.path.exists(checkpoint):
            continue

        model = load_dgcnn_model(checkpoint, num_classes=num_classes, k=20)
        if device is not None:
            model = model.to(device)
        model.eval()
        models.append(model)
        available_folds.append(fold)

    return models, available_folds


def safe_load_points(npy_path: str) -> np.ndarray:
    pts = np.load(npy_path)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
        raise ValueError(f'Invalid point shape {pts.shape} in {os.path.basename(npy_path)}')
    return pts[:, :3]


def compute_ensemble_saliency(models, pts: np.ndarray, device: torch.device) -> np.ndarray:
    if not models:
        raise RuntimeError('No models provided for saliency computation')

    saliency_sum = np.zeros(pts.shape[0], dtype=np.float32)
    base_tensor = torch.from_numpy(pts).unsqueeze(0).to(device)

    for model in models:
        pts_tensor = base_tensor.detach().clone().requires_grad_(True)
        logits = model(pts_tensor.transpose(1, 2))
        pred_idx = logits.argmax(dim=1).item()
        score = logits[0, pred_idx]

        model.zero_grad(set_to_none=True)
        score.backward()

        grads = pts_tensor.grad[0]
        saliency_sum += grads.norm(dim=1).detach().cpu().numpy().astype(np.float32)

    return saliency_sum / float(len(models))


def compute_critical_points_for_tests(
    *,
    batch_output_root: str,
    tests_root_dir: str,
    test_names: Iterable[str],
    output_kfold_dir: str,
    saliency_folds: Sequence[int],
    critical_fraction: float = 0.10,
    skip_existing_critical: bool = True,
    num_classes: int = 3,
    random_seed: int = 42,
    device: Optional[torch.device] = None,
):
    set_critical_random_seed(random_seed)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading ensemble models...')
    models, available_folds = load_ensemble_models(
        output_kfold_dir,
        saliency_folds,
        num_classes=num_classes,
        device=device,
    )
    print(f'✓ Loaded {len(models)} models from folds: {available_folds}\n')

    if not models:
        raise RuntimeError('No ensemble models found.')

    test_results: Dict[str, Dict] = {}
    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for test_name in sorted(test_names):
        test_batch_dir = os.path.join(batch_output_root, test_name)
        source_npy_dir, source_type, _ = resolve_npy_source_dir(batch_output_root, tests_root_dir, test_name)

        if source_npy_dir is None:
            print(f'⏭️  {test_name}: No NPY files found in batch output or dataset root')
            continue

        npy_files = sorted([f for f in os.listdir(source_npy_dir) if f.lower().endswith('.npy')])
        if not npy_files:
            print(f'⏭️  {test_name}: No NPY files')
            continue

        critical_points_dir = os.path.join(test_batch_dir, 'critical_points')
        os.makedirs(critical_points_dir, exist_ok=True)
        normalize_existing_critical_filenames(critical_points_dir)

        source_npy_map = list_source_npy_by_flake_id(source_npy_dir)
        source_flake_ids = sorted(source_npy_map.keys())
        if not source_flake_ids:
            print(f'⏭️  {test_name}: No usable NPY files')
            continue

        if skip_existing_critical:
            existing_outputs = {
                canonicalize_flake_id(f[:-len('_critical.npy')])
                for f in os.listdir(critical_points_dir)
                if f.endswith('_critical.npy')
            }
            pending_flake_ids = [flake_id for flake_id in source_flake_ids if flake_id not in existing_outputs]
        else:
            pending_flake_ids = source_flake_ids

        if not pending_flake_ids:
            print(f'⏭️  {test_name}: All {len(source_flake_ids)} flakes already have critical-point files')
            test_results[test_name] = {
                'source_npy_dir': source_npy_dir,
                'source_type': source_type,
                'n_files': len(source_flake_ids),
                'processed': 0,
                'skipped_existing': len(source_flake_ids),
                'errors': 0,
                'critical_points_dir': critical_points_dir,
            }
            total_skipped += len(source_flake_ids)
            continue

        print(f"\n{'=' * 60}")
        print(
            f'Processing: {test_name} | source={source_type} | total={len(source_flake_ids)} | pending={len(pending_flake_ids)}'
        )

        processed_count = 0
        error_count = 0

        for flake_id in tqdm(pending_flake_ids, desc=f'Computing saliency - {test_name}'):
            npy_file = source_npy_map[flake_id]
            npy_path = os.path.join(source_npy_dir, npy_file)
            output_path = os.path.join(critical_points_dir, f'{flake_id}_critical.npy')

            try:
                pts_normalized = safe_load_points(npy_path)
                avg_saliency = compute_ensemble_saliency(models, pts_normalized, device)

                min_sal, max_sal = np.min(avg_saliency), np.max(avg_saliency)
                if max_sal > min_sal:
                    norm_saliency = (avg_saliency - min_sal) / (max_sal - min_sal)
                else:
                    norm_saliency = np.ones_like(avg_saliency, dtype=np.float32) * 0.5

                num_critical = max(1, int(len(pts_normalized) * critical_fraction))
                top_idx = np.argpartition(avg_saliency, -num_critical)[-num_critical:]
                critical_data = np.hstack([pts_normalized[top_idx], norm_saliency[top_idx].reshape(-1, 1)])

                np.save(output_path, critical_data.astype(np.float32))
                processed_count += 1
            except Exception as exc:
                error_count += 1
                print(f'  Error {flake_id}: {exc}')

        skipped_existing = len(source_flake_ids) - len(pending_flake_ids)
        test_results[test_name] = {
            'source_npy_dir': source_npy_dir,
            'source_type': source_type,
            'n_files': len(source_flake_ids),
            'processed': processed_count,
            'skipped_existing': skipped_existing,
            'errors': error_count,
            'critical_points_dir': critical_points_dir,
        }

        total_processed += processed_count
        total_skipped += skipped_existing
        total_errors += error_count

    summary = {
        'total_processed': total_processed,
        'total_skipped': total_skipped,
        'total_errors': total_errors,
    }
    return test_results, summary, models, available_folds, device


def compute_rank_curves_for_test(
    test_name: str,
    source_npy_dir: str,
    models,
    percentiles: Sequence[int],
    device: torch.device,
):
    flake_curves = []
    source_npy_map = list_source_npy_by_flake_id(source_npy_dir)

    for flake_id in tqdm(sorted(source_npy_map.keys()), desc=f'Computing rank curves - {test_name}'):
        filename = source_npy_map[flake_id]
        pts = np.load(os.path.join(source_npy_dir, filename)).astype(np.float32)

        try:
            avg_importance = compute_ensemble_saliency(models, pts, device)
            min_imp, max_imp = np.min(avg_importance), np.max(avg_importance)
            norm_sal = (
                (avg_importance - min_imp) / (max_imp - min_imp)
                if max_imp > min_imp
                else np.ones_like(avg_importance) * 0.5
            )

            sorted_sal = np.sort(norm_sal)[::-1]
            total = np.sum(sorted_sal)
            cumulative = np.cumsum(sorted_sal) / total if total > 0 else np.linspace(0, 1, len(sorted_sal))

            curve = [
                float(cumulative[int((p / 100) * len(cumulative) - 1)])
                if int((p / 100) * len(cumulative) - 1) >= 0
                else 0.0
                for p in percentiles
            ]
            flake_curves.append({'flake_id': flake_id, 'curve': curve})
        except Exception as exc:
            print(f'  Error {flake_id}: {exc}')

    return flake_curves
