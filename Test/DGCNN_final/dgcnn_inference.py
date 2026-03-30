from __future__ import annotations

import hashlib
import json
import os
import random
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import trimesh  # type: ignore
except ImportError:  # pragma: no cover - trimesh is optional
    trimesh = None

from model import DGCNN_cls
from dgcnn_model_loader import initialize_model, create_scheduler
from plyfile import PlyData

_DGCNN_MODEL_DEFAULTS = {'k': 20, 'emb_dims': 1024, 'dropout': 0.5}
_DGCNN_CONFIG_CANDIDATES = ('fold_config.json', 'config.json', 'model_config.json')


class IndependentMeshDataset(Dataset):
    """Loads independent mesh or point cloud files on demand."""

    _SUPPORTED_EXTENSIONS = {'.ply', '.stl', '.obj', '.npy', '.off'}

    def __init__(self, input_folder: str, npoints: int = 1024, normalize: bool = True, class_names=None, label_dict=None, seed: int = 42):
        self.input_folder = input_folder
        self.npoints = int(npoints)
        self.normalize = bool(normalize)
        self.class_names = class_names or {}
        self.label_dict = label_dict or {}
        self.seed = seed if seed is not None else 42

        files = sorted(
            file
            for file in os.listdir(input_folder)
            if any(file.lower().endswith(ext) for ext in self._SUPPORTED_EXTENSIONS)
        )
        if not files:
            raise ValueError(f'No supported files found in {input_folder}')
        self.input_files = files
        self.file_ids = [os.path.splitext(name)[0] for name in files]

    def __len__(self):
        return len(self.input_files)

    @staticmethod
    def _pc_normalize(points: np.ndarray):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        radius = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if radius > 0:
            points = points / radius
        return points

    @staticmethod
    def _fps(points: np.ndarray, npoint: int, rng: np.random.Generator | None = None):
        xyz = points[:, :3]
        centroids = np.zeros((npoint,), dtype=np.int32)
        distances = np.ones((points.shape[0],), dtype=np.float32) * 1e10
        farthest = int((rng or np.random.default_rng()).integers(0, points.shape[0]))
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, axis=1)
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest = int(np.argmax(distances))
        return points[centroids]

    def _load_points(self, file_path: str):
        ext = os.path.splitext(file_path.lower())[1]
        if ext == '.ply':
            plydata = PlyData.read(file_path)
            pts = np.vstack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')]).T
            return pts.astype(np.float32)
        if ext == '.npy':
            pts = np.load(file_path).astype(np.float32)
            if pts.ndim != 2 or pts.shape[1] < 3:
                raise ValueError(f'Unexpected npy shape {pts.shape} for {file_path}')
            return pts[:, :3]
        if ext in {'.stl', '.obj', '.off'}:
            import trimesh

            mesh = trimesh.load(file_path)
            if hasattr(mesh, 'sample'):
                pts = mesh.sample(max(self.npoints * 2, self.npoints))
            else:
                pts = mesh.vertices
            return np.asarray(pts, dtype=np.float32)
        raise ValueError(f'Unsupported extension {ext}')

    def _per_file_rng(self, file_id: str) -> np.random.Generator:
        """Stable per-file RNG derived from base seed + file identity hash."""
        h = int(hashlib.sha256(file_id.encode()).hexdigest(), 16) & 0xFFFFFFFF
        return np.random.default_rng(self.seed ^ h)

    def _prepare_points_for_inference(self, points: np.ndarray, file_name: str, rng: np.random.Generator | None = None) -> np.ndarray:
        file_ext = os.path.splitext(file_name.lower())[1]

        # Canonical NPY inputs are already normalized and sampled; use them as-is.
        if file_ext == '.npy' and points.shape[0] == self.npoints:
            return points[:, :3].astype(np.float32, copy=False)

        rng = rng or np.random.default_rng(self.seed)
        if points.shape[0] >= self.npoints:
            points = self._fps(points, self.npoints, rng=rng)
        else:
            deficit = self.npoints - points.shape[0]
            extra = points[rng.choice(points.shape[0], deficit, replace=True)]
            points = np.vstack([points, extra])
        if self.normalize:
            points = self._pc_normalize(points)
        return points[:, :3].astype(np.float32, copy=False)

    def __getitem__(self, idx: int):
        file_name = self.input_files[idx]
        file_id = self.file_ids[idx]
        file_path = os.path.join(self.input_folder, file_name)

        points = self._load_points(file_path)
        rng = self._per_file_rng(file_id)
        points = self._prepare_points_for_inference(points, file_name, rng=rng)

        tensor = torch.from_numpy(points)

        label = None
        if self.label_dict and file_id in self.label_dict:
            class_name = self.label_dict[file_id]
            if self.class_names and class_name in self.class_names:
                label = self.class_names[class_name]
        label_value = int(label) if label is not None else -1
        return tensor, torch.tensor(label_value, dtype=torch.long), file_id


def _prepare_class_maps(class_names: Iterable, num_classes: int):
    if isinstance(class_names, dict):
        name_to_idx = dict(class_names)
        idx_to_name = {idx: name for name, idx in class_names.items()}
        ordered = [idx_to_name.get(i, f'class_{i}') for i in range(num_classes)]
    else:
        ordered = list(class_names)
        name_to_idx = {name: idx for idx, name in enumerate(ordered)}
        idx_to_name = {idx: name for idx, name in enumerate(ordered)}
        if len(ordered) < num_classes:
            ordered.extend([f'class_{i}' for i in range(len(ordered), num_classes)])
    return name_to_idx, idx_to_name, ordered


def _resolve_model_kwargs(fold_dir: Optional[str], base_kwargs: Dict, config_filenames: Sequence[str]):
    resolved = dict(base_kwargs)
    if not fold_dir:
        return resolved
    for filename in config_filenames:
        candidate = os.path.join(fold_dir, filename)
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                for key in ('model_kwargs', 'model_config', 'model_params', 'hyperparameters'):
                    if key in payload and isinstance(payload[key], dict):
                        resolved.update(payload[key])
                for key in ('k', 'emb_dims', 'dropout'):
                    if key in payload:
                        resolved[key] = payload[key]
        except Exception as exc:  # pragma: no cover
            print(f'Warning: could not parse {candidate}: {exc}')
        break
    return resolved


def _build_model(num_classes: int, device, resolved_kwargs: Dict):
    """Wrapper to create model and return (model, actual_kwargs) using the shared loader."""
    k = int(resolved_kwargs.get('k', _DGCNN_MODEL_DEFAULTS['k']))
    emb_dims = int(resolved_kwargs.get('emb_dims', _DGCNN_MODEL_DEFAULTS['emb_dims']))
    dropout = float(resolved_kwargs.get('dropout', _DGCNN_MODEL_DEFAULTS['dropout']))

    model, criterion, optimizer, scheduler, actual_kwargs = initialize_model(
        num_classes=num_classes,
        device=device,
        k=k,
        emb_dims=emb_dims,
        dropout=dropout,
    )
    model.eval()
    return model, actual_kwargs


def _cache_point_clouds_for_non_npy_inputs(input_folder: str, output_dir: str, npoints: int = 1024, seed: int = 42):
    """Cache normalized point clouds only when the source folder is not already canonical NPY input."""
    supported_extensions = {'.ply', '.stl', '.obj', '.npy', '.off'}
    input_files = sorted(
        f for f in os.listdir(input_folder) if os.path.splitext(f.lower())[1] in supported_extensions
    )
    if not input_files:
        print(f'⚠️  No supported files found in {input_folder}')
        return

    if all(file_name.lower().endswith('.npy') for file_name in input_files):
        print('\n📦 Using source NPY files directly for critical analysis; no processed_npy cache created.')
        return

    dataset = IndependentMeshDataset(
        input_folder=input_folder,
        npoints=npoints,
        normalize=True,
        seed=seed,
    )
    processed_npy_dir = os.path.join(output_dir, 'processed_npy')
    os.makedirs(processed_npy_dir, exist_ok=True)

    print('\n📦 Caching normalized point clouds for non-NPY inputs')
    print(f'   Output: {processed_npy_dir}')
    saved_count = 0

    for idx, file_name in enumerate(tqdm(dataset.input_files, desc='Caching point clouds')):
        if file_name.lower().endswith('.npy'):
            continue
        try:
            file_path = os.path.join(input_folder, file_name)
            points = dataset._load_points(file_path)
            normalized_points = dataset._prepare_points_for_inference(points, file_name)
            flake_id = dataset.file_ids[idx]
            output_npy_path = os.path.join(processed_npy_dir, f'{flake_id}_normalized_npy.npy')
            np.save(output_npy_path, normalized_points.astype(np.float32))
            saved_count += 1
        except Exception as exc:
            print(f'  ✗ Error processing {file_name}: {exc}')

    print(f'✅ Saved {saved_count} cached point clouds')


def run_dgcnn_inference_on_mesh_folder(
    input_folder: str,
    kfold_results_dir: str,
    k_folds: int,
    num_classes: int,
    class_names,
    device,
    output_dir: str,
    *,
    npoints: int = 1024,
    normalize: bool = True,
    batch_size: int = 32,
    label_dict: Optional[Dict] = None,
    independent_true_class: Optional[str] = None,
    run_individual_models: bool = True,
    run_aggregation: bool = True,
    aggregate_methods: Sequence[str] = ('voting',),
    model_kwargs: Optional[Dict] = None,
    config_filenames: Sequence[str] = _DGCNN_CONFIG_CANDIDATES,
    dataset_cls=None,
    seed: int = 42,
    deterministic: bool = True,
):
    if dataset_cls is None:
        dataset_cls = IndependentMeshDataset

    if deterministic:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        print(f"Deterministic mode: ON (seed={seed})")

    print('🔬 RUNNING DGCNN INFERENCE ON INDEPENDENT FILES')
    print('=' * 70)
    print(f'Input folder: {input_folder}')
    print(f'K-fold models from: {kfold_results_dir}')
    print(f'Output directory: {output_dir}')
    print(f"Individual models: {'✅' if run_individual_models else '❌'}")
    print(f"Aggregation: {'✅' if run_aggregation else '❌'}")
    if run_aggregation:
        print(f'Aggregation methods: {aggregate_methods}')
    if independent_true_class is not None:
        print(f'Independent true class provided: {independent_true_class}')
    print('Supported formats: PLY, STL, NPY, OBJ, OFF')

    os.makedirs(output_dir, exist_ok=True)

    dataset = dataset_cls(
        input_folder=input_folder,
        npoints=npoints,
        normalize=normalize,
        class_names=class_names,
        label_dict=label_dict,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f'✅ Created dataset with {len(dataset)} files')
    print(f'   DataLoader batches: {len(dataloader)}')

    # Only cache mesh-derived point clouds; canonical NPY inputs are used directly.
    _cache_point_clouds_for_non_npy_inputs(
        input_folder=input_folder,
        output_dir=output_dir,
        npoints=npoints,
        seed=seed,
    )

    name_to_idx, idx_to_name, class_order = _prepare_class_maps(class_names, num_classes)
    base_model_kwargs = dict(_DGCNN_MODEL_DEFAULTS)
    if model_kwargs:
        base_model_kwargs.update(model_kwargs)
    config_filenames = tuple(config_filenames) if config_filenames else ()
    aggregate_methods = tuple(aggregate_methods) if aggregate_methods else ()

    all_model_predictions = {}
    individual_results = {}
    fold_model_configs = {}
    all_fold_preds = []

    for fold in range(1, k_folds + 1):
        print(f'\n📂 RUNNING INFERENCE WITH FOLD {fold} MODEL')
        print('-' * 40)
        fold_dir = os.path.join(kfold_results_dir, f'fold_{fold}')
        model_path = os.path.join(fold_dir, 'best_model_balanced.pth')
        if not os.path.exists(model_path):
            print(f'❌ Model not found: {model_path}')
            continue

        resolved_kwargs = _resolve_model_kwargs(fold_dir, base_model_kwargs, config_filenames)
        model, actual_kwargs = _build_model(num_classes, device, resolved_kwargs)
        fold_model_configs[f'fold_{fold}'] = actual_kwargs

        print(f'🔄 Loading model: {model_path}')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        fold_predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Inference Fold {fold}'):
                points, labels, file_ids = batch
                labels_cpu = labels.cpu()

                points = points.to(device)
                logits = model(points.permute(0, 2, 1))
                probs_cpu = F.softmax(logits, dim=1).detach().cpu()
                pred_indices = probs_cpu.argmax(dim=1)

                for i in range(probs_cpu.size(0)):
                    file_id = file_ids[i]
                    if isinstance(file_id, (list, tuple)):
                        file_id = file_id[0]
                    file_id = str(file_id)
                    pred_idx = int(pred_indices[i].item())
                    prob_vector = probs_cpu[i]
                    confidence = float(prob_vector[pred_idx].item()) if 0 <= pred_idx < prob_vector.numel() else float('nan')

                    # Base result with consistent key names and types
                    result = {
                        'fold': int(fold),
                        'file_id': file_id,
                        'pred_label': int(pred_idx),
                        'pred_class': str(idx_to_name.get(pred_idx, f'class_{pred_idx}')),
                        'confidence': float(confidence),
                        'true_label': None,
                        'true_class': None,
                        'correct': None,
                    }

                    # Fill per-class probabilities (guarantee all class_order columns exist)
                    for class_idx, class_name in enumerate(class_order):
                        key = f'{class_name.lower()}_probability'
                        if class_idx < prob_vector.numel():
                            result[key] = float(prob_vector[class_idx].item())
                        else:
                            result[key] = float('nan')

                    # If dataset provided true labels, use them (preferred)
                    true_val = int(labels_cpu[i].item())
                    if true_val >= 0:
                        result['true_label'] = int(true_val)
                        result['true_class'] = str(idx_to_name.get(true_val, f'class_{true_val}'))
                        result['correct'] = (true_val == pred_idx)
                    elif independent_true_class is not None:
                        if isinstance(independent_true_class, int):
                            tlabel = int(independent_true_class)
                            tclass = idx_to_name.get(tlabel, str(tlabel))
                        else:
                            tclass = str(independent_true_class)
                            tlabel = name_to_idx.get(independent_true_class, -1)
                        result['true_label'] = int(tlabel) if tlabel is not None else -1
                        result['true_class'] = str(tclass)
                        # compare on integer labels when possible
                        try:
                            result['correct'] = (int(result['pred_label']) == int(result['true_label']))
                        except Exception:
                            result['correct'] = (result['pred_class'] == result['true_class'])

                    fold_predictions.append(result)

        all_model_predictions[f'fold_{fold}'] = fold_predictions
        all_fold_preds.extend(fold_predictions)

        if run_individual_models and fold_predictions:
            print(f'\n📊 ANALYZING FOLD {fold} INDIVIDUAL PERFORMANCE')
            print('-' * 40)
            fold_df = pd.DataFrame(fold_predictions)
            individual_results[f'fold_{fold}'] = fold_df

            if 'correct' in fold_df.columns and not fold_df.empty:
                accuracy = fold_df['correct'].mean() * 100
                print(f'  Fold {fold} Accuracy: {accuracy:.2f}%')

                print('  Per-class Performance:')
                for class_name in class_order:
                    class_samples = fold_df[fold_df['true_class'] == class_name]
                    if not class_samples.empty:
                        class_acc = class_samples['correct'].mean() * 100
                        print(f'    {class_name}: {class_acc:.2f}% ({len(class_samples)} samples)')

            if not fold_df.empty:
                print('  Confidence Statistics:')
                print(f"    Mean: {fold_df['confidence'].mean():.3f}")
                print(f"    Std: {fold_df['confidence'].std():.3f}")
                print(f"    Min: {fold_df['confidence'].min():.3f}")
                print(f"    Max: {fold_df['confidence'].max():.3f}")

                print('  Prediction Distribution:')
                pred_counts = fold_df['pred_class'].value_counts()
                for class_name, count in pred_counts.items():
                    percentage = (count / len(fold_df)) * 100
                    print(f'    {class_name}: {count} ({percentage:.1f}%)')

                individual_output_dir = os.path.join(output_dir, 'individual_models')
                os.makedirs(individual_output_dir, exist_ok=True)
                individual_path = os.path.join(individual_output_dir, f'fold_{fold}_predictions.csv')
                fold_df.to_csv(individual_path, index=False)
                print(f'  💾 Saved: {individual_path}')

        print(f'✅ Completed inference for Fold {fold}: {len(fold_predictions)} predictions')

    if run_individual_models and len(individual_results) > 1:
        print('\n🔍 COMPARING INDIVIDUAL MODEL PERFORMANCE')
        print('=' * 50)
        comparison_data = []
        for fold_name, fold_df in individual_results.items():
            if fold_df.empty or 'correct' not in fold_df.columns:
                continue
            fold_num = int(fold_name.split('_')[1])
            accuracy = fold_df['correct'].mean()
            per_class_acc = {}
            for class_name in class_order:
                class_samples = fold_df[fold_df['true_class'] == class_name]
                per_class_acc[f'{class_name}_accuracy'] = class_samples['correct'].mean() if not class_samples.empty else 0.0
            comparison_row = {
                'fold': fold_num,
                'accuracy': accuracy,
                'mean_confidence': fold_df['confidence'].mean(),
                'confidence_std': fold_df['confidence'].std(),
                **per_class_acc,
            }
            comparison_data.append(comparison_row)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data).sort_values('fold')
            print('Individual Model Performance Summary:')
            print(comparison_df.round(4))
            if 'accuracy' in comparison_df.columns:
                best_fold = comparison_df.loc[comparison_df['accuracy'].idxmax()]
                worst_fold = comparison_df.loc[comparison_df['accuracy'].idxmin()]
                print(f"\n🏆 Best Model: Fold {int(best_fold['fold'])} (Accuracy: {best_fold['accuracy']:.4f})")
                print(f"🔻 Worst Model: Fold {int(worst_fold['fold'])} (Accuracy: {worst_fold['accuracy']:.4f})")
                print(f"📊 Performance Range: {comparison_df['accuracy'].max() - comparison_df['accuracy'].min():.4f}")
                print(f"📈 Mean Accuracy: {comparison_df['accuracy'].mean():.4f} ± {comparison_df['accuracy'].std():.4f}")
            comparison_path = os.path.join(output_dir, 'model_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False)
            print(f'💾 Model comparison saved: {comparison_path}')

    aggregated_results = {}
    if run_aggregation and len(all_model_predictions) > 1:
        print('\n🎯 RUNNING MODEL AGGREGATION')
        print('=' * 50)
        all_file_ids = set()
        for fold_preds in all_model_predictions.values():
            for pred in fold_preds:
                all_file_ids.add(pred['file_id'])
        all_file_ids = sorted(all_file_ids)
        print(f'Aggregating predictions for {len(all_file_ids)} files')

        for method in aggregate_methods:
            final_predictions = []
            for file_id in tqdm(all_file_ids, desc=f'Aggregating ({method})'):
                file_preds = []
                for fold in range(1, k_folds + 1):
                    fold_key = f'fold_{fold}'
                    if fold_key in all_model_predictions:
                        for pred in all_model_predictions[fold_key]:
                            if pred['file_id'] == file_id:
                                file_preds.append(pred)
                                break
                if not file_preds:
                    continue

                result = {'file_id': file_id, 'models_considered': len(file_preds)}
                # compute per-file fold confidence summary
                confs = [float(p.get('confidence', float('nan'))) for p in file_preds]
                mean_conf = float(np.nanmean(confs)) if confs else float('nan')
                std_conf = float(np.nanstd(confs, ddof=0)) if confs else float('nan')
                result['mean_fold_confidence'] = mean_conf
                result['fold_confidence_std'] = std_conf
                # Always include true_label and true_class (even if None for unlabeled sets)
                if 'true_label' in file_preds[0]:
                    result['true_label'] = file_preds[0]['true_label']
                    result['true_class'] = file_preds[0]['true_class']
                else:
                    result['true_label'] = None
                    result['true_class'] = None

                if method == 'voting':
                    from collections import Counter

                    pred_classes = [p['pred_class'] for p in file_preds]
                    vote_counts = Counter(pred_classes)
                    final_pred_class = vote_counts.most_common(1)[0][0]
                    final_pred_label = name_to_idx.get(final_pred_class, -1)
                    winning_confidences = [p['confidence'] for p in file_preds if p['pred_class'] == final_pred_class]
                    final_confidence = float(np.mean(winning_confidences)) if winning_confidences else float('nan')
                    result.update({
                        'pred_class': final_pred_class,
                        'pred_label': final_pred_label,
                        'confidence': final_confidence,
                        'vote_agreement': len(winning_confidences) / len(file_preds) if file_preds else float('nan'),
                    })
                elif method == 'averaging':
                    avg_probs = np.zeros(num_classes, dtype=np.float32)
                    for p in file_preds:
                        for class_idx, class_name in enumerate(class_order):
                            prob_key = f'{class_name.lower()}_probability'
                            if prob_key in p:
                                avg_probs[class_idx] += p[prob_key]
                    avg_probs /= len(file_preds)
                    final_pred_label = int(np.argmax(avg_probs))
                    final_pred_class = idx_to_name.get(final_pred_label, f'class_{final_pred_label}')
                    final_confidence = float(avg_probs[final_pred_label])
                    result.update({
                        'pred_class': final_pred_class,
                        'pred_label': final_pred_label,
                        'confidence': final_confidence,
                    })
                    for class_idx, class_name in enumerate(class_order):
                        result[f'{class_name.lower()}_probability'] = float(avg_probs[class_idx])
                elif method == 'best_model':
                    best_pred = max(file_preds, key=lambda x: x['confidence'])
                    result.update({
                        'pred_class': best_pred['pred_class'],
                        'pred_label': best_pred['pred_label'],
                        'confidence': best_pred['confidence'],
                        'best_fold': best_pred['fold'],
                    })
                    for class_idx, class_name in enumerate(class_order):
                        prob_key = f'{class_name.lower()}_probability'
                        if prob_key in best_pred:
                            result[prob_key] = best_pred[prob_key]
                else:
                    continue

                # Prefer comparing integer labels for correctness; fall back to class-name comparison
                if 'true_label' in result and result.get('true_label') is not None:
                    try:
                        result['correct'] = (int(result.get('true_label', -1)) == int(result.get('pred_label', -1)))
                    except Exception:
                        result['correct'] = (str(result.get('true_class', '')) == str(result.get('pred_class', '')))
                elif 'true_class' in result:
                    result['correct'] = (str(result.get('true_class', '')) == str(result.get('pred_class', '')))

                for fold_pred in file_preds:
                    fold_num = fold_pred['fold']
                    result[f'fold_{fold_num}_pred'] = fold_pred['pred_class']
                    result[f'fold_{fold_num}_conf'] = fold_pred['confidence']

                final_predictions.append(result)

            if final_predictions:
                aggregated_df = pd.DataFrame(final_predictions)
                aggregated_results[method] = aggregated_df

                # Also write a standardized ensemble CSV at the top-level output_dir
                try:
                    # Ensure probability columns exist and are ordered according to class_order
                    prob_cols = [f'{cn.lower()}_probability' for cn in class_order]
                    # Guarantee these columns exist in the DataFrame
                    for pc in prob_cols:
                        if pc not in aggregated_df.columns:
                            aggregated_df[pc] = np.nan

                    # Preferred column order for ensemble file (with true_label/true_class always included)
                    ensemble_cols = [
                        'file_id',
                        'models_considered',
                        'true_class',
                        'true_label',
                        'correct',
                        'pred_class',
                        'pred_label',
                        'confidence',
                        'mean_fold_confidence',
                        'fold_confidence_std',
                    ] + prob_cols

                    # Coerce types: file_id str, models_considered int, true_class str, true_label int, pred_label int
                    if 'file_id' in aggregated_df.columns:
                        aggregated_df['file_id'] = aggregated_df['file_id'].astype(str)
                    if 'models_considered' in aggregated_df.columns:
                        aggregated_df['models_considered'] = aggregated_df['models_considered'].astype('int64')
                    # true_label can be int or None; use pd.NA for None values
                    if 'true_label' in aggregated_df.columns:
                        aggregated_df['true_label'] = pd.to_numeric(aggregated_df['true_label'], errors='coerce')
                    # Ensure true_class is there (keep None values as they are)
                    if 'true_class' not in aggregated_df.columns:
                        aggregated_df['true_class'] = None
                    if 'pred_label' in aggregated_df.columns:
                        aggregated_df['pred_label'] = pd.to_numeric(aggregated_df['pred_label'], errors='coerce').fillna(-1).astype('int64')
                    if 'pred_class' in aggregated_df.columns:
                        aggregated_df['pred_class'] = aggregated_df['pred_class'].astype(str)

                    # Fill any missing probability columns with NaN and ensure floats
                    for pc in prob_cols:
                        aggregated_df[pc] = pd.to_numeric(aggregated_df.get(pc, np.nan), errors='coerce').astype('float64')

                    # Final ensemble DataFrame with requested columns (add missing as NAs)
                    ensemble_df = aggregated_df.reindex(columns=ensemble_cols)
                    ensemble_path = os.path.join(output_dir, 'ensemble_predictions.csv')
                    ensemble_df.to_csv(ensemble_path, index=False)
                except Exception as exc:  # pragma: no cover
                    print('Warning: could not write standardized ensemble file:', exc)

    print('\n🎉 INFERENCE COMPLETED!')
    print('=' * 60)
    results_summary = {
        'total_files': len(dataset),
        'models_used': len(all_model_predictions),
        'individual_results': individual_results if run_individual_models else None,
        'aggregated_results': aggregated_results if run_aggregation else None,
        'output_directory': output_dir,
        'model_configs': fold_model_configs,
    }
    print('📊 Summary:')
    print(f'  Total files processed: {len(dataset)}')
    print(f'  Models used: {len(all_model_predictions)}')
    if run_individual_models:
        print('  Individual model results: ✅ Saved in individual_models/')
    if run_aggregation:
        print('  Aggregated results: ✅ Saved in aggregated_results/')
        print(f'  Aggregation methods: {aggregate_methods}')

    if all_fold_preds:
        fold_results_df = pd.DataFrame(all_fold_preds)
        # standardize detailed predictions columns and dtypes to match PointNet++ / previous DGCNN
        prob_cols = [f'{cn.lower()}_probability' for cn in class_order]
        for pc in prob_cols:
            if pc not in fold_results_df.columns:
                fold_results_df[pc] = np.nan

        # Always include all columns, even for unlabeled sets (they'll have None/NaN values)
        desired_cols = ['fold', 'file_id', 'pred_label', 'pred_class', 'confidence'] + prob_cols + ['true_class', 'true_label', 'correct']

        # Coerce types
        if 'fold' in fold_results_df.columns:
            fold_results_df['fold'] = pd.to_numeric(fold_results_df['fold'], errors='coerce').astype('Int64')
        if 'file_id' in fold_results_df.columns:
            fold_results_df['file_id'] = fold_results_df['file_id'].astype(str)
        if 'pred_label' in fold_results_df.columns:
            fold_results_df['pred_label'] = pd.to_numeric(fold_results_df['pred_label'], errors='coerce').fillna(-1).astype('int64')
        if 'pred_class' in fold_results_df.columns:
            fold_results_df['pred_class'] = fold_results_df['pred_class'].astype(str)
        if 'confidence' in fold_results_df.columns:
            fold_results_df['confidence'] = pd.to_numeric(fold_results_df['confidence'], errors='coerce').astype('float64')
        for pc in prob_cols:
            fold_results_df[pc] = pd.to_numeric(fold_results_df.get(pc, np.nan), errors='coerce').astype('float64')
        if 'true_label' in fold_results_df.columns:
            fold_results_df['true_label'] = pd.to_numeric(fold_results_df['true_label'], errors='coerce')
        if 'true_class' in fold_results_df.columns:
            fold_results_df['true_class'] = fold_results_df['true_class'].astype(str)
        if 'correct' in fold_results_df.columns:
            fold_results_df['correct'] = fold_results_df['correct'].astype('bool', errors='ignore')

        detailed_path = os.path.join(output_dir, 'all_model_predictions_detailed.csv')
        # Reindex to desired order, adding any missing cols as NA
        out_df = fold_results_df.reindex(columns=desired_cols)
        out_df.to_csv(detailed_path, index=False)

    config = {
        'input_folder': input_folder,
        'kfold_results_dir': kfold_results_dir,
        'k_folds': k_folds,
        'num_classes': num_classes,
        'class_names': class_names,
        'npoints': npoints,
        'normalize': normalize,
        'batch_size': batch_size,
        'run_individual_models': run_individual_models,
        'run_aggregation': run_aggregation,
        'aggregate_methods': list(aggregate_methods),
        'total_samples': len(dataset),
        'model_defaults': base_model_kwargs,
        'model_configs': fold_model_configs,
    }
    config_path = os.path.join(output_dir, 'dgcnn_inference_config.json')
    with open(config_path, 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)

    return results_summary


def load_dgcnn_model(model_path, num_classes, k=20, emb_dims=1024, dropout=0.5):
    """Load a trained DGCNN model from checkpoint."""
    from types import SimpleNamespace
    args = SimpleNamespace(k=k, emb_dims=emb_dims, dropout=dropout)
    model = DGCNN_cls(args, output_channels=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    return model
