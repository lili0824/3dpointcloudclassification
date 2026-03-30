"""Standalone utilities for PointNet++ inference across independent datasets."""

from __future__ import annotations

import os
import random
import hashlib
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from plyfile import PlyData
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pointnet2_model_loader import initialize_pointnet2_model

try:
    import trimesh  # type: ignore
except ImportError:  # pragma: no cover - trimesh is optional
    trimesh = None

__all__ = [
    "IndependentMeshDataset",
    "run_inference_on_mesh_folder",
    "load_pointnet2_model",
]


class IndependentMeshDataset(Dataset):
    """Loads independent mesh or point cloud files on demand."""

    _SUPPORTED_EXTENSIONS = {".ply", ".stl", ".obj", ".npy", ".off"}

    def __init__(
        self,
        input_folder: str,
        npoints: int = 1024,
        normalize: bool = True,
        class_names: Iterable | None = None,
        label_dict: dict | None = None,
        seed: int | None = None,
    ) -> None:
        self.input_folder = input_folder
        self.npoints = npoints
        self.normalize = normalize
        self.class_names = dict(class_names or {})
        self.label_dict = {str(k): v for k, v in (label_dict or {}).items()}
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.input_files = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if self._is_supported(file):
                    self.input_files.append(os.path.join(root, file))
        self.input_files = sorted(self.input_files)
        if not self.input_files:
            raise ValueError(
                f"No supported files found in {input_folder}. Supported extensions: {sorted(self._SUPPORTED_EXTENSIONS)}"
            )

        self.file_ids = [os.path.splitext(os.path.basename(file))[0] for file in self.input_files]

    def __len__(self) -> int:
        return len(self.input_files)

    def _get_rng(self, file_name: str) -> np.random.Generator:
        base_seed = int(self.seed) if self.seed is not None else 0
        stable_hash = int(hashlib.md5(file_name.encode("utf-8")).hexdigest()[:8], 16)
        return np.random.default_rng(base_seed + stable_hash)

    def __getitem__(self, index: int):
        input_file = self.input_files[index]
        file_id = self.file_ids[index]
        file_path = input_file

        try:
            points = self._load_points(file_path)

            file_name = os.path.basename(file_path)
            rng = self._get_rng(file_name)
            points = self._prepare_points_for_inference(points, file_name, rng=rng)

            tensor = torch.from_numpy(points)

            label = None
            if file_id in self.label_dict and self.class_names:
                label_name = self.label_dict[file_id]
                if label_name in self.class_names:
                    label = self.class_names[label_name]

            if label is not None:
                return tensor, torch.tensor(label, dtype=torch.long), file_id
            return tensor, file_id
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Error loading {input_file}: {exc}")
            dummy = torch.zeros(self.npoints, 3, dtype=torch.float32)
            if self.label_dict:
                return dummy, torch.tensor(-1, dtype=torch.long), file_id
            return dummy, file_id

    def _is_supported(self, filename: str) -> bool:
        return os.path.splitext(filename.lower())[1] in self._SUPPORTED_EXTENSIONS

    @staticmethod
    def _pc_normalize(points: np.ndarray) -> np.ndarray:
        centroid = np.mean(points, axis=0)
        normalized = points - centroid
        scale = np.max(np.sqrt(np.sum(normalized**2, axis=1)))
        if scale > 0:
            normalized = normalized / scale
        return normalized

    @staticmethod
    def _farthest_point_sample(points: np.ndarray, npoint: int, rng: np.random.Generator) -> np.ndarray:
        N = points.shape[0]
        xyz = points[:, :3] if points.shape[1] >= 3 else points
        centroids = np.zeros((npoint,), dtype=np.int32)
        distance = np.ones((N,)) * 1e10
        farthest = int(rng.integers(0, N))

        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)

        return points[centroids]

    def _load_points(self, file_path: str) -> np.ndarray:
        ext = os.path.splitext(file_path.lower())[1]

        if ext == ".ply":
            plydata = PlyData.read(file_path)
            pts = np.vstack([plydata["vertex"][axis] for axis in ("x", "y", "z")]).T
            return pts.astype(np.float32)
        if ext == ".npy":
            pts = np.load(file_path).astype(np.float32)
            if pts.ndim != 2 or pts.shape[1] < 3:
                raise ValueError(f"Unexpected npy shape {pts.shape} for {file_path}")
            return pts[:, :3]
        if ext in {".stl", ".obj"}:
            if trimesh is None:  # pragma: no cover - optional dependency
                raise ImportError(
                    "trimesh is required to read STL/OBJ files. Install with `pip install trimesh`."
                )
            mesh = trimesh.load(file_path)
            if hasattr(mesh, "sample"):
                pts = mesh.sample(max(self.npoints * 2, self.npoints))
            else:
                pts = mesh.vertices
            return np.asarray(pts, dtype=np.float32)
        if ext == ".off":
            return self._load_off_file(file_path)

        raise ValueError(f"Unsupported extension {ext}")

    def _prepare_points_for_inference(
        self,
        points: np.ndarray,
        file_name: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        file_ext = os.path.splitext(file_name.lower())[1]

        if file_ext == ".npy" and points.shape[0] == self.npoints:
            return points[:, :3].astype(np.float32, copy=False)

        if points.shape[0] >= self.npoints:
            points = self._farthest_point_sample(points, self.npoints, rng=rng)
        else:
            deficit = self.npoints - points.shape[0]
            extra_idx = rng.choice(points.shape[0], deficit, replace=True)
            extra = points[extra_idx]
            points = np.vstack([points, extra])

        if self.normalize:
            points = self._pc_normalize(points)

        return points[:, :3].astype(np.float32, copy=False)


    def _load_off_file(self, file_path: str) -> np.ndarray:
        """Load OFF file and extract vertices.

        Supports simple OFF files where the header contains 'OFF' then counts.
        Ignores face data and reads just vertex coordinates.
        """
        try:
            verts = []
            with open(file_path, 'r') as f:
                # Read header, skip empty/comment lines
                header = None
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line == '' or line.startswith('#'):
                        continue
                    header = line
                    break
                if header is None:
                    raise ValueError(f"Empty or invalid OFF file: {file_path}")
                if header.upper().startswith('OFF'):
                    # Next non-empty line should contain counts
                    while True:
                        line = f.readline()
                        if not line:
                            raise ValueError(f"Unexpected end of OFF file: {file_path}")
                        line = line.strip()
                        if line == '' or line.startswith('#'):
                            continue
                        parts = line.split()
                        if len(parts) < 3:
                            continue
                        try:
                            n_verts = int(parts[0])
                            n_faces = int(parts[1])
                        except ValueError:
                            raise ValueError(f"Invalid OFF header counts in {file_path}: {line}")
                        break
                    # Read vertex lines
                    for _ in range(n_verts):
                        line = f.readline()
                        while line is not None and line.strip() == '':
                            line = f.readline()
                        if not line:
                            raise ValueError(f"Unexpected end of OFF file when reading vertices: {file_path}")
                        coords = list(map(float, line.strip().split()))
                        verts.append(coords[:3])
                else:
                    # Some OFFs may put counts on the same line as 'OFF' (e.g. 'OFF 8 12 0')
                    parts = header.split()
                    if len(parts) >= 4 and parts[0].upper().startswith('OFF'):
                        try:
                            n_verts = int(parts[1])
                            # skip reading faces count
                        except Exception:
                            raise ValueError(f"Cannot parse OFF header: {header}")
                        for _ in range(n_verts):
                            line = f.readline()
                            while line is not None and line.strip() == '':
                                line = f.readline()
                            if not line:
                                raise ValueError(f"Unexpected end of OFF file when reading vertices: {file_path}")
                            coords = list(map(float, line.strip().split()))
                            verts.append(coords[:3])
            points = np.array(verts, dtype=np.float32)
            return points
        except Exception as e:
            print(f"Error reading OFF file {file_path}: {e}")
            raise


def run_inference_on_mesh_folder(
    input_folder: str,
    kfold_results_dir: str,
    k_folds: int,
    num_classes: int,
    class_names: Sequence[str] | dict,
    device,
    output_dir: str,
    *,
    npoints: int = 1024,
    normalize: bool = True,
    batch_size: int = 32,
    label_dict: dict | None = None,
    seed: int = 42,
    deterministic: bool = True,
) -> dict:
    """Run inference across saved folds and emit combined + ensemble CSV outputs."""

    print("🔬 RUNNING POINTNET++ INFERENCE ON INDEPENDENT FILES")
    print("=" * 70)
    print(f"Input folder: {input_folder}")
    print(f"K-fold models from: {kfold_results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total folds expected: {k_folds}")
    print("Supported formats: PLY, STL, NPY, OBJ")

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

    os.makedirs(output_dir, exist_ok=True)

    dataset = IndependentMeshDataset(
        input_folder=input_folder,
        npoints=npoints,
        normalize=normalize,
        class_names=class_names,
        label_dict=label_dict,
        seed=seed,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"✅ Created dataset with {len(dataset)} files")
    print(f"   DataLoader batches: {len(dataloader)}")

    _cache_point_clouds_for_non_npy_inputs(
        input_folder=input_folder,
        output_dir=output_dir,
        npoints=npoints,
        normalize=normalize,
        seed=seed,
    )

    if isinstance(class_names, dict):
        ordered_classes = list(class_names.keys())
        name_to_idx = dict(class_names)
    else:
        ordered_classes = list(class_names)
        name_to_idx = {name: idx for idx, name in enumerate(ordered_classes)}

    idx_to_name = {idx: name for name, idx in name_to_idx.items()}
    lower_to_name = {name.lower(): name for name in ordered_classes}
    probability_columns = [f"{name.lower()}_probability" for name in ordered_classes]

    all_predictions = []
    missing_models = []

    for fold in range(1, k_folds + 1):
        print(f"\n📂 RUNNING INFERENCE WITH FOLD {fold} MODEL")
        fold_dir = os.path.join(kfold_results_dir, f"fold_{fold}")
        model_path = os.path.join(fold_dir, "best_model_balanced.pth")
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            missing_models.append(fold)
            continue

        model, _, _, _ = initialize_pointnet2_model(
            num_classes=num_classes,
            device=device,
            normal_channel=False,
            learning_rate=0.001,
        )

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Inference Fold {fold}"):
                if len(batch) == 3:
                    points, labels, file_ids = batch
                    labels_np = labels.cpu().numpy()
                    has_labels = True
                else:
                    points, file_ids = batch
                    labels_np = None
                    has_labels = False

                points = points.to(device)
                logits, _ = model(points.transpose(2, 1))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pred_indices = probs.argmax(axis=1)

                for idx_in_batch, file_id in enumerate(file_ids):
                    file_id = file_id if isinstance(file_id, str) else str(file_id)
                    pred_idx = int(pred_indices[idx_in_batch])
                    prob_vector = probs[idx_in_batch]
                    record = {
                        "fold": fold,
                        "file_id": file_id,
                        "pred_label": pred_idx,
                        "pred_class": idx_to_name.get(pred_idx, f"class_{pred_idx}"),
                        "confidence": float(prob_vector[pred_idx]),
                    }
                    for class_pos, class_name in enumerate(ordered_classes):
                        record[f"{class_name.lower()}_probability"] = float(prob_vector[class_pos])

                    if has_labels and labels_np is not None:
                        true_val = int(labels_np[idx_in_batch])
                        if true_val >= 0:
                            record["true_label"] = true_val
                            record["true_class"] = idx_to_name.get(true_val, f"class_{true_val}")
                            record["correct"] = true_val == pred_idx

                    all_predictions.append(record)

    if missing_models:
        print(f"⚠️ Missing models for folds: {missing_models}")

    if not all_predictions:
        raise RuntimeError("No predictions were generated. Check model paths and dataset inputs.")

    combined_df = pd.DataFrame(all_predictions)
    combined_df["file_id"] = combined_df["file_id"].astype(str)

    for col in probability_columns:
        if col not in combined_df.columns:
            combined_df[col] = np.nan

    if label_dict:
        lookup = {str(key): value for key, value in label_dict.items()}
        if "true_class" in combined_df.columns:
            combined_df["true_class"] = combined_df["true_class"].fillna(
                combined_df["file_id"].map(lookup)
            )
        else:
            combined_df["true_class"] = combined_df["file_id"].map(lookup)

    if "true_class" in combined_df.columns and "true_label" not in combined_df.columns:
        combined_df["true_label"] = combined_df["true_class"].map(name_to_idx)
    if "true_label" in combined_df.columns and "correct" not in combined_df.columns:
        combined_df["correct"] = combined_df["pred_label"] == combined_df["true_label"]

    combined_df.sort_values(["file_id", "fold"], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Always include true_class, true_label, correct columns (even for ModelNet - they'll be NaN)
    if "true_class" not in combined_df.columns:
        combined_df["true_class"] = None
    if "true_label" not in combined_df.columns:
        combined_df["true_label"] = None
    if "correct" not in combined_df.columns:
        combined_df["correct"] = None

    ordered_columns = [
        "fold",
        "file_id",
        "pred_label",
        "pred_class",
        "confidence",
        *probability_columns,
        "true_class",
        "true_label",
        "correct",
    ]
    combined_df = combined_df[ordered_columns]

    detailed_path = os.path.join(output_dir, "all_model_predictions_detailed.csv")
    combined_df.to_csv(detailed_path, index=False)
    print(f"✅ Combined predictions saved: {detailed_path}")

    probability_means = combined_df.groupby("file_id")[probability_columns].mean().reset_index()
    confidence_stats = (
        combined_df.groupby("file_id")["confidence"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_fold_confidence", "std": "fold_confidence_std"})
        .reset_index()
    )
    fold_counts = (
        combined_df.groupby("file_id")["fold"].nunique()
        .rename("models_considered")
        .reset_index()
    )

    ensemble_df = probability_means.merge(confidence_stats, on="file_id", how="left").merge(
        fold_counts, on="file_id", how="left"
    )

    truth_cols = [col for col in ("true_class", "true_label") if col in combined_df.columns]
    if truth_cols:
        truth_frame = combined_df[["file_id", *truth_cols]].drop_duplicates("file_id")
        ensemble_df = ensemble_df.merge(truth_frame, on="file_id", how="left")

    best_cols = ensemble_df[probability_columns].idxmax(axis=1)
    ensemble_df["pred_class"] = best_cols.str.replace("_probability", "", regex=False).map(lower_to_name)
    ensemble_df["pred_label"] = ensemble_df["pred_class"].map(name_to_idx)

    def _extract_confidence(row):
        pred_class = row.get("pred_class")
        if isinstance(pred_class, str):
            key = f"{pred_class.lower()}_probability"
            return row.get(key, np.nan)
        return np.nan

    ensemble_df["confidence"] = ensemble_df.apply(_extract_confidence, axis=1)

    if "true_class" in ensemble_df.columns:
        ensemble_df["correct"] = ensemble_df["pred_class"] == ensemble_df["true_class"]
    else:
        ensemble_df["correct"] = None

    # Always include true_class, true_label, correct columns (even for ModelNet - they'll be NaN)
    if "true_class" not in ensemble_df.columns:
        ensemble_df["true_class"] = None
    if "true_label" not in ensemble_df.columns:
        ensemble_df["true_label"] = None
    if "correct" not in ensemble_df.columns:
        ensemble_df["correct"] = None

    ensemble_df["models_considered"] = ensemble_df["models_considered"].fillna(0).astype(int)
    ensemble_df.sort_values("file_id", inplace=True)

    ordered_ensemble_cols = [
        "file_id",
        "models_considered",
        "true_class",
        "true_label",
        "correct",
        "pred_class",
        "pred_label",
        "confidence",
        "mean_fold_confidence",
        "fold_confidence_std",
        *probability_columns,
    ]
    ensemble_df = ensemble_df[ordered_ensemble_cols]

    ensemble_path = os.path.join(output_dir, "ensemble_predictions.csv")
    ensemble_df.to_csv(ensemble_path, index=False)
    print(f"✅ Ensemble predictions saved: {ensemble_path}")

    print("🎉 Inference complete!")

    return {
        "combined": combined_df,
        "ensemble": ensemble_df,
        "missing_folds": missing_models,
    }


def _cache_point_clouds_for_non_npy_inputs(
    input_folder: str,
    output_dir: str,
    npoints: int = 1024,
    normalize: bool = True,
    seed: int = 42,
) -> None:
    """Cache normalized sampled NPY only for non-NPY source folders."""
    np.random.seed(seed)

    supported_exts = {".ply", ".stl", ".obj", ".npy", ".off"}
    input_files = sorted(
        os.path.join(root, name)
        for root, _, files in os.walk(input_folder)
        for name in files
        if os.path.splitext(name.lower())[1] in supported_exts
    )
    if not input_files:
        print(f"⚠️  No supported input files found in {input_folder}")
        return

    if all(path.lower().endswith(".npy") for path in input_files):
        print("\n📦 Using source NPY files directly for critical analysis; no processed_npy cache created.")
        return

    dataset = IndependentMeshDataset(
        input_folder=input_folder,
        npoints=npoints,
        normalize=normalize,
        seed=seed,
    )

    processed_npy_dir = os.path.join(output_dir, "processed_npy")
    os.makedirs(processed_npy_dir, exist_ok=True)
    saved_count = 0

    for idx, file_path in enumerate(tqdm(dataset.input_files, desc="Caching processed_npy")):
        file_name = os.path.basename(file_path)
        if file_name.lower().endswith(".npy"):
            continue
        try:
            points = dataset._load_points(file_path)
            rng = dataset._get_rng(file_name)
            sampled_normalized = dataset._prepare_points_for_inference(points, file_name, rng=rng)
            flake_id = dataset.file_ids[idx]
            np.save(os.path.join(processed_npy_dir, f"{flake_id}.npy"), sampled_normalized.astype(np.float32))
            saved_count += 1
        except Exception as exc:
            print(f"Error preprocessing {file_name}: {exc}")

    print(f"✅ Saved cached processed point clouds: {saved_count}")
    print(f"   processed_npy: {processed_npy_dir}")


def load_pointnet2_model(model_path, num_classes, use_normals=False):
    """Load a trained PointNet++ checkpoint for inference/saliency helpers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = initialize_pointnet2_model(num_classes, device, normal_channel=use_normals)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model