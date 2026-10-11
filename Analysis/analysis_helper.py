import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mpltern  # noqa: F401  (registers the "ternary" projection used by plot_datasets_grid)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "train_balanced_acc",
    "val_balanced_acc",
    "val_f1",
    "val_class_acc_Discoide",
    "val_class_acc_Levallois",
    "val_class_acc_Laminaire",
    "val_composite_score",
)

# Logged validation score used during training to pick each fold's checkpoint (Table 1 reports that epoch).
SELECTION_COLUMN = "val_composite_score"

CANONICAL_CLASSES = ("Discoidal", "Levallois", "Laminar")
INDEX_TO_CLASS = {0: "Discoidal", 1: "Levallois", 2: "Laminar"}
CLASS_TO_INDEX = {name: idx for idx, name in INDEX_TO_CLASS.items()}
CLASS_ALIASES = {
    "Discoidal": ("discoidal", "discoide"),
    "Levallois": ("levallois",),
    "Laminar": ("laminar", "laminaire"),
}

DATASET_DISPLAY = {
    "Nubian": "Nubian",
    "GBP_Disc": "Discoidal",
    "UoM_Blade": "UoM Blade",
    "UoM_Disc": "UoM Discoidal",
    "UoM_Leva": "UoM Levallois",
    "ModelNet": "ModelNet",
}

DATASET_FIGURE_GROUPS = {
    1: ["Nubian", "GBP_Disc"],
    2: ["UoM_Blade", "UoM_Disc", "UoM_Leva"],
    3: ["ModelNet"],
}

DATASET_TO_GROUP = {
    dataset: group
    for group, datasets in DATASET_FIGURE_GROUPS.items()
    for dataset in datasets
}

COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
}

MODELS = ["PointNet++", "DGCNN"]

# Default subplot spacing for the ternary grids (Figures 4-6), with per-figure overrides.
TERNARY_LAYOUT = dict(left=0.20, right=0.91, top=0.89, bottom=0.12, wspace=0.25, hspace=0.44)
TERNARY_GROUP_LAYOUT = {
    1: dict(wspace=0.75),
    2: dict(),
    3: dict(wspace=0.75, top=0.98, bottom=0.02),
}
# Extra height scaling per figure (the single-row ModelNet figure needs much less height).
TERNARY_GROUP_HEIGHT_SCALE = {3: 0.55}

EMBEDDING_COL_PREFIX = "emb_"

CROSS_COLLECTION_CLASS_PAIRS = {
    "Discoide": ["GBP_Disc", "UoM_Disc"],
    "Levallois": ["Nubian", "UoM_Leva"],
}


# ---------------------------------------------------------------------------
# Loading ensemble predictions
# ---------------------------------------------------------------------------

def resolve_ensemble_path(root_dir: str, dataset_key: str) -> str | None:
    """Path to {root_dir}/{dataset_key}/ensemble_predictions.csv, or None if it does not exist."""
    path = os.path.join(root_dir, dataset_key, "ensemble_predictions.csv")
    return path if os.path.isfile(path) else None


def load_ensemble_df(root_dir: str, dataset_key: str) -> pd.DataFrame:
    path = resolve_ensemble_path(root_dir, dataset_key)
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    if "__dataset" not in df.columns:
        df["__dataset"] = dataset_key
    if "file_id" not in df.columns:
        df["file_id"] = [f"row_{idx}" for idx in range(len(df))]
    return df


def _col_map(df: pd.DataFrame) -> dict[str, str]:
    return {column.lower().replace(" ", "").replace("_", ""): column for column in df.columns}


def resolve_pred_col(df: pd.DataFrame) -> str | None:
    cols = _col_map(df)
    return cols.get("predclass") or cols.get("predictedclass") or cols.get("predlabel")


def resolve_true_col(df: pd.DataFrame) -> str | None:
    cols = _col_map(df)
    return cols.get("trueclass") or cols.get("truelabel") or cols.get("true") or cols.get("truelabels")


def resolve_prob_col(df: pd.DataFrame, canonical_class: str) -> str | None:
    cols = {column.lower(): column for column in df.columns}
    for alias in CLASS_ALIASES[canonical_class]:
        for candidate in (f"{alias}_probability", f"prob_{alias}"):
            if candidate in cols:
                return cols[candidate]
    return None


def normalize_class(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return INDEX_TO_CLASS.get(int(value))
    value_str = str(value).strip().lower()
    if value_str.isdigit():
        return INDEX_TO_CLASS.get(int(value_str))
    for canonical, aliases in CLASS_ALIASES.items():
        if value_str in aliases:
            return canonical
    return None


def to_class_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_class)


def mean_probabilities(df: pd.DataFrame) -> dict[str, float]:
    means: dict[str, float] = {}
    for canonical_class in CANONICAL_CLASSES:
        prob_col = resolve_prob_col(df, canonical_class)
        if not prob_col:
            means[canonical_class] = float("nan")
            continue
        values = pd.to_numeric(df[prob_col], errors="coerce").dropna()
        means[canonical_class] = float(values.mean()) if not values.empty else float("nan")
    return means


def parse_predictions(df: pd.DataFrame) -> list[tuple[int, list[float]]]:
    """Convert an ensemble predictions frame into (pred_class_idx, [p_discoidal, p_levallois, p_laminar])
    tuples, with each probability triplet renormalized to sum to 1.

    Raises if the predicted-class or probability columns are missing, or if any row has an
    unrecognised class label or a non-numeric probability.
    """
    pred_col = resolve_pred_col(df)
    prob_cols = [resolve_prob_col(df, canonical_class) for canonical_class in CANONICAL_CLASSES]
    if pred_col is None or None in prob_cols:
        raise KeyError(f"Expected a predicted-class column and one probability column per class; got {list(df.columns)}")

    pred_classes = to_class_series(df[pred_col])
    unknown = df.loc[pred_classes.isna(), pred_col]
    if not unknown.empty:
        raise ValueError(f"Unrecognised predicted class labels: {sorted(unknown.astype(str).unique())}")

    probs = df[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(probs).any():
        raise ValueError(f"Non-numeric or missing values in probability columns {prob_cols}")
    totals = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, totals, out=probs.copy(), where=totals > 0)

    return [(CLASS_TO_INDEX[pred], row.tolist()) for pred, row in zip(pred_classes, probs)]


def load_predictions_by_group(model_roots: dict[str, str]) -> dict:
    datasets_by_group = {group_idx: {} for group_idx in DATASET_FIGURE_GROUPS}

    for dataset_key, group in DATASET_TO_GROUP.items():
        datasets_by_group[group][dataset_key] = {}
        for model in MODELS:
            ens_df = load_ensemble_df(model_roots[model], dataset_key)
            datasets_by_group[group][dataset_key][model] = [] if ens_df.empty else parse_predictions(ens_df)

    return datasets_by_group


# ---------------------------------------------------------------------------
# Training metrics (Figure 3, Table 1)
# ---------------------------------------------------------------------------

def load_training_metrics(root_dir: str) -> Dict[int, pd.DataFrame]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Metrics directory not found: {root_dir}")

    metrics: Dict[int, pd.DataFrame] = {}
    for entry in sorted(os.listdir(root_dir)):
        fold_path = os.path.join(root_dir, entry)
        if not os.path.isdir(fold_path) or not entry.startswith("fold_"):
            continue
        try:
            fold_idx = int(entry.split("_")[-1])
        except ValueError:
            continue

        csv_path = os.path.join(fold_path, "training_metrics.csv")
        if not os.path.isfile(csv_path):
            continue

        df = pd.read_csv(csv_path)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {csv_path}: {missing}")

        df = df.copy()
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.dropna(subset=["epoch"]).sort_values("epoch")
        df["epoch"] = df["epoch"].astype(int)
        metrics[fold_idx] = df.reset_index(drop=True)

    if not metrics:
        raise RuntimeError(f"No training_metrics.csv files found under {root_dir}")
    return dict(sorted(metrics.items()))


def _mean_std_by_epoch(frames: List[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby("epoch")["value"].agg(mean="mean", std="std").reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def _prepare_metric_stats(metrics_by_fold: Dict[int, pd.DataFrame], train_col: str, val_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_frames: List[pd.DataFrame] = []
    val_frames: List[pd.DataFrame] = []

    for df in metrics_by_fold.values():
        if train_col not in df.columns or val_col not in df.columns:
            continue
        train_frames.append(df[["epoch", train_col]].rename(columns={train_col: "value"}))
        val_frames.append(df[["epoch", val_col]].rename(columns={val_col: "value"}))

    if not train_frames or not val_frames:
        raise KeyError(f"Columns {train_col} / {val_col} not found in any fold")

    return _mean_std_by_epoch(train_frames), _mean_std_by_epoch(val_frames)


def _plot_with_band(ax, stats: pd.DataFrame, color: str, label: str) -> None:
    ax.plot(stats["epoch"], stats["mean"], color=color, label=label)
    ax.fill_between(
        stats["epoch"],
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        color=color,
        alpha=0.2,
    )


def _label_panel(ax, text: str) -> None:
    ax.text(
        0.02,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def selected_epoch_rows(metrics_by_fold: Dict[int, pd.DataFrame]) -> Dict[int, pd.Series]:
    """Each fold's metrics at the checkpoint selected during training: the epoch with the highest
    logged SELECTION_COLUMN (the earliest such epoch if tied)."""
    rows: Dict[int, pd.Series] = {}
    for fold, df in metrics_by_fold.items():
        scores = pd.to_numeric(df[SELECTION_COLUMN], errors="coerce")
        if scores.isna().all():
            raise ValueError(f"No {SELECTION_COLUMN} values logged for fold {fold}")
        rows[fold] = df.loc[scores.idxmax()]
    return rows


def selected_epochs_table(metrics_by_fold: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Selected epoch per fold, alongside the last logged epoch, for reporting/checking."""
    selected = selected_epoch_rows(metrics_by_fold)
    return pd.DataFrame(
        [
            {
                "fold": fold,
                "selected_epoch": int(row["epoch"]),
                "last_epoch": int(metrics_by_fold[fold]["epoch"].iloc[-1]),
                SELECTION_COLUMN: float(row[SELECTION_COLUMN]),
                "val_balanced_acc": float(row["val_balanced_acc"]),
            }
            for fold, row in selected.items()
        ]
    )


def _selected_metric_values(metrics_by_fold: Dict[int, pd.DataFrame], column: str) -> List[float]:
    values = [float(row[column]) for row in selected_epoch_rows(metrics_by_fold).values() if pd.notna(row[column])]
    if not values:
        raise KeyError(f"Column {column} missing in all folds")
    return values


def _format_mean_std(values: List[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.3f} ± {arr.std(ddof=0):.3f}"


def summarize_validation_metrics(
    pointnet2_dir: str,
    dgcnn_dir: str,
    *,
    pointnet2_metrics: Dict[int, pd.DataFrame] | None = None,
    dgcnn_metrics: Dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Table 1: validation metrics at each fold's selected checkpoint (see `selected_epoch_rows`),
    summarised as mean ± SD across folds."""
    pn_metrics = pointnet2_metrics or load_training_metrics(pointnet2_dir)
    dg_metrics = dgcnn_metrics or load_training_metrics(dgcnn_dir)

    specs = [
        ("Validation balanced accuracy", "val_balanced_acc"),
        ("Validation F1 score", "val_f1"),
        ("Validation accuracy (Discoide)", "val_class_acc_Discoide"),
        ("Validation accuracy (Levallois)", "val_class_acc_Levallois"),
        ("Validation accuracy (Laminaire)", "val_class_acc_Laminaire"),
    ]

    rows = []
    for label, column in specs:
        rows.append(
            {
                "Metric": label,
                "PointNet++ (mean ± SD)": _format_mean_std(_selected_metric_values(pn_metrics, column)),
                "DGCNN (mean ± SD)": _format_mean_std(_selected_metric_values(dg_metrics, column)),
            }
        )
    return pd.DataFrame(rows)


def plot_training_overview(
    pointnet2_dir: str,
    dgcnn_dir: str,
    *,
    figsize: Tuple[float, float] = (11.0, 8.0),
    train_color: str = "#1B9E77",
    val_color: str = "#D95F02",
    save_path: str | None = None,
    dpi: int = 300,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    pn_metrics = load_training_metrics(pointnet2_dir)
    dg_metrics = load_training_metrics(dgcnn_dir)

    pn_loss_train, pn_loss_val = _prepare_metric_stats(pn_metrics, "train_loss", "val_loss")
    pn_acc_train, pn_acc_val = _prepare_metric_stats(pn_metrics, "train_balanced_acc", "val_balanced_acc")
    dg_loss_train, dg_loss_val = _prepare_metric_stats(dg_metrics, "train_loss", "val_loss")
    dg_acc_train, dg_acc_val = _prepare_metric_stats(dg_metrics, "train_balanced_acc", "val_balanced_acc")

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="col")

    _plot_with_band(axes[0, 0], pn_loss_train, train_color, "Train")
    _plot_with_band(axes[0, 0], pn_loss_val, val_color, "Validation")
    axes[0, 0].set_title("PointNet++ Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3, linestyle="--")
    axes[0, 0].legend(loc="upper right")

    _plot_with_band(axes[0, 1], pn_acc_train, train_color, "Train")
    _plot_with_band(axes[0, 1], pn_acc_val, val_color, "Validation")
    axes[0, 1].set_title("PointNet++ Balanced Accuracy")
    axes[0, 1].set_ylabel("Balanced accuracy")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(alpha=0.3, linestyle="--")
    axes[0, 1].legend(loc="lower right")

    _plot_with_band(axes[1, 0], dg_loss_train, train_color, "Train")
    _plot_with_band(axes[1, 0], dg_loss_val, val_color, "Validation")
    axes[1, 0].set_title("DGCNN Loss")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(alpha=0.3, linestyle="--")
    axes[1, 0].legend(loc="upper right")

    _plot_with_band(axes[1, 1], dg_acc_train, train_color, "Train")
    _plot_with_band(axes[1, 1], dg_acc_val, val_color, "Validation")
    axes[1, 1].set_title("DGCNN Balanced Accuracy")
    axes[1, 1].set_ylabel("Balanced accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(alpha=0.3, linestyle="--")
    axes[1, 1].legend(loc="lower right")

    for ax, label in zip(axes.flatten(), ["a", "b", "c", "d"]):
        _label_panel(ax, label)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return pn_metrics, dg_metrics


# ---------------------------------------------------------------------------
# CLR + PERMANOVA on predicted probability compositions
# ---------------------------------------------------------------------------

def clr_transform(compositions: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Centered log-ratio transform (Aitchison 1986) for a 2D array of compositions (rows sum to ~1).

    Zero/near-zero entries (softmax can underflow to exactly 0.0 in float32) are replaced with
    `epsilon` and the row is renormalized to sum to 1 before taking logs, since CLR is undefined at 0.
    """
    x = np.asarray(compositions, dtype=float)
    x = np.where(x <= epsilon, epsilon, x)
    x = x / x.sum(axis=1, keepdims=True)
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def _permanova_pseudo_f(x: np.ndarray, groups: np.ndarray) -> float:
    """Pseudo-F statistic for PERMANOVA on Euclidean distances, computed via centroid sum-of-squares.

    For Euclidean distances this is algebraically equivalent to the distance-matrix formulation in
    Anderson (2001), and much cheaper than building the full N x N distance matrix.
    """
    n_total = x.shape[0]
    unique_groups = np.unique(groups)
    a = len(unique_groups)

    grand_mean = x.mean(axis=0)
    ss_total = float(np.sum((x - grand_mean) ** 2))

    ss_within = 0.0
    for g in unique_groups:
        mask = groups == g
        group_mean = x[mask].mean(axis=0)
        ss_within += float(np.sum((x[mask] - group_mean) ** 2))

    ss_between = ss_total - ss_within
    df_between = a - 1
    df_within = n_total - a
    if df_between <= 0 or df_within <= 0 or ss_within <= 0:
        return float("nan")
    return (ss_between / df_between) / (ss_within / df_within)


def permanova_euclidean(
    x: np.ndarray,
    groups: np.ndarray,
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """Permutational MANOVA (Anderson 2001) on Euclidean distances between rows of `x`.

    Intended to be run on CLR-transformed compositional data (see `clr_transform`), which turns
    the constrained simplex into an unconstrained Euclidean space where this test is valid.
    Returns a dict with the observed pseudo-F, permutation p-value, group/df info, and sample size.
    """
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)

    f_obs = _permanova_pseudo_f(x, groups)
    if np.isnan(f_obs):
        return {
            "n": x.shape[0],
            "n_groups": len(np.unique(groups)),
            "F": float("nan"),
            "p_value": float("nan"),
            "permutations": permutations,
        }

    count_ge = 0
    for _ in range(permutations):
        permuted = rng.permutation(groups)
        f_perm = _permanova_pseudo_f(x, permuted)
        if not np.isnan(f_perm) and f_perm >= f_obs:
            count_ge += 1

    p_value = (count_ge + 1) / (permutations + 1)
    return {
        "n": x.shape[0],
        "n_groups": len(np.unique(groups)),
        "F": f_obs,
        "p_value": p_value,
        "permutations": permutations,
    }


def clr_permanova_by_predicted_class(
    predictions: list[tuple[int, list[float]]],
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """CLR-transform predicted class-probability triplets, then PERMANOVA-test whether the
    distribution of probability vectors differs across the *predicted*-class groups.

    `predictions` is the list of (pred_class_idx, [p_discoidal, p_levallois, p_laminar]) tuples
    produced by `parse_predictions` / `load_predictions_by_group`. Note: within a single
    dataset_key, all rows share one ground-truth label, so grouping by true class is not
    possible at this call site — predicted class is used instead.
    """
    if not predictions:
        return {"n": 0, "n_groups": 0, "F": float("nan"), "p_value": float("nan"), "permutations": permutations}

    groups = np.array([pred_idx for pred_idx, _ in predictions])
    comps = np.array([probs for _, probs in predictions])

    if len(np.unique(groups)) < 2:
        return {"n": comps.shape[0], "n_groups": len(np.unique(groups)), "F": float("nan"), "p_value": float("nan"), "permutations": permutations}

    clr_comps = clr_transform(comps)
    return permanova_euclidean(clr_comps, groups, permutations=permutations, seed=seed)


def clr_permanova_between_models(
    datasets_by_group: dict,
    dataset_figure_groups: dict = DATASET_FIGURE_GROUPS,
    dataset_display: dict = DATASET_DISPLAY,
    models: list[str] = MODELS,
    permutations: int = 999,
    seed: int = 42,
) -> pd.DataFrame:
    """For each dataset, test whether probability compositions differ between models."""
    rows = []
    for group_idx, dataset_keys in dataset_figure_groups.items():
        for dataset_key in dataset_keys:
            comps, labels = [], []
            for model in models:
                preds = datasets_by_group.get(group_idx, {}).get(dataset_key, {}).get(model, [])
                for _, probs in preds:
                    comps.append(probs)
                    labels.append(model)
            if len(set(labels)) < 2:
                continue
            clr_comps = clr_transform(np.array(comps))
            result = permanova_euclidean(clr_comps, np.array(labels), permutations=permutations, seed=seed)
            rows.append({
                "Dataset": dataset_display.get(dataset_key, dataset_key),
                "Comparison": " vs ".join(models),
                **result,
                "significant (p<0.05)": pd.notna(result["p_value"]) and result["p_value"] < 0.05,
            })
    return pd.DataFrame(rows)


def clr_permanova_between_datasets(
    datasets_by_group: dict,
    dataset_figure_groups: dict = DATASET_FIGURE_GROUPS,
    dataset_display: dict = DATASET_DISPLAY,
    models: list[str] = MODELS,
    permutations: int = 999,
    seed: int = 42,
) -> pd.DataFrame:
    """For each model, test whether probability compositions differ across datasets."""
    rows = []
    for model in models:
        comps, labels = [], []
        for group_idx, dataset_keys in dataset_figure_groups.items():
            for dataset_key in dataset_keys:
                preds = datasets_by_group.get(group_idx, {}).get(dataset_key, {}).get(model, [])
                for _, probs in preds:
                    comps.append(probs)
                    labels.append(dataset_display.get(dataset_key, dataset_key))
        if len(set(labels)) < 2:
            continue
        clr_comps = clr_transform(np.array(comps))
        result = permanova_euclidean(clr_comps, np.array(labels), permutations=permutations, seed=seed)
        rows.append({
            "Model": model,
            "Comparison": "Across datasets",
            **result,
            "significant (p<0.05)": pd.notna(result["p_value"]) and result["p_value"] < 0.05,
        })
    return pd.DataFrame(rows)


def permanova_summary_table(between_model_df: pd.DataFrame, between_dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Table S8: between-model results (one row per dataset) followed by among-dataset results
    (one row per model), from `clr_permanova_between_models` / `clr_permanova_between_datasets`."""
    def rows(df: pd.DataFrame, comparison: str, label_col: str) -> pd.DataFrame:
        return pd.DataFrame({
            "Comparison": comparison,
            "Dataset / Model": df[label_col],
            "n": df["n"],
            "F": df["F"].round(3),
            "p value": df["p_value"].round(3),
            "Significant (p<0.05)": df["significant (p<0.05)"],
        })

    return pd.concat(
        [
            rows(between_model_df, "PointNet++ vs DGCNN (per dataset)", "Dataset"),
            rows(between_dataset_df, "Among six datasets (per model)", "Model"),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Ternary plots (Figures 4-6)
# ---------------------------------------------------------------------------

def plot_datasets_grid(dataset_keys, predictions_by_dataset, models_list, figsize=(10, 8), **layout):
    """One row of ternary plots per dataset and one column per model. Points are each flake's
    predicted probabilities, coloured by predicted class; the grey cross is the mean composition.

    `layout` overrides entries of TERNARY_LAYOUT (left/right/top/bottom/wspace/hspace).
    """
    params = {**TERNARY_LAYOUT, **layout}
    fig = plt.figure(figsize=figsize)
    n_rows, n_cols = len(dataset_keys), len(models_list)

    axes_grid = [
        [fig.add_subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1, projection="ternary") for col_idx in range(n_cols)]
        for row_idx in range(n_rows)
    ]
    # Apply the final spacing before placing figure-level labels, so they line up with the axes.
    fig.subplots_adjust(**params)

    for row_idx, dataset_key in enumerate(dataset_keys):
        row_box = axes_grid[row_idx][0].get_position()
        y_pos = (row_box.y0 + row_box.y1) / 2
        dataset_name = DATASET_DISPLAY.get(dataset_key, dataset_key)
        fig.text(0.12, y_pos, dataset_name, rotation=90, va="center", ha="left", fontsize=10, fontweight="bold")

        for col_idx, model in enumerate(models_list):
            ax = axes_grid[row_idx][col_idx]
            predictions = predictions_by_dataset.get(dataset_key, {}).get(model, [])

            if not predictions:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            pred_class_idx = np.array([pred_idx for pred_idx, _ in predictions])
            probs = np.array([pred_dist for _, pred_dist in predictions])
            point_colors = [COLORS.get(idx, "#808080") for idx in pred_class_idx]
            ax.scatter(probs[:, 0], probs[:, 1], probs[:, 2], color=point_colors, s=24, alpha=0.6)

            mean_comp = probs.mean(axis=0)
            if mean_comp.sum() > 0:
                mean_comp = mean_comp / mean_comp.sum()
            ax.scatter([mean_comp[0]], [mean_comp[1]], [mean_comp[2]], color="#808080", marker="x", s=85, linewidths=2, zorder=10)

            ax.set_tlabel("Discoidal", fontsize=9, labelpad=6)
            ax.set_llabel("Levallois", fontsize=9, labelpad=6)
            ax.set_rlabel("Laminar", fontsize=9, labelpad=6)
            ax.grid(True, alpha=0.3)
            panel_label = chr(97 + row_idx * n_cols + col_idx)
            ax.text(0.05, 0.95, panel_label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    for col_idx, model in enumerate(models_list):
        col_box = axes_grid[0][col_idx].get_position()
        fig.text((col_box.x0 + col_box.x1) / 2, 0.98, model, ha="center", fontsize=11)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=COLORS[idx], markersize=6)
        for idx, name in INDEX_TO_CLASS.items()
    ]
    fig.legend(handles=legend_elements, title="Predicted class", loc="lower center", ncol=3, frameon=False, fontsize=9, title_fontsize=9)
    return fig


def _ternary_axes_fraction(dataset_keys, datasets_dict, models_list, figsize=(1.0, 1.0)) -> Tuple[float, float]:
    """Width/height of one ternary subplot as a fraction of the figure, using the default TERNARY_LAYOUT.
    The ternary axes keep a fixed aspect ratio, so the result depends on `figsize`."""
    fig = plot_datasets_grid(dataset_keys, datasets_dict, models_list, figsize=figsize)
    box = fig.axes[0].get_position()
    plt.close(fig)
    return box.width, box.height


def save_ternary_figures(
    datasets_by_group: dict,
    output_dir: str | Path,
    models_list: list[str] = MODELS,
    *,
    first_figure_number: int = 4,
    reference_group: int = 2,
    reference_figsize: Tuple[float, float] = (8.6, 9.4),
    dpi: int = 300,
    show: bool = True,
) -> None:
    """Save Figures 4-6 (one per entry in DATASET_FIGURE_GROUPS), sizing each figure so that every
    ternary subplot has the same physical size as in the reference group's figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_w, ref_h = _ternary_axes_fraction(
        DATASET_FIGURE_GROUPS[reference_group], datasets_by_group[reference_group], models_list, reference_figsize
    )
    target_ax_w_in = ref_w * reference_figsize[0]
    target_ax_h_in = ref_h * reference_figsize[1]

    for offset, (group_idx, dataset_keys) in enumerate(DATASET_FIGURE_GROUPS.items()):
        frac_w, frac_h = _ternary_axes_fraction(dataset_keys, datasets_by_group[group_idx], models_list)
        fig_w = max(2.0, target_ax_w_in / frac_w)
        fig_h = max(2.0, target_ax_h_in / frac_h) * TERNARY_GROUP_HEIGHT_SCALE.get(group_idx, 1.0)

        layout = TERNARY_GROUP_LAYOUT.get(group_idx, {})

        fig = plot_datasets_grid(dataset_keys, datasets_by_group[group_idx], models_list, figsize=(fig_w, fig_h), **layout)
        figure_num = first_figure_number + offset
        save_path = output_dir / f"Figure_{figure_num}.jpg"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved Figure {figure_num} to {save_path} (figsize={fig_w:.2f}x{fig_h:.2f})")
        if show:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary tables (Tables 2-4, S2-S6, S11)
# ---------------------------------------------------------------------------

def predicted_distribution(df: pd.DataFrame) -> dict[str, float]:
    pred_col = resolve_pred_col(df)
    if pred_col is None:
        return {canonical_class: float("nan") for canonical_class in CANONICAL_CLASSES}
    pred_series = to_class_series(df[pred_col]).dropna()
    total = len(pred_series)
    return {
        canonical_class: ((pred_series == canonical_class).sum() / total * 100) if total else 0.0
        for canonical_class in CANONICAL_CLASSES
    }


def target_recall(df: pd.DataFrame, canonical_class: str) -> float:
    pred_col = resolve_pred_col(df)
    if pred_col is None:
        return float("nan")
    pred_series = to_class_series(df[pred_col]).dropna()
    if pred_series.empty:
        return float("nan")
    return float((pred_series == canonical_class).mean() * 100)


def mean_prob_for_target(df: pd.DataFrame, canonical_class: str) -> float:
    prob_col = resolve_prob_col(df, canonical_class)
    if prob_col:
        values = pd.to_numeric(df[prob_col], errors="coerce").dropna()
        if not values.empty:
            return float(values.mean())
    if "confidence" in df.columns:
        values = pd.to_numeric(df["confidence"], errors="coerce").dropna()
        if not values.empty:
            return float(values.mean())
    return float("nan")


def merge_dataset_ensembles(root_dir: str, dataset_keys: list[str]) -> pd.DataFrame:
    frames = []
    for dataset_key in dataset_keys:
        df = load_ensemble_df(root_dir, dataset_key)
        if not df.empty:
            df = df.copy()
            df["__dataset"] = dataset_key
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def combined_multiclass_metrics(df: pd.DataFrame) -> dict[str, float]:
    pred_col = resolve_pred_col(df)
    true_col = resolve_true_col(df)
    if pred_col is None or true_col is None or df.empty:
        return {
            "balanced_accuracy": float("nan"),
            "Discoidal": float("nan"),
            "Levallois": float("nan"),
            "Laminar": float("nan"),
            "mean_pred_prob": float("nan"),
        }

    pred_series = to_class_series(df[pred_col]).str.strip()
    true_series = to_class_series(df[true_col]).str.strip()

    recalls = {}
    for canonical_class in CANONICAL_CLASSES:
        mask = true_series == canonical_class
        recalls[canonical_class] = float((pred_series[mask] == canonical_class).mean() * 100) if mask.any() else float("nan")

    true_class_probs = []
    for canonical_class in CANONICAL_CLASSES:
        prob_col = resolve_prob_col(df, canonical_class)
        if not prob_col:
            continue
        mask = true_series == canonical_class
        values = pd.to_numeric(df.loc[mask, prob_col], errors="coerce").dropna()
        if not values.empty:
            true_class_probs.extend(values.tolist())

    return {
        "balanced_accuracy": float(np.nanmean(list(recalls.values()))),
        "Discoidal": recalls["Discoidal"],
        "Levallois": recalls["Levallois"],
        "Laminar": recalls["Laminar"],
        "mean_pred_prob": float(np.mean(true_class_probs)) if true_class_probs else float("nan"),
    }


def mean_probs_by_true(df: pd.DataFrame, *, filter_correct: bool = False) -> pd.DataFrame:
    pred_col = resolve_pred_col(df)
    true_col = resolve_true_col(df)
    if pred_col is None or true_col is None or df.empty:
        return pd.DataFrame()

    pred_series = to_class_series(df[pred_col]).str.strip()
    true_series = to_class_series(df[true_col]).str.strip()
    rows = []

    for canonical_class in CANONICAL_CLASSES:
        mask = true_series == canonical_class
        if filter_correct:
            mask &= pred_series == canonical_class
        subset = df.loc[mask]
        means = mean_probabilities(subset) if not subset.empty else {name: np.nan for name in CANONICAL_CLASSES}
        rows.append(
            {
                "True Class": canonical_class,
                "Mean P(Discoidal)": means["Discoidal"],
                "Mean P(Levallois)": means["Levallois"],
                "Mean P(Laminar)": means["Laminar"],
            }
        )

    out = pd.DataFrame(rows)
    for col in ["Mean P(Discoidal)", "Mean P(Levallois)", "Mean P(Laminar)"]:
        out[col] = out[col].map(lambda value: f"{value:.3f}" if pd.notna(value) else "N/A")
    return out


def mean_probs_by_pred_class(df: pd.DataFrame) -> pd.DataFrame:
    pred_col = resolve_pred_col(df)
    if pred_col is None or df.empty:
        return pd.DataFrame()
    pred_series = to_class_series(df[pred_col])
    rows = []
    for canonical_class in CANONICAL_CLASSES:
        subset = df.loc[pred_series == canonical_class]
        means = mean_probabilities(subset) if not subset.empty else {name: np.nan for name in CANONICAL_CLASSES}
        rows.append(
            {
                "Predicted Class": canonical_class,
                "Mean P(Discoidal)": means["Discoidal"],
                "Mean P(Levallois)": means["Levallois"],
                "Mean P(Laminar)": means["Laminar"],
            }
        )
    out = pd.DataFrame(rows)
    for col in ["Mean P(Discoidal)", "Mean P(Levallois)", "Mean P(Laminar)"]:
        out[col] = out[col].map(lambda value: f"{value:.3f}" if pd.notna(value) else "N/A")
    return out


def canonical_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["__dataset", "file_id", "pred", "true"])
    pred_col = resolve_pred_col(df)
    true_col = resolve_true_col(df)
    out = df[["__dataset", "file_id"]].copy()
    out["pred"] = df[pred_col].apply(normalize_class) if pred_col else np.nan
    out["true"] = df[true_col].apply(normalize_class) if true_col else np.nan
    return out


def build_summary_table(merged: pd.DataFrame, has_true: bool) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame()

    if has_true:
        true_series = merged["true_pn"].combine_first(merged["true_dg"])
        valid = true_series.notna() & merged["pred_pn"].notna() & merged["pred_dg"].notna()
        eval_df = merged.loc[valid].copy()
        true_eval = true_series.loc[valid]

        correct_pn = eval_df["pred_pn"] == true_eval
        correct_dg = eval_df["pred_dg"] == true_eval
        counts = [
            int((correct_pn & correct_dg).sum()),
            int((~correct_pn & ~correct_dg).sum()),
            int((correct_pn & ~correct_dg).sum()),
            int((~correct_pn & correct_dg).sum()),
            int(len(eval_df)),
        ]
        outcomes = [
            "Correctly classified by both models",
            "Misclassified by both models",
            "Correct only by PointNet++",
            "Correct only by DGCNN",
            "Total",
        ]
    else:
        valid = merged["pred_pn"].notna() & merged["pred_dg"].notna()
        eval_df = merged.loc[valid].copy()
        agree = eval_df["pred_pn"] == eval_df["pred_dg"]
        counts = [int(agree.sum()), int((~agree).sum()), 0, 0, int(len(eval_df))]
        outcomes = [
            "Both models agree",
            "Both models disagree",
            "PointNet++ only",
            "DGCNN only",
            "Total",
        ]

    total = counts[-1]
    percentages = [f"{(100.0 * count / total) if total else 0.0:.1f}" for count in counts[:-1]] + ["100.0" if total else "0.0"]
    return pd.DataFrame(
        {
            "Prediction outcome": outcomes,
            "Number of flakes": counts,
            "Percentage (%)": percentages,
        }
    )


# ---------------------------------------------------------------------------
# Saliency rank curves
# ---------------------------------------------------------------------------

def extract_curve_from_npy(npy_path: Path, pct_steps: list[int]):
    arr = np.load(npy_path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None

    saliency = arr[:, 3].astype(float)
    if saliency.size == 0:
        return None

    saliency_sorted = np.sort(saliency)[::-1]
    total = saliency_sorted.sum()
    cumulative_norm = np.zeros_like(saliency_sorted) if total <= 0 else np.cumsum(saliency_sorted) / total

    points = []
    n = len(saliency_sorted)
    for pct in pct_steps:
        k = max(1, math.ceil(n * pct / 100.0))
        points.append((pct, float(cumulative_norm[k - 1])))
    return points


def load_saliency_curves(batches_root: str | Path, source_folders: list[str], pct_steps: list[int]) -> pd.DataFrame:
    """Cumulative saliency curves for every flake in {batches_root}/{folder}/critical_points/*_critical.npy."""
    records = []
    for folder in source_folders:
        cp_dir = Path(batches_root) / folder / "critical_points"
        if not cp_dir.exists():
            continue
        for npy_file in sorted(cp_dir.glob("*_critical.npy")):
            curve = extract_curve_from_npy(npy_file, pct_steps)
            if curve is None:
                continue
            flake_id = npy_file.name.replace("_critical.npy", "")
            for pct, cum_val in curve:
                records.append(
                    {"source_dataset": folder, "flake_id": flake_id, "percentage": pct, "cumulative_saliency": cum_val}
                )
    return pd.DataFrame(records, columns=["source_dataset", "flake_id", "percentage", "cumulative_saliency"])


def summarize_saliency_curves(curve_df: pd.DataFrame) -> pd.DataFrame:
    return (
        curve_df.groupby("percentage", as_index=False)
        .agg(
            mean_cumulative_saliency=("cumulative_saliency", "mean"),
            std_cumulative_saliency=("cumulative_saliency", "std"),
            n_flakes=("flake_id", "nunique"),
        )
        .sort_values("percentage")
    )


def plot_saliency_curves(curve_df: pd.DataFrame, title: str) -> go.Figure:
    """One stacked panel per flake showing its cumulative saliency rank curve."""
    flake_order = sorted(curve_df["flake_id"].unique())
    fig = make_subplots(
        rows=len(flake_order),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.003,
        subplot_titles=flake_order,
    )

    for row_idx, flake in enumerate(flake_order, start=1):
        flake_df = curve_df[curve_df["flake_id"] == flake].sort_values("percentage")
        fig.add_trace(
            go.Scatter(
                x=flake_df["percentage"],
                y=flake_df["cumulative_saliency"],
                mode="lines+markers",
                line=dict(color="royalblue", width=2),
                marker=dict(size=4),
                name=flake,
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(range=[0, 1], row=row_idx, col=1)

    fig.update_layout(
        title=f"{title} ({len(flake_order)} flakes)",
        template="plotly_white",
        height=max(500, 80 * len(flake_order)),
        width=1200,
        margin=dict(l=60, r=20, t=70, b=60),
    )
    fig.update_xaxes(title_text="Top-k percentage of ranked critical points", row=len(flake_order), col=1)
    return fig


# ---------------------------------------------------------------------------
# Cross-validation misclassifications by reduction sequence (Tables S12-S13)
# ---------------------------------------------------------------------------

def extract_sequence_id(file_id: str) -> str:
    """Extract the reduction-sequence identifier from a flake file_id, defined as the
    substring before the first underscore (matching the convention used for
    sequence-grouped cross-validation)."""
    return str(file_id).split("_", 1)[0]


def load_cv_confusion_data(
    training_root: str,
    n_folds: int = 5,
    split: str = "val",
    filename_prefix: str = "",
) -> pd.DataFrame:
    """Load and concatenate per-fold confusion data (val or train) from a PointNet2/DGCNN
    training output directory, tagging each row with its fold and extracted sequence ID.

    Expects files at {training_root}/fold_{k}/{filename_prefix}{split}_confusion_data.csv
    with columns 'Flake ID', 'True Labels', 'Predicted Labels'.

    filename_prefix lets you match different naming conventions across model outputs, e.g.:
      - PointNet++: filename_prefix="" -> "val_confusion_data.csv"
      - DGCNN:      filename_prefix="best_" -> "best_val_confusion_data.csv"
    """
    frames = []
    for fold in range(1, n_folds + 1):
        path = os.path.join(training_root, f"fold_{fold}", f"{filename_prefix}{split}_confusion_data.csv")
        if not os.path.exists(path):
            print(f"Warning: missing {path}")
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={"Flake ID": "file_id", "True Labels": "true", "Predicted Labels": "pred"})
        df["fold"] = fold
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["file_id", "true", "pred", "fold"])
    out = pd.concat(frames, ignore_index=True)
    out["true"] = out["true"].apply(normalize_class)
    out["pred"] = out["pred"].apply(normalize_class)
    out["sequence_id"] = out["file_id"].apply(extract_sequence_id)
    out["misclassified"] = out["true"] != out["pred"]
    return out


def class_frequency_bias(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Table S7: for each fold, the proportion of flakes in each true class versus the proportion
    predicted as that class, from the held-out (validation) predictions in `load_cv_confusion_data`."""
    rows = []
    for fold, fold_df in cv_df.groupby("fold"):
        for canonical_class in CANONICAL_CLASSES:
            true_prop = float((fold_df["true"] == canonical_class).mean())
            pred_prop = float((fold_df["pred"] == canonical_class).mean())
            rows.append(
                {
                    "Fold": fold,
                    "Class": canonical_class,
                    "n": len(fold_df),
                    "True proportion": true_prop,
                    "Predicted proportion": pred_prop,
                    "Predicted − True": pred_prop - true_prop,
                }
            )
    return pd.DataFrame(rows)


def check_sequence_class_collisions(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Report sequence_ids that occur under more than one true class (see `disambiguate_sequence_id`)."""
    class_counts = (
        cv_df.groupby("sequence_id")["true"]
        .nunique()
        .reset_index(name="n_distinct_true_classes")
    )
    collisions = class_counts.loc[class_counts["n_distinct_true_classes"] > 1]
    if collisions.empty:
        print("No collisions found.")
    else:
        print(f"Found {len(collisions)} colliding sequence_id(s):")
        for seq_id in collisions["sequence_id"]:
            detail = cv_df.loc[cv_df["sequence_id"] == seq_id, ["file_id", "true"]].drop_duplicates()
            print(f"\n  sequence_id = {seq_id}")
            display(detail)
    return collisions


def disambiguate_sequence_id(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve sequence_id collisions across classes by appending the true class label,
    since a single reduction sequence cannot span more than one true class. This is
    safe even for non-colliding sequences (it just makes every ID more specific)."""
    out = cv_df.copy()
    out["sequence_id"] = out["sequence_id"].astype(str) + "_" + out["true"].astype(str)
    return out


def cv_shared_misclassification_by_sequence(pn_cv_df: pd.DataFrame, dg_cv_df: pd.DataFrame) -> dict:
    """Identify training flakes misclassified by BOTH PointNet++ and DGCNN during
    cross-validation, and summarize whether these shared errors cluster by reduction
    sequence or by true class.
    """
    merged = pd.merge(
        pn_cv_df[["file_id", "true", "pred", "sequence_id", "misclassified"]],
        dg_cv_df[["file_id", "true", "pred", "misclassified"]],
        on="file_id",
        suffixes=("_pn", "_dg"),
    )
    if merged.empty:
        return {"by_sequence": pd.DataFrame(), "by_true_class": pd.DataFrame(), "chi2_test": {}, "merged": merged}

    # Sanity check: true labels should agree between the two models' CV files
    mismatch = merged["true_pn"] != merged["true_dg"]
    if mismatch.any():
        print(f"Warning: {mismatch.sum()} flakes have mismatched true labels between PN and DG CV files — check file_id alignment.")

    merged["true"] = merged["true_pn"]
    merged["misclassified_both"] = merged["misclassified_pn"] & merged["misclassified_dg"]

    seq_summary = (
        merged.groupby("sequence_id")
        .agg(total_flakes=("file_id", "count"), misclassified_both=("misclassified_both", "sum"))
        .reset_index()
    )
    seq_summary["error_rate"] = seq_summary["misclassified_both"] / seq_summary["total_flakes"]
    seq_summary = seq_summary.sort_values("error_rate", ascending=False).reset_index(drop=True)

    class_summary = (
        merged.loc[merged["misclassified_both"]]
        .groupby("true")
        .size()
        .reindex(CANONICAL_CLASSES, fill_value=0)
        .rename("misclassified_both_count")
        .reset_index()
        .rename(columns={"true": "True class"})
    )
    total_errors = int(merged["misclassified_both"].sum())
    class_summary["% of shared errors"] = (100 * class_summary["misclassified_both_count"] / max(total_errors, 1)).round(1)

    chi2_test = {}
    if total_errors > 0 and len(seq_summary) > 1:
        expected = seq_summary["total_flakes"] / seq_summary["total_flakes"].sum() * total_errors
        observed = seq_summary["misclassified_both"].to_numpy(dtype=float)
        chi2_stat = float(np.sum((observed - expected) ** 2 / np.where(expected > 0, expected, np.nan)))
        dof = len(seq_summary) - 1
        p_value = float(1 - scipy_stats.chi2.cdf(chi2_stat, dof)) if dof > 0 else float("nan")
        chi2_test = {
            "chi2": chi2_stat, "dof": dof, "p_value": p_value,
            "min_expected_count": float(expected.min()) if len(expected) else float("nan"),
            "note": "Chi-square approximation may be unreliable if min_expected_count < 5.",
        }

    return {"by_sequence": seq_summary, "by_true_class": class_summary, "chi2_test": chi2_test, "merged": merged}


# ---------------------------------------------------------------------------
# Learned embeddings: t-SNE, k-NN class purity, silhouette (Figure S1, Tables S9-S10)
# ---------------------------------------------------------------------------

def load_embeddings_all_folds(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a multi-fold embeddings CSV (flake_id, dataset, true_class, pred_idx, fold,
    emb_0...emb_N) and split into metadata and the raw embedding matrix."""
    df = pd.read_csv(csv_path)
    emb_cols = [c for c in df.columns if c.startswith(EMBEDDING_COL_PREFIX)]
    meta = df.drop(columns=emb_cols).reset_index(drop=True)
    embeddings = df[emb_cols].to_numpy(dtype=float)
    return meta, embeddings


def run_tsne_on_embeddings(
    embeddings: np.ndarray,
    pca_components: int = 50,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embedding dimensionality via PCA (standard preprocessing for t-SNE on
    high-dimensional features), then project to 2D with t-SNE."""
    n_components = min(pca_components, embeddings.shape[0] - 1, embeddings.shape[1])
    reduced = PCA(n_components=n_components, random_state=random_state).fit_transform(embeddings)
    perplexity = min(perplexity, (embeddings.shape[0] - 1) / 3)  # sklearn constraint
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca")
    return tsne.fit_transform(reduced)


def compute_neighbor_class_purity(coords: np.ndarray, true_classes: np.ndarray, k: int = 10) -> np.ndarray:
    """For each point, the fraction of its k nearest neighbors (excluding itself)
    sharing its own true_class label."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idxs = nn.kneighbors(coords)
    purity = np.empty(len(coords))
    for i in range(len(coords)):
        neighbor_idxs = idxs[i][1:]
        purity[i] = np.mean(true_classes[neighbor_idxs] == true_classes[i])
    return purity


def run_tsne_purity_all_folds(
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    pca_components: int = 50,
    perplexity: float = 30.0,
    k: int = 10,
    random_state: int = 42,
) -> dict:
    """For each fold independently: run PCA+t-SNE, compute per-dataset k-NN class
    purity, and compute within-class cross-collection silhouette scores. Returns
    per-fold results plus a cross-fold summary (mean/std) — embeddings/coordinates
    are never pooled or averaged across folds.
    """
    folds = sorted(meta["fold"].unique())
    per_fold_purity_rows = []
    per_fold_silhouette_rows = []
    tsne_coords_by_fold = {}

    for fold in folds:
        fold_mask = (meta["fold"] == fold).to_numpy()
        fold_meta = meta.loc[fold_mask].reset_index(drop=True)
        fold_emb = embeddings[fold_mask]

        coords = run_tsne_on_embeddings(
            fold_emb, pca_components=pca_components, perplexity=perplexity, random_state=random_state
        )
        tsne_coords_by_fold[fold] = coords

        labeled_mask = fold_meta["true_class"].notna().to_numpy()
        labeled_coords = coords[labeled_mask]
        labeled_true = fold_meta.loc[labeled_mask, "true_class"].to_numpy()
        labeled_dataset = fold_meta.loc[labeled_mask, "dataset"].to_numpy()

        purity = compute_neighbor_class_purity(labeled_coords, labeled_true, k=k)
        purity_df = pd.DataFrame({"dataset": labeled_dataset, "true_class": labeled_true, "purity": purity})
        fold_summary = purity_df.groupby(["dataset", "true_class"])["purity"].mean().reset_index()
        fold_summary["fold"] = fold
        per_fold_purity_rows.append(fold_summary)

        for true_class, ds_pair in CROSS_COLLECTION_CLASS_PAIRS.items():
            sub_mask = np.isin(labeled_dataset, ds_pair)
            if sub_mask.sum() < 4 or len(set(labeled_dataset[sub_mask])) < 2:
                continue
            sil = silhouette_score(labeled_coords[sub_mask], labeled_dataset[sub_mask])
            per_fold_silhouette_rows.append({
                "true_class": true_class, "dataset_pair": " vs ".join(ds_pair),
                "silhouette": sil, "fold": fold,
            })

    purity_by_fold = pd.concat(per_fold_purity_rows, ignore_index=True)
    silhouette_by_fold = pd.DataFrame(per_fold_silhouette_rows)

    purity_summary = (
        purity_by_fold.groupby(["dataset", "true_class"])["purity"]
        .agg(mean_purity="mean", std_purity="std", n_folds="count")
        .reset_index()
        .sort_values("mean_purity", ascending=False)
    )
    silhouette_summary = (
        silhouette_by_fold.groupby(["true_class", "dataset_pair"])["silhouette"]
        .agg(mean_silhouette="mean", std_silhouette="std", n_folds="count")
        .reset_index()
    )

    return {
        "purity_by_fold": purity_by_fold,
        "purity_summary": purity_summary,
        "silhouette_by_fold": silhouette_by_fold,
        "silhouette_summary": silhouette_summary,
        "tsne_coords_by_fold": tsne_coords_by_fold,
    }


def plot_embedding_scatter_on_ax(
    ax,
    coords: np.ndarray,
    labels: pd.Series,
    title: str,
    palette: dict | None = None,
    show_legend: bool = True,
):
    """Scatter 2D t-SNE coordinates onto an existing matplotlib Axes, coloured by a categorical label."""
    unique_labels = sorted(labels.dropna().unique())
    cmap = plt.get_cmap("tab10")
    colors = palette or {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}

    for lab in unique_labels:
        mask = (labels == lab).to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.75, label=str(lab), color=colors.get(lab))

    nan_mask = labels.isna().to_numpy()
    if nan_mask.any():
        ax.scatter(coords[nan_mask, 0], coords[nan_mask, 1], s=14, alpha=0.75, label="N/A", color="grey")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.tick_params(labelsize=7)
    if show_legend:
        ax.legend(loc="best", fontsize=6, markerscale=1.0)
