import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "val_balanced_acc",
    "val_f1",
    "val_class_acc_Discoide",
    "val_class_acc_Levallois",
    "val_class_acc_Laminaire",
)

CANONICAL_CLASSES = ("Discoidal", "Levallois", "Laminar")
INDEX_TO_CLASS = {0: "Discoidal", 1: "Levallois", 2: "Laminar"}
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


def resolve_ensemble_path(root_dir: str, dataset_key: str) -> str | None:
    candidates = [
        os.path.join(root_dir, dataset_key, "ensemble_predictions.csv"),
        os.path.join(root_dir, "independent_test_batches", dataset_key, "ensemble_predictions.csv"),
        os.path.join(root_dir, "independent_test_results_modelnet", "ensemble_predictions.csv") if dataset_key == "ModelNet" else None,
        os.path.join(root_dir, "independent_test_modelnet", "ensemble_predictions.csv") if dataset_key == "ModelNet" else None,
    ]
    candidates = [path for path in candidates if path]
    for path in candidates:
        if os.path.isfile(path):
            return path

    hits = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "ensemble_predictions.csv" in filenames and dataset_key.lower() in dirpath.lower():
            hits.append(os.path.join(dirpath, "ensemble_predictions.csv"))
    return max(hits, key=os.path.getmtime) if hits else None


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


def _final_metric_values(metrics_by_fold: Dict[int, pd.DataFrame], column: str) -> List[float]:
    values: List[float] = []
    for df in metrics_by_fold.values():
        series = df[column].dropna() if column in df.columns else pd.Series(dtype=float)
        if not series.empty:
            values.append(float(series.iloc[-1]))
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
                "PointNet++ (mean ± SD)": _format_mean_std(_final_metric_values(pn_metrics, column)),
                "DGCNN (mean ± SD)": _format_mean_std(_final_metric_values(dg_metrics, column)),
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
    pn_acc_train, pn_acc_val = _prepare_metric_stats(pn_metrics, "train_acc", "val_acc")
    dg_loss_train, dg_loss_val = _prepare_metric_stats(dg_metrics, "train_loss", "val_loss")
    dg_acc_train, dg_acc_val = _prepare_metric_stats(dg_metrics, "train_acc", "val_acc")

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="col")

    _plot_with_band(axes[0, 0], pn_loss_train, train_color, "Train")
    _plot_with_band(axes[0, 0], pn_loss_val, val_color, "Validation")
    axes[0, 0].set_title("PointNet++ Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3, linestyle="--")
    axes[0, 0].legend(loc="upper right")

    _plot_with_band(axes[0, 1], pn_acc_train, train_color, "Train")
    _plot_with_band(axes[0, 1], pn_acc_val, val_color, "Validation")
    axes[0, 1].set_title("PointNet++ Accuracy")
    axes[0, 1].set_ylabel("Accuracy")
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
    axes[1, 1].set_title("DGCNN Accuracy")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(alpha=0.3, linestyle="--")
    axes[1, 1].legend(loc="lower right")

    for ax, label in zip(axes.flatten(), ["a", "b", "c", "d"]):
        _label_panel(ax, label)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved Figure 3 to {save_path}")

    return pn_metrics, dg_metrics


def parse_prediction_row(row: pd.Series) -> tuple[int, list[float]]:
    pred_class_str = str(row.get("pred_class", "0")).strip().lower()
    if pred_class_str in ["discoide", "discoidal", "0"]:
        pred_class_idx = 0
    elif pred_class_str in ["levallois", "1"]:
        pred_class_idx = 1
    elif pred_class_str in ["laminaire", "laminar", "2"]:
        pred_class_idx = 2
    else:
        pred_class_idx = 0

    discoide_prob = float(row.get("discoide_probability", 0))
    levallois_prob = float(row.get("levallois_probability", 0))
    laminaire_prob = float(row.get("laminaire_probability", 0))

    total = discoide_prob + levallois_prob + laminaire_prob
    if total > 0:
        discoide_prob /= total
        levallois_prob /= total
        laminaire_prob /= total

    return pred_class_idx, [discoide_prob, levallois_prob, laminaire_prob]


def load_predictions_by_group(model_roots: dict[str, str]) -> dict:
    datasets_by_group = {group_idx: {} for group_idx in DATASET_FIGURE_GROUPS}

    for dataset_key in DATASET_TO_GROUP:
        group = DATASET_TO_GROUP[dataset_key]
        datasets_by_group[group][dataset_key] = {}

        for model in MODELS:
            ens_path = resolve_ensemble_path(model_roots[model], dataset_key)
            if not ens_path:
                datasets_by_group[group][dataset_key][model] = []
                continue

            ens_df = pd.read_csv(ens_path)
            ens_df.columns = ens_df.columns.str.lower().str.replace(" ", "_")
            datasets_by_group[group][dataset_key][model] = [
                parse_prediction_row(row) for _, row in ens_df.iterrows()
            ]

    return datasets_by_group


def plot_datasets_grid(datasets_data, datasets_dict, models_list, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize)
    n_datasets = len(datasets_data)
    n_models = len(models_list)

    axes_grid = []
    panel_labels = []
    for row_idx in range(n_datasets):
        row_axes = []
        for col_idx in range(n_models):
            ax_idx = row_idx * n_models + col_idx + 1
            ax = fig.add_subplot(n_datasets, n_models, ax_idx, projection="ternary")
            row_axes.append(ax)
            panel_labels.append(chr(97 + row_idx * n_models + col_idx))
        axes_grid.append(row_axes)

    panel_idx = 0
    for row_idx, dataset_key in enumerate(datasets_data):
        dataset_name = DATASET_DISPLAY.get(dataset_key, dataset_key)

        row_box = axes_grid[row_idx][0].get_position()
        y_pos = (row_box.y0 + row_box.y1) / 2
        fig.text(0.12, y_pos, dataset_name, rotation=90, va="center", ha="left", fontsize=10, fontweight="bold")

        for col_idx, model in enumerate(models_list):
            ax = axes_grid[row_idx][col_idx]
            predictions = datasets_dict.get(dataset_key, {}).get(model, [])

            if not predictions:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                panel_idx += 1
                continue

            t_vals, l_vals, r_vals, point_colors = [], [], [], []
            for true_class_idx, pred_dist in predictions:
                t_vals.append(pred_dist[0])
                l_vals.append(pred_dist[1])
                r_vals.append(pred_dist[2])
                point_colors.append(COLORS.get(true_class_idx, "#808080"))

            for t, l, r, col in zip(t_vals, l_vals, r_vals, point_colors):
                ax.scatter([t], [l], [r], color=col, s=24, alpha=0.6)

            t_mean, l_mean, r_mean = float(np.mean(t_vals)), float(np.mean(l_vals)), float(np.mean(r_vals))
            mean_total = t_mean + l_mean + r_mean
            if mean_total > 0:
                t_mean, l_mean, r_mean = t_mean / mean_total, l_mean / mean_total, r_mean / mean_total
            ax.scatter([t_mean], [l_mean], [r_mean], color="#808080", marker="x", s=85, linewidths=2, zorder=10)

            ax.set_tlabel("Discoidal", fontsize=9, labelpad=6)
            ax.set_llabel("Levallois", fontsize=9, labelpad=6)
            ax.set_rlabel("Laminar", fontsize=9, labelpad=6)
            ax.grid(True, alpha=0.3)
            ax.text(0.05, 0.95, panel_labels[panel_idx], transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
            panel_idx += 1

    for col_idx, model in enumerate(models_list):
        fig.text(0.34 + col_idx * 0.42, 0.98, model, ha="center", fontsize=11)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", label="Discoidal", markerfacecolor=COLORS[0], markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label="Levallois", markerfacecolor=COLORS[1], markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label="Laminar", markerfacecolor=COLORS[2], markersize=6),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=False, fontsize=9)
    plt.subplots_adjust(left=0.20, right=0.91, top=0.89, bottom=0.12, wspace=0.25, hspace=0.44)
    return fig


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
