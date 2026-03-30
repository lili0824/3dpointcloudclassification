from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


_NORMALIZED_SUFFIX = '_normalized_npy'


def _flake_id_candidates(file_name_or_id: str) -> list[str]:
    raw_flake_id = os.path.basename(file_name_or_id)
    if raw_flake_id.lower().endswith('.npy'):
        raw_flake_id = raw_flake_id[:-4]
    candidates = [raw_flake_id]
    current = raw_flake_id
    while current.endswith(_NORMALIZED_SUFFIX):
        current = current[: -len(_NORMALIZED_SUFFIX)]
        if current not in candidates:
            candidates.append(current)
    return candidates


def _resolve_critical_path(critical_points_dir: str, raw_flake_id: str):
    for candidate in _flake_id_candidates(raw_flake_id):
        crit_path = os.path.join(critical_points_dir, f'{candidate}_critical.npy')
        if os.path.exists(crit_path):
            return candidate, crit_path
    return None, None


def _resolve_prediction_info(predictions: dict, raw_flake_id: str) -> dict:
    for candidate in _flake_id_candidates(raw_flake_id):
        if candidate in predictions:
            return predictions[candidate]
    return {}


def build_critical_overlay_figure(
    test_name: str,
    ensemble_predictions_csv: str,
    source_npy_dir: str,
    critical_points_dir: str,
):
    ensemble_df = pd.read_csv(ensemble_predictions_csv)
    id_col = 'file_id' if 'file_id' in ensemble_df.columns else 'flake_id'
    predictions = {
        row[id_col]: {
            'pred_name': row.get('pred_class', 'Unknown'),
            'confidence': row.get('confidence', 0.0),
            'correct': row.get('correct', None),
        }
        for _, row in ensemble_df.iterrows()
    }

    npy_files = sorted([f for f in os.listdir(source_npy_dir) if f.lower().endswith('.npy')])
    figs_data = []

    used_ids = set()
    for npy_file in npy_files:
        raw_flake_id = os.path.splitext(npy_file)[0]
        flake_id, crit_path = _resolve_critical_path(critical_points_dir, raw_flake_id)
        if flake_id is None or crit_path is None:
            continue
        if flake_id in used_ids:
            continue

        sampled_pts = np.load(os.path.join(source_npy_dir, npy_file))
        critical_pts = np.load(crit_path)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=sampled_pts[:, 0],
                y=sampled_pts[:, 1],
                z=sampled_pts[:, 2],
                mode='markers',
                marker=dict(size=3, color='#505050', opacity=0.6),
                name='Sampled Points',
                hoverinfo='skip',
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=critical_pts[:, 0],
                y=critical_pts[:, 1],
                z=critical_pts[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=critical_pts[:, 3],
                    colorscale='Hot_r',
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title='Saliency', len=0.5, thickness=15),
                ),
                name='Critical Points',
                hovertemplate='<b>Critical Point</b><br>Saliency: %{marker.color:.3f}<extra></extra>',
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                bgcolor='white',
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            width=600,
            height=600,
        )

        pred_info = _resolve_prediction_info(predictions, raw_flake_id)
        figs_data.append((flake_id, fig, pred_info))
        used_ids.add(flake_id)

    if not figs_data:
        return None, 0

    cols = 3
    rows = (len(figs_data) + cols - 1) // cols
    specs = [[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)]

    default_vertical_spacing = 0.08
    if rows <= 1:
        vertical_spacing = default_vertical_spacing
    else:
        max_vertical_spacing = max(0.001, (1.0 / (rows - 1)) - 1e-6)
        vertical_spacing = min(default_vertical_spacing, max_vertical_spacing)

    default_horizontal_spacing = 0.12
    if cols <= 1:
        horizontal_spacing = default_horizontal_spacing
    else:
        max_horizontal_spacing = max(0.001, (1.0 / (cols - 1)) - 1e-6)
        horizontal_spacing = min(default_horizontal_spacing, max_horizontal_spacing)

    fig_grid = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        column_widths=[0.33, 0.33, 0.33],
    )

    colorbar_added = False
    for idx, (_, fig, _) in enumerate(figs_data):
        row, col = idx // cols + 1, idx % cols + 1
        for trace in fig.data:
            trace_copy = go.Scatter3d(trace)
            if trace.name == 'Critical Points' and not colorbar_added:
                trace_copy.marker.colorbar = dict(
                    title='Saliency', orientation='h', thickness=15, len=0.15, x=0.5, y=1.05
                )
                colorbar_added = True
            else:
                trace_copy.marker.colorbar = None
            fig_grid.add_trace(trace_copy, row=row, col=col)

    annotations = []
    for idx, (flake_id, _, pred_info) in enumerate(figs_data):
        row, col = idx // cols + 1, idx % cols + 1
        x_end = col / cols - 0.01
        y_top = 1 - ((row - 1) / rows) - 0.01

        pred_name = pred_info.get('pred_name', flake_id)
        confidence = pred_info.get('confidence', 0.0)
        correct = pred_info.get('correct', None)
        status = '✓' if str(correct).lower() in ['true', '1'] else '✗' if correct is not None else ''
        text = (
            f'{flake_id}<br>{status} {pred_name} ({confidence:.1%})'
            if status
            else f'{flake_id}<br>{pred_name} ({confidence:.1%})'
        )

        annotations.append(
            dict(
                text=text,
                xref='paper',
                yref='paper',
                x=x_end,
                y=y_top,
                xanchor='right',
                yanchor='top',
                showarrow=False,
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.9)',
                borderpad=4,
            )
        )

    fig_grid.update_layout(
        title=f'Critical Points - {test_name}',
        showlegend=False,
        annotations=annotations,
        margin=dict(l=40, r=40, t=80, b=40),
        height=rows * 600 + 100,
        autosize=True,
    )

    for i in range(1, len(figs_data) + 1):
        row, col = (i - 1) // cols + 1, (i - 1) % cols + 1
        x_min, x_max = (col - 1) / cols, col / cols
        y_min, y_max = 1 - (row / rows), 1 - ((row - 1) / rows)
        x_span, y_span = (x_max - x_min) * 0.96, (y_max - y_min) * 0.95
        scene_name = f'scene{i}' if i > 1 else 'scene'
        fig_grid.update_layout(
            **{
                scene_name: dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectmode='data',
                    bgcolor='white',
                    domain=dict(
                        x=[
                            x_min + (x_max - x_min) / 2 - x_span / 2,
                            x_min + (x_max - x_min) / 2 + x_span / 2,
                        ],
                        y=[
                            y_min + (y_max - y_min) / 2 - y_span / 2,
                            y_min + (y_max - y_min) / 2 + y_span / 2,
                        ],
                    ),
                )
            }
        )

    return fig_grid, len(figs_data)


def plot_rank_curves(test_name: str, flake_curves, percentiles: Sequence[int], output_dir: str):
    if not flake_curves:
        return None

    cols, rows = 4, math.ceil(len(flake_curves) / 4)
    specs = [[{'type': 'xy'} for _ in range(cols)] for _ in range(rows)]

    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=[item['flake_id'] for item in flake_curves])

    for i, item in enumerate(flake_curves):
        row, col = (i // cols) + 1, (i % cols) + 1
        fig.add_trace(
            go.Scatter(x=percentiles, y=item['curve'], mode='lines+markers', line=dict(color='blue'), showlegend=False),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=max(500, rows * 320),
        title_text=f'Saliency Rank Curves - {test_name}',
        showlegend=False,
        autosize=True,
        margin=dict(l=40, r=40, t=70, b=40),
    )

    for i in range(1, rows * cols + 1):
        fig.update_xaxes(title_text='Top Percentage (%)', row=(i - 1) // cols + 1, col=(i - 1) % cols + 1)
        fig.update_yaxes(title_text='Cumulative Saliency', range=[0, 1], row=(i - 1) // cols + 1, col=(i - 1) % cols + 1)

    html_path = os.path.join(output_dir, f'{test_name}_saliency_rank_curves.html')
    pio.write_html(
        fig,
        file=html_path,
        auto_open=False,
        config={'responsive': True},
        include_plotlyjs='cdn',
        full_html=True,
        default_width='100%',
    )
    return html_path
