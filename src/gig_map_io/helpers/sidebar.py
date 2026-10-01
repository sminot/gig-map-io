"""
A column of colour beside a heatmap, one cell per sample, showing which
category each sample belongs to.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def add_category_sidebar(
    fig: go.Figure,
    values: pd.Series,
    categories: Sequence,
    colors: Dict,
    row: int,
    col: int,
    x,
    legend_group: str,
) -> None:
    """
    Draw ``values`` (one per sample, in the order the heatmap's rows are
    drawn) as a single-column heatmap at ``x``, each category in its colour,
    and add one legend entry per category under the heading ``legend_group``.
    """
    # Each category takes one band of a stepped colour scale, with its code
    # centred in the band so that no cell falls on a boundary
    n = len(categories)
    code = {category: i for i, category in enumerate(categories)}
    scale = []
    for i, category in enumerate(categories):
        scale.extend([[i / n, colors[category]], [(i + 1) / n, colors[category]]])
    fig.add_trace(
        go.Heatmap(
            z=[[code.get(v, np.nan)] for v in values],
            x=[x],
            y=list(values.index),
            colorscale=scale,
            zmin=-0.5,
            zmax=n - 0.5,
            showscale=False,
            name=legend_group,
            hovertemplate="%{y}: %{text}<extra></extra>",
            text=[[str(v)] for v in values],
        ),
        row=row,
        col=col,
    )
    for category in categories:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=colors[category], symbol="square", size=10),
                name=str(category), legendgroup=legend_group,
                legendgrouptitle_text=legend_group if category == categories[0] else None,
                showlegend=True,
            ),
            row=row,
            col=col,
        )
