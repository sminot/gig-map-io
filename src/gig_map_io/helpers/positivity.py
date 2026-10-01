"""
Plots summarizing per-sample detection of features above a threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .clustering import linkage_order
from .save_image import save_image
from .sidebar import add_category_sidebar
from .style import NEUTRAL, PRIMARY, QUALITATIVE, TEMPLATE, TOP_LEGEND, ZERO_LINE, group_colors


def log2_odds(pos_a: int, neg_a: int, pos_b: int, neg_b: int) -> float | None:
    """
    The log2 odds of being positive in group A against group B, or ``None``
    when any cell of the 2x2 table is empty and the ratio is undefined.
    """
    if min(pos_a, neg_a, pos_b, neg_b) == 0:
        return None
    return float(np.log2((pos_a * neg_b) / (neg_a * pos_b)))


def plot_feature_positivity(
    rpkm: pd.DataFrame,
    grouping: pd.Series,
    group_order: list,
    threshold: float = 100,
    normalize: bool = False,
    width: int = 800,
    height: int = 500,
    file_prefix: str | None = None,
    legend_title: str | None = None,
) -> go.Figure:
    """
    Samples by how many features they carry, for two groups.

    For each cutoff N, the top panel gives how many samples (or what fraction,
    with ``normalize``) of each group have at least N features at or above
    ``threshold``; the bottom panel gives the log2 odds of reaching that cutoff
    in the first group of ``group_order`` against the second.
    """
    if len(group_order) != 2:
        raise ValueError("plot_feature_positivity compares exactly two groups")
    shared_idx = rpkm.index.intersection(grouping.index)
    positive_counts = (rpkm.loc[shared_idx] >= threshold).sum(axis=1)
    groups = grouping.loc[shared_idx]
    max_n = int(positive_counts.max())
    group_sizes = groups.groupby(groups).size()

    plot_df = pd.DataFrame([
        {
            "n_features": n,
            "_group": group,
            "n_samples": int((positive_counts.loc[group_idx] >= n).sum()),
            "group_size": group_sizes[group],
        }
        for n in range(max_n + 1)
        for group, group_idx in groups.groupby(groups).groups.items()
    ])
    if normalize:
        plot_df["n_samples"] = plot_df["n_samples"] / plot_df["group_size"]
    y_label = "Fraction of samples" if normalize else "Number of samples"
    legend_title = legend_title or grouping.name or "Group"

    bars = px.bar(
        plot_df,
        x="n_features",
        y="n_samples",
        color="_group",
        color_discrete_map=group_colors(group_order),
        category_orders={"_group": list(group_order)},
        barmode="group",
        labels={"n_features": "Minimum number of bins detected", "n_samples": y_label, "_group": legend_title},
        template=TEMPLATE,
    )

    group_a, group_b = group_order
    odds = []
    for n in range(max_n + 1):
        pos_a = int((positive_counts[groups == group_a] >= n).sum())
        pos_b = int((positive_counts[groups == group_b] >= n).sum())
        value = log2_odds(pos_a, group_sizes[group_a] - pos_a, pos_b, group_sizes[group_b] - pos_b)
        if value is not None:
            odds.append({"n_features": n, "log2_odds": value})
    odds = pd.DataFrame(odds)
    or_label = f"{group_a} vs. {group_b}"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)
    for trace in bars.data:
        fig.add_trace(trace, row=1, col=1)
    fig.add_trace(
        go.Bar(x=odds["n_features"], y=odds["log2_odds"], name=or_label, showlegend=False, marker_color=NEUTRAL),
        row=2, col=1,
    )
    fig.add_hline(y=0, row=2, col=1, **ZERO_LINE)
    fig.update_layout(
        barmode="group",
        template=TEMPLATE,
        height=height,
        width=width,
        legend=dict(TOP_LEGEND, title_text=legend_title),
        margin=dict(t=60),
    )
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_yaxes(title_text=f"log2 odds ratio<br>({or_label})", row=2, col=1)
    fig.update_xaxes(tickmode="array", tickvals=list(range(max_n + 1)))
    fig.update_xaxes(title_text="Minimum number of bins detected", row=2, col=1)
    save_image(fig, file_prefix)
    return fig


def plot_positivity_heatmap(
    rpkm: pd.DataFrame,
    grouping: pd.DataFrame,
    group_orders: dict[str, list],
    threshold: float = 100,
    width: int = 1200,
    height: int = 800,
    file_prefix: str | None = None,
    show_sample_labels: bool = True,
) -> go.Figure:
    """
    Which features each sample carries, with the samples annotated.

    Three panels: the binary heatmap of features (columns) at or above
    ``threshold`` in each sample (rows), both axes ordered by hierarchical
    clustering; one column of colour per grouping variable beside it; and
    above it, for each grouping variable, the log2 odds of carrying each
    feature in the category that is most enriched for the features against
    the rest of the samples.

    ``group_orders`` gives each grouping variable's categories in display
    order, the first being the case-like one, which sets their colours.
    """
    shared_idx = rpkm.index.intersection(grouping.index)
    positive = (rpkm.loc[shared_idx] >= threshold).astype(int)
    meta = grouping.loc[shared_idx]

    positive = positive.iloc[linkage_order(positive.values), linkage_order(positive.values.T)]
    meta = meta.loc[positive.index]
    feature_labels = [str(f) for f in positive.columns]

    categories = {}
    colors = {}
    for var_name in meta.columns:
        present = set(meta[var_name].dropna().unique())
        if var_name in group_orders:
            categories[var_name] = [c for c in group_orders[var_name] if c in present]
            colors[var_name] = group_colors(group_orders[var_name])
        else:
            categories[var_name] = sorted(present)
            colors[var_name] = dict(zip(categories[var_name], QUALITATIVE))

    # (1,1) odds-ratio bars over (2,1) the heatmap, with (2,2) the sidebar
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        specs=[[{"type": "bar"}, None], [{"type": "heatmap"}, {"type": "heatmap"}]],
    )

    # For each variable, the category whose members carry the features most
    for var_name in meta.columns:
        best = None
        for category in categories[var_name]:
            in_cat = meta[var_name] == category
            out_cat = meta[var_name].notna() & ~in_cat
            odds = [
                log2_odds(int(positive.loc[in_cat, f].sum()), int(in_cat.sum() - positive.loc[in_cat, f].sum()),
                          int(positive.loc[out_cat, f].sum()), int(out_cat.sum() - positive.loc[out_cat, f].sum()))
                for f in positive.columns
            ]
            odds = [0.0 if value is None else value for value in odds]
            if best is None or np.mean(odds) > np.mean(best[1]):
                best = (category, odds)
        fig.add_trace(go.Bar(
            x=feature_labels, y=best[1], name=f"{best[0]} vs. rest",
            marker_color=colors[var_name][best[0]], marker_opacity=0.7,
            legendgroup="odds_ratio", legendgrouptitle_text="log2 odds ratio", showlegend=True,
        ), row=1, col=1)

    fig.add_trace(
        go.Heatmap(
            z=positive.values,
            x=feature_labels,
            y=[str(s) for s in positive.index],
            colorscale=[[0, "white"], [1, PRIMARY]],
            showscale=False,
            zmin=0, zmax=1,
            xgap=1, ygap=1,
        ),
        row=2, col=1,
    )
    for var_name in meta.columns:
        add_category_sidebar(
            fig, meta[var_name].set_axis([str(s) for s in positive.index]), categories[var_name], colors[var_name],
            row=2, col=2, x=str(var_name), legend_group=str(var_name),
        )

    fig.update_layout(
        barmode="overlay",
        template=TEMPLATE,
        height=height,
        width=width,
        yaxis3=dict(matches="y2", showticklabels=False),
        xaxis=dict(matches="x2", showticklabels=False),
        legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
        margin=dict(t=40, b=150),
    )
    fig.update_xaxes(showticklabels=True, tickangle=-90, row=2, col=1)
    fig.update_yaxes(showticklabels=show_sample_labels, title_text="Samples", row=2, col=1)
    fig.update_yaxes(title_text="log2 odds ratio", row=1, col=1)
    fig.update_xaxes(showgrid=False, row=2, col=2)
    fig.update_yaxes(showgrid=False, row=2, col=2)
    save_image(fig, file_prefix)
    return fig
