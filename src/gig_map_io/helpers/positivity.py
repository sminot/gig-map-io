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
from .style import NEUTRAL, PRIMARY, QUALITATIVE, TEMPLATE, ZERO_LINE, group_colors


def plot_feature_positivity(
    rpkm: pd.DataFrame,
    grouping: pd.Series,
    threshold: float = 100,
    normalize: bool = False,
    height: int = 500,
    width: int = 800,
    file_prefix: str | None = None,
    legend_title: str | None = None,
    group_order: list | None = None,
) -> go.Figure:
    """
    Bar plot of samples by number of features exceeding a positivity threshold.

    For each value of N, the bar height is the number of samples in that group
    that have at least N features with a value >= ``threshold``.  This gives a
    cumulative view of how positivity accumulates across samples.

    Parameters
    ----------
    rpkm : pd.DataFrame
        Feature matrix. Rows are observations, columns are features.
        Index must overlap with ``grouping.index``.
    grouping : pd.Series
        Categorical group label for each observation.
        Index must overlap with ``rpkm.index``.
    threshold : float
        Minimum value a feature must reach to be counted as positive.
        Default: 100.
    normalize : bool
        If True, y-axis shows the fraction of each group's samples (0–1)
        rather than raw counts. Default: False.
    height : int
        Figure height in pixels. Default: 500.
    width : int
        Figure width in pixels. Default: 800.
    group_order : list, optional
        The groups in display order, the first being the case-like one; sets
        their colours.

    Returns
    -------
    plotly.graph_objects.Figure
        Grouped bar chart with N features on the x-axis, sample count on the
        y-axis, and one bar series per group.
    """
    shared_idx = rpkm.index.intersection(grouping.index)

    positive_counts = (rpkm.loc[shared_idx] >= threshold).sum(axis=1)
    groups = grouping.loc[shared_idx]

    max_n = int(positive_counts.max())

    group_sizes = groups.groupby(groups).size()

    rows = [
        {
            "n_features": n,
            "_group": group,
            "n_samples": int((positive_counts.loc[group_idx] >= n).sum()),
            "group_size": group_sizes[group],
        }
        for n in range(max_n + 1)
        for group, group_idx in groups.groupby(groups).groups.items()
    ]

    plot_df = pd.DataFrame(rows)

    if normalize:
        plot_df["n_samples"] = plot_df["n_samples"] / plot_df["group_size"]

    y_label = "Fraction of samples" if normalize else "Number of samples"
    tick_vals = list(range(max_n + 1))

    order = list(group_order) if group_order else sorted(groups.unique())
    bar_fig = px.bar(
        plot_df,
        x="n_features",
        y="n_samples",
        color="_group",
        color_discrete_map=group_colors(order),
        category_orders={"_group": order},
        barmode="group",
        labels={
            "n_features": "Minimum number of bins detected",
            "n_samples": y_label,
            "_group": legend_title or grouping.name or "Group",
        },
        template=TEMPLATE,
    )

    # --- Odds ratio subplot (two-group case only) --------------------------
    if groups.nunique() == 2:
        group_a, group_b = sorted(groups.unique())
        size_a = int((groups == group_a).sum())
        size_b = int((groups == group_b).sum())

        or_rows = []
        for n in range(max_n + 1):
            pos_a = int((positive_counts[groups == group_a] >= n).sum())
            pos_b = int((positive_counts[groups == group_b] >= n).sum())
            neg_a = size_a - pos_a
            neg_b = size_b - pos_b
            if pos_a > 0 and neg_a > 0 and pos_b > 0 and neg_b > 0:
                or_rows.append({"n_features": n, "or": (pos_a * neg_b) / (neg_a * pos_b)})

        or_df = pd.DataFrame(or_rows)

        or_df["or"] = np.log2(or_df["or"])

        if or_df["or"].mean() < 0:
            or_df["or"] = -or_df["or"]
            or_label = f"{group_b} vs. {group_a}"
        else:
            or_label = f"{group_a} vs. {group_b}"

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.08,
        )

        for trace in bar_fig.data:
            fig.add_trace(trace, row=1, col=1)

        fig.add_trace(
            go.Bar(
                x=or_df["n_features"], y=or_df["or"], name=or_label, showlegend=False,
                marker_color=NEUTRAL,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, row=2, col=1, **ZERO_LINE)

        fig.update_layout(
            barmode="group",
            template=TEMPLATE,
            height=height,
            width=width,
            legend=dict(
                title_text=legend_title or grouping.name or "Group",
                orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom",
            ),
            margin=dict(t=60),
        )
        fig.update_yaxes(title_text=y_label, row=1, col=1)
        fig.update_yaxes(title_text=f"log2 odds ratio<br>({or_label})", row=2, col=1)
        fig.update_xaxes(
            title_text="Minimum number of bins detected",
            tickmode="array",
            tickvals=tick_vals,
            row=2, col=1,
        )
    else:
        fig = bar_fig
        fig.update_layout(height=height, width=width)

    fig.update_xaxes(tickmode="array", tickvals=tick_vals)

    save_image(fig, file_prefix)

    return fig


def plot_positivity_heatmap(
    rpkm: pd.DataFrame,
    grouping: pd.Series | pd.DataFrame,
    threshold: float = 100,
    width: int = 1200,
    height: int = 800,
    file_prefix: str | None = None,
    show_sample_labels: bool = True,
    grouping_width: float = 0.2,
    group_orders: dict[str, list] | None = None,
) -> go.Figure:
    """
    Clustered positivity heatmap with grouping annotations and per-feature OR bars.

    Samples and features are ordered by hierarchical clustering. The figure
    contains three panels:

    - **Main heatmap** (top-left): binary positivity (feature value >= threshold)
      with features on the x-axis and samples on the y-axis.
    - **Grouping annotation** (right margin, sample axis): one column per
      grouping variable; each unique category is colour-coded with a legend.
    - **Log2 OR bars** (bottom margin, feature axis): for each (grouping
      variable, category), a vertical bar shows the log2 odds ratio of
      feature positivity for that category versus all other samples.

    Parameters
    ----------
    rpkm : pd.DataFrame
        Feature matrix. Rows are observations, columns are features.
        Index must overlap with ``grouping.index``.
    grouping : pd.Series or pd.DataFrame
        One or more categorical grouping variables. A Series is treated as a
        single variable. Index must overlap with ``rpkm.index``.
    threshold : float
        Minimum value for a feature to be counted as positive. Default: 100.
    width : int
        Figure width in pixels. Default: 1200.
    height : int
        Figure height in pixels. Default: 800.
    file_prefix : str or None
        If provided, saves the figure as .html, .pdf, and .png via
        ``save_image``. Default: None.
    group_orders : dict, optional
        For each grouping variable, its categories in display order, the
        first being the case-like one; sets their colours.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if isinstance(grouping, pd.Series):
        grouping = grouping.to_frame()
    group_orders = group_orders or {}

    shared_idx = rpkm.index.intersection(grouping.index)
    pos = (rpkm.loc[shared_idx] >= threshold).astype(int)
    meta = grouping.loc[shared_idx]

    # --- Cluster-sort samples and features ---------------------------------
    sample_order = linkage_order(pos.values)
    feature_order = linkage_order(pos.values.T)

    pos_sorted = pos.iloc[sample_order].iloc[:, feature_order]
    meta_sorted = meta.iloc[sample_order]

    sample_labels = [str(s) for s in pos_sorted.index]
    feature_labels = [str(f) for f in pos_sorted.columns]

    # Shared colour map: var_name → {category → colour}
    # Built once so OR bars and grouping heatmap use identical colours.
    def _categories(var_name: str) -> list:
        present = set(meta[var_name].dropna().unique())
        if var_name in group_orders:
            return [cat for cat in group_orders[var_name] if cat in present]
        return sorted(present)

    cat_colors: dict[str, dict] = {
        var_name: (
            {cat: color for cat, color in group_colors(group_orders[var_name]).items()}
            if var_name in group_orders
            else {cat: QUALITATIVE[i % len(QUALITATIVE)] for i, cat in enumerate(_categories(var_name))}
        )
        for var_name in meta.columns
    }

    # --- Log2 OR per grouping variable per feature -------------------------
    # For each variable, compute OR for every category vs. rest, then keep
    # only the category with the highest mean log2 OR across features.
    or_traces: list[go.Bar] = []

    for var_name in meta.columns:
        groups_var = meta[var_name]
        best_cat, best_log2_ors = None, None

        for cat in sorted(groups_var.dropna().unique()):
            in_cat = groups_var == cat
            out_cat = groups_var.notna() & (groups_var != cat)
            log2_ors = []
            for feat in pos_sorted.columns:
                col = pos[feat]
                pos_in  = int(col[in_cat].sum())
                neg_in  = int(in_cat.sum()) - pos_in
                pos_out = int(col[out_cat].sum())
                neg_out = int(out_cat.sum()) - pos_out
                if pos_in > 0 and neg_in > 0 and pos_out > 0 and neg_out > 0:
                    log2_ors.append(np.log2((pos_in * neg_out) / (neg_in * pos_out)))
                else:
                    log2_ors.append(0.0)
            if best_log2_ors is None or np.mean(log2_ors) > np.mean(best_log2_ors):
                best_cat, best_log2_ors = cat, log2_ors

        or_traces.append(go.Bar(
            x=feature_labels,
            y=best_log2_ors,
            name=f"{best_cat} vs. rest",
            marker_color=cat_colors[var_name][best_cat],
            marker_opacity=0.7,
            legendgroup="odds_ratio",
            legendgrouptitle_text="log2 odds ratio",
            showlegend=True,
        ))

    # --- Subplot grid ------------------------------------------------------
    # (1,1) OR bar chart  |  (1,2) [empty/None]
    # (2,1) main heatmap  |  (2,2) grouping annotation
    #
    # Axis mapping with None at (1,2):
    #   (1,1) → xaxis,  yaxis
    #   (2,1) → xaxis2, yaxis2
    #   (2,2) → xaxis3, yaxis3
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[1 - grouping_width, grouping_width],
        row_heights=[0.2, 0.8],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        specs=[
            [{"type": "bar"},     None               ],
            [{"type": "heatmap"}, {"type": "heatmap"}],
        ],
    )

    # (1,1) OR bars
    for trace in or_traces:
        fig.add_trace(trace, row=1, col=1)

    # (2,1) Main binary heatmap — z shape: (n_samples, n_features)
    fig.add_trace(
        go.Heatmap(
            z=pos_sorted.values,
            x=feature_labels,
            y=sample_labels,
            colorscale=[[0, "white"], [1, PRIMARY]],
            showscale=False,
            zmin=0, zmax=1,
            xgap=1, ygap=1,
        ),
        row=2, col=1,
    )

    # (2,2) Categorical grouping heatmap — one trace per variable
    legend_traces: list[go.Scatter] = []

    for var_name in meta.columns:
        col_data = meta_sorted[var_name]
        unique_cats = _categories(var_name)
        N = len(unique_cats)
        colors = [cat_colors[var_name][cat] for cat in unique_cats]

        # Encode each category as the midpoint of its colour band (avoids
        # boundary artefacts when building the discrete colorscale)
        cat_to_mid = {cat: i + 0.5 for i, cat in enumerate(unique_cats)}
        cs = []
        for i, color in enumerate(colors):
            cs.extend([[i / N, color], [(i + 1) / N, color]])

        fig.add_trace(
            go.Heatmap(
                z=[[cat_to_mid.get(v, np.nan)] for v in col_data],
                x=[str(var_name)],
                y=sample_labels,
                colorscale=cs,
                showscale=False,
                zmin=0, zmax=N,
            ),
            row=2, col=2,
        )

        for cat, color in zip(unique_cats, colors):
            legend_traces.append(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color, symbol='square'),
                name=str(cat),
                legendgroup=str(var_name),
                legendgrouptitle_text=str(var_name) if cat == unique_cats[0] else None,
                showlegend=True,
            ))

    for trace in legend_traces:
        fig.add_trace(trace)

    # --- Link shared axes and finalize -------------------------------------
    fig.update_layout(
        barmode='overlay',
        template=TEMPLATE,
        height=height,
        width=width,
        # Samples: link grouping annotation y-axis to main heatmap y-axis
        yaxis3=dict(matches='y2', showticklabels=False),
        # Features: link OR bar x-axis to main heatmap x-axis
        xaxis=dict(matches='x2', showticklabels=False),
        legend=dict(x=1.02, y=1.0, xanchor='left', yanchor='top'),
        margin=dict(t=40, b=150),
    )
    fig.update_xaxes(showticklabels=True, tickangle=-90, row=2, col=1)
    fig.update_yaxes(showticklabels=show_sample_labels, title_text="Samples", row=2, col=1)
    fig.update_yaxes(title_text="log2 odds ratio", row=1, col=1)
    fig.update_xaxes(showgrid=False, row=2, col=2)
    fig.update_yaxes(showgrid=False, row=2, col=2)

    save_image(fig, file_prefix)

    return fig
