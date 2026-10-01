"""
Enrichment of categories among a set of bins, drawn as paired bar panels.

Shared by the per-study organism and annotation enrichment figures and by
the candidate-bin annotation figure, so that every enrichment result in an
analysis reads the same way.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .save_image import save_image
from .style import CASE_COLOR, CONTROL_COLOR, TEMPLATE, ZERO_LINE, group_colors

#: How the significant bins are split by direction of effect, and the color
#: of each side: bins higher in cases take the case color
ASSOCIATION_COLORS = {
    "Positively associated": CASE_COLOR,
    "Negatively associated": CONTROL_COLOR,
}


def enrich_organisms(
    foreground: Iterable[tuple],
    universe: Iterable[tuple],
    organisms: Sequence[str],
    alternative: str = "greater",
) -> pd.DataFrame:
    """
    Test whether each organism contributed more (pangenome, bin) pairs to
    ``foreground`` than its share of ``universe``, the pairs that could have
    been in it. Fisher's exact test per organism, FDR-corrected across the
    organisms given.

    Returns one row per organism: its bins in the foreground and in the
    background, the totals, the odds ratio, and the p- and q-values.
    """
    from collections import Counter

    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    foreground, universe = set(foreground), set(universe)
    untested = foreground - universe
    if untested:
        raise ValueError(
            f"{len(untested)} of the foreground bins are not in the universe, "
            f"e.g. {sorted(untested)[0]}"
        )
    foreground_counts = Counter(organism for organism, _ in foreground)
    universe_counts = Counter(organism for organism, _ in universe)
    n_foreground = len(foreground)
    n_background = len(universe) - n_foreground

    rows = []
    for organism in organisms:
        in_foreground = foreground_counts.get(organism, 0)
        in_background = universe_counts.get(organism, 0) - in_foreground
        odds_ratio, pvalue = stats.fisher_exact(
            [
                [in_foreground, in_background],
                [n_foreground - in_foreground, n_background - in_background],
            ],
            alternative=alternative,
        )
        rows.append({
            "organism": organism,
            "n_foreground": in_foreground,
            "n_background": in_background,
            "n_foreground_total": n_foreground,
            "n_background_total": n_background,
            "odds_ratio": odds_ratio,
            "pvalue": pvalue,
        })
    df = pd.DataFrame(rows)
    df["qvalue"] = multipletests(df["pvalue"], method="fdr_bh")[1]
    # Organisms contributing no bins all share a p-value of 1
    return df.sort_values(["pvalue", "organism"]).reset_index(drop=True)


def plot_enrichment(
    enrichment: pd.DataFrame,
    label: str,
    axis_title: str,
    title: str,
    qvalue_threshold: float,
    order: List[str] | None,
    width: int,
    height: int | None,
    file_prefix: str | None,
    mark: str = "star",
    show_legend: bool = True,
) -> go.Figure:
    """
    Two panels sharing a category axis: how many bins of each category fell in
    each group, and how enriched that is. Bars are grouped by the set the bins
    came from, so the two directions of effect can be read against each other.

    Odds ratios are shown on a log2 scale, which is symmetric about no
    enrichment, and marked where they clear the q-value threshold. A category
    with no bins at all in a group has nothing to say about enrichment, so no
    bar is drawn for it; one whose bins are *all* in the foreground has an
    infinite odds ratio and is drawn at the edge of the observed range.

    When ``height`` is ``None`` it follows the number of categories, so that a
    figure of three terms is not drawn on the same canvas as one of twenty.

    ``show_legend`` is off for a figure of one set whose title already names it.

    ``mark`` is what is printed beside each odds-ratio bar: a star where the
    q-value clears the threshold (``"star"``), or the q-value itself
    (``"qvalue"``), for a figure whose terms all clear it already.
    """
    df = enrichment.copy()
    for column in ["odds_ratio", "qvalue", "n_foreground", "n_background"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    log2_odds = np.log2(df["odds_ratio"].replace({0: np.nan, np.inf: np.nan}))
    limit = float(np.nanmax(np.abs(log2_odds))) if log2_odds.notna().any() else 1.0

    unbounded = np.isinf(df["odds_ratio"]) & (df["n_foreground"] > 0)
    df["log2_odds_ratio"] = log2_odds.where(~unbounded, limit)
    # Nothing observed means no evidence either way, rather than depletion
    df.loc[df["n_foreground"] == 0, "log2_odds_ratio"] = np.nan
    if mark == "star":
        df["mark"] = np.where(df["qvalue"] < qvalue_threshold, "*", "")
        odds_title = f"log2 odds ratio (* q < {qvalue_threshold})"
    elif mark == "qvalue":
        df["mark"] = "q = " + df["qvalue"].map("{:.2g}".format)
        odds_title = "log2 odds ratio"
    else:
        raise ValueError("mark must be 'star' or 'qvalue'")

    groups = list(dict.fromkeys(enrichment["group"]))
    category_orders = {"group": groups}
    if order is not None:
        category_orders[label] = order
    n_categories = len(order) if order is not None else df[label].nunique()
    colors = {**group_colors(groups), **ASSOCIATION_COLORS}

    hover = {"qvalue": ":.2e", "odds_ratio": ":.3g", "n_foreground": True, "n_background": True}
    shared = dict(
        data_frame=df, y=label, color="group", orientation="h", barmode="group",
        template=TEMPLATE, hover_data=hover, category_orders=category_orders,
        color_discrete_map=colors,
    )

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06)
    for trace in px.bar(x="n_foreground", **shared).data:
        fig.add_trace(trace, row=1, col=1)
    for trace in px.bar(x="log2_odds_ratio", text="mark", **shared).data:
        fig.add_trace(
            trace.update(showlegend=False, textposition="outside", textfont_size=11),
            row=1, col=2,
        )

    fig.add_vline(x=0, row=1, col=2, **ZERO_LINE)
    # px applied category_orders to its own figure; the traces copied from it
    # carry no axis order, so it is set again here, first category at the top
    if order is not None:
        fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(order)))
    if mark == "qvalue":
        upper = float(np.nanmax(df["log2_odds_ratio"])) if df["log2_odds_ratio"].notna().any() else 1.0
        fig.update_xaxes(range=[min(0.0, float(np.nanmin(df["log2_odds_ratio"]))), upper * 1.45], row=1, col=2)

    fig.update_layout(
        width=width,
        height=height if height is not None else max(320, 24 * n_categories + 190),
        title=title,
        template=TEMPLATE,
        barmode="group",
        showlegend=show_legend,
        legend=dict(
            title_text="", orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom",
        ),
        margin=dict(t=95 if show_legend else 70),
        xaxis=dict(title="Bins in set"),
        xaxis2=dict(title=odds_title),
        yaxis=dict(title=axis_title, automargin=True),
    )

    # An empty panel is indistinguishable from a broken one, so say which it
    # is, and do not draw axes around nothing
    if df.empty or not (df["n_foreground"] > 0).any():
        fig.add_annotation(
            text="No bins met the significance threshold",
            showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(color="gray"),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(showlegend=False)

    save_image(fig, file_prefix)
    return fig
