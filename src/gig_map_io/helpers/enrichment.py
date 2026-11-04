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
from .style import CASE_COLOR, CONTROL_COLOR, LEFT_TOP_LEGEND, TEMPLATE, ZERO_LINE, group_colors

#: The first word of the label of each set the significant bins are split
#: into by direction of effect, and its color: bins higher in cases take the
#: case color, whatever the cases are called
ASSOCIATION_COLORS = {"Higher": CASE_COLOR, "Lower": CONTROL_COLOR}

#: Sizing: the two panels and their margins, the width one character of a
#: category name takes at the template's 13 px, and the bounds on the width;
#: then the height of one category's bars, the shortest data region, and the
#: height of a figure that has nothing to draw
PANELS_PX = 600
LABEL_CHAR_PX = 6.5
MIN_WIDTH = 560
MAX_WIDTH = 1000
CATEGORY_PX = 26
MIN_PLOT_PX = 110
EMPTY_HEIGHT = 140
#: Room beneath the plot for the axis titles, and for each extra line of one
BOTTOM_PX = 60
TITLE_LINE_PX = 16
#: A category name longer than this breaks over two lines, so that a long
#: annotation term does not take the width the bars need in a printed panel;
#: every category then takes the height two lines need, since plotly spaces
#: categories evenly and two lines in the room of one overlap their neighbors
WRAP_CHARS = 24
WRAPPED_CATEGORY_PX = 40
#: Room right of the plot for a q-value printed beyond the last bar, and the
#: gap between a bar and its q-value
MARK_MARGIN_PX = 70
MARK_GAP_PX = 3


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


def _count_dtick(largest: int) -> int | None:
    """Tick spacing for an axis counting bins up to ``largest``: plotly's own past ten."""
    if largest <= 5:
        return 1
    return 2 if largest <= 10 else None


def plot_enrichment(
    enrichment: pd.DataFrame,
    label: str,
    axis_title: str,
    qvalue_threshold: float,
    order: List[str] | None,
    width: int | None,
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
    figure of three terms is not drawn on the same canvas as one of twenty;
    when ``width`` is ``None`` it follows the longest category name, which
    the two panels would otherwise have to give up room to.

    ``show_legend`` is off for a figure of one set, where a legend would name nothing.

    ``mark`` is what is printed beside each odds-ratio bar: a star where the
    q-value clears the threshold (``"star"``), or the q-value itself
    (``"qvalue"``), for a figure whose terms all clear it already.
    """
    df = enrichment.copy()
    df[label] = df[label].map(_wrap_label)
    if order is not None:
        order = [_wrap_label(name) for name in order]
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
        # Two lines, here and for the q-values: on one it runs into the count
        # axis's title in a narrow panel
        odds_title = f"log2 odds ratio<br>(* q < {qvalue_threshold})"
    elif mark == "qvalue":
        df["mark"] = "q = " + df["qvalue"].map("{:.2g}".format)
        odds_title = "log2<br>odds ratio"
    else:
        raise ValueError("mark must be 'star' or 'qvalue'")

    groups = list(dict.fromkeys(enrichment["group"]))
    category_orders = {"group": groups}
    if order is not None:
        category_orders[label] = order
    n_categories = len(order) if order is not None else df[label].nunique()
    colors = {
        **group_colors(groups),
        **{group: ASSOCIATION_COLORS[group.split()[0]] for group in groups if group.split()[0] in ASSOCIATION_COLORS},
    }

    hover = {"qvalue": ":.2e", "odds_ratio": ":.3g", "n_foreground": True, "n_background": True}
    shared = dict(
        data_frame=df, y=label, color="group", orientation="h", barmode="group",
        template=TEMPLATE, hover_data=hover, category_orders=category_orders,
        color_discrete_map=colors,
    )

    # Room between the halves keeps their axis titles apart in a narrow panel
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.12)
    for trace in px.bar(x="n_foreground", **shared).data:
        fig.add_trace(trace, row=1, col=1)
    for trace in px.bar(x="log2_odds_ratio", text="mark" if mark == "star" else None, **shared).data:
        fig.add_trace(
            trace.update(showlegend=False, textposition="outside", textfont_size=11, cliponaxis=False),
            row=1, col=2,
        )
    if mark == "qvalue":
        # Right of the bar, or of the zero line for a bar below it: printed
        # outside a short negative bar, a q-value runs over the axis. It may
        # run past the end of the axis, into the margin kept for it
        drawn = df.dropna(subset=["log2_odds_ratio"])
        for value, category, text in zip(drawn["log2_odds_ratio"], drawn[label], drawn["mark"]):
            fig.add_annotation(
                x=max(value, 0.0), y=category, xref="x2", yref="y2", text=text, showarrow=False,
                xanchor="left", xshift=MARK_GAP_PX, font_size=11,
            )

    fig.add_vline(x=0, row=1, col=2, **ZERO_LINE)
    # px applied category_orders to its own figure; the traces copied from it
    # carry no axis order, so it is set again here, first category at the top
    if order is not None:
        fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(order)))
    if mark == "qvalue":
        upper = float(np.nanmax(df["log2_odds_ratio"])) if df["log2_odds_ratio"].notna().any() else 1.0
        fig.update_xaxes(range=[min(0.0, float(np.nanmin(df["log2_odds_ratio"]))), upper * 1.45], row=1, col=2)

    longest = max((len(line) for name in df[label] for line in str(name).split("<br>")), default=0)
    category_px = WRAPPED_CATEGORY_PX if df[label].astype(str).str.contains("<br>").any() else CATEGORY_PX
    top = 50 if show_legend else 15
    fig.update_layout(
        width=width if width is not None else int(np.clip(
            PANELS_PX + LABEL_CHAR_PX * longest, MIN_WIDTH, MAX_WIDTH,
        )),
        height=height if height is not None else (
            top + BOTTOM_PX + TITLE_LINE_PX + max(MIN_PLOT_PX, category_px * n_categories)
        ),
        template=TEMPLATE,
        barmode="group",
        showlegend=show_legend,
        # The plot beside long category names is too narrow for both
        # entries on one row unless the legend may take the margin too
        legend=dict(LEFT_TOP_LEGEND, title_text=""),
        margin=dict(t=top, b=BOTTOM_PX + TITLE_LINE_PX, r=MARK_MARGIN_PX if mark == "qvalue" else None),
        # A gridline per bin reads as a count while there are few; up to ten,
        # one every two keeps the labels apart in a narrow panel
        xaxis=dict(title="Bins in set", dtick=_count_dtick(df["n_foreground"].max())),
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
        fig.update_layout(showlegend=False, shapes=[], width=MIN_WIDTH, height=EMPTY_HEIGHT)

    save_image(fig, file_prefix)
    return fig


def _wrap_label(name):
    """
    ``name`` over two lines when it is longer than ``WRAP_CHARS``, broken at
    the space or after the slash nearest its middle; as it is otherwise, or
    when it has nowhere to break.
    """
    if not isinstance(name, str) or len(name) <= WRAP_CHARS:
        return name
    # A space is replaced by the break; a slash stays at the end of the first line
    breaks = [(i, i + 1) for i, c in enumerate(name) if c == " "] + [(i + 1, i + 1) for i, c in enumerate(name[:-1]) if c == "/"]
    if not breaks:
        return name
    end, start = min(breaks, key=lambda b: (abs(b[0] - len(name) / 2), b[0]))
    return f"{name[:end]}<br>{name[start:]}"
