"""
Observed counts against the counts expected if two variables were independent.

A contingency table asks one question — does this combination happen more
often than chance? — but reading that off a grid of cells shaded by their
departure from expectation is work. Drawn as paired bars, one category per
tick, the answer is the difference in height between two bars standing next to
each other.

This is shared by the study-concordance figure and the bin-presence figure so
that the two look and read the same way. Each also has a heatmap of the same
table, on request, for reading the whole table at a glance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .save_image import save_image
from .style import PRIMARY, SIMPLE_TEMPLATE, TEMPLATE, TOP_LEGEND

EXPECTED_COLOR = "#c3cedb"

#: Where the tick labels' lines sit beneath the axis, and the width of one
#: character of the names set beside them
TICK_LINE_FIRST_PX = 17
TICK_LINE_PX = 17
CHARACTER_PX = 6.5
#: A group heading longer than this many characters per tick under it breaks
#: over two lines
HEADING_CHARS_PER_TICK = 6

#: Width given to each tick's pair of bars, along one axis and in a faceted
#: row, which holds fewer pairs and so can give each one more room
IN_LINE_TICK_PX = 58
FACETED_TICK_PX = 110
#: Height of the data region along one axis, and of each faceted row
IN_LINE_PLOT_PX = 300
FACETED_ROW_PX = 110
#: Gap between faceted rows, which holds the label of the row below it
ROW_GAP_PX = 34
#: Room for the legend above the plot
MIN_WIDTH = 420

#: The heatmap's color scale runs from 1/2^limit to 2^limit times expected, the
#: same for every table, so that a color means the same in each
HEATMAP_LOG2_LIMIT = 2
#: Length of the heatmap's color scale, and the band above the grid it sits in
COLORBAR_PX = 150
COLORBAR_BAND_PX = 70

#: Below this expected count a fold change says more about rounding than about
#: the data, and the chi-squared approximation is unreliable
MIN_EXPECTED_TO_LABEL = 5


def expected_counts(table: pd.DataFrame) -> pd.DataFrame:
    """The counts a contingency table would hold if its rows and columns were independent."""
    return pd.DataFrame(
        np.outer(table.sum(axis=1), table.sum(axis=0)) / table.values.sum(),
        index=table.index,
        columns=table.columns,
    )


def find_axis_break(
    values: Sequence[float],
    min_gap: float = 3.0,
    min_span: float = 6.0,
    max_fraction_above: float = 0.40,
) -> tuple[float, float] | None:
    """
    Where to split the count axis so that small categories stay legible beside
    large ones, or ``None`` when one scale serves.

    Only splits that leave most of the bars below the break are considered. The
    point of the discontinuity is to lift a few outlying categories out of the
    way; splitting at the widest gap wherever it falls can instead isolate one
    small category and push everything else above the break, which is worse
    than no break at all.
    """
    positive = sorted(v for v in values if v > 0)
    n = len(positive)
    if n < 3 or positive[-1] / positive[0] < min_span:
        return None

    first = max(1, int(np.ceil(n - 1 - n * max_fraction_above)))
    candidates = [
        (positive[i + 1] / positive[i], positive[i], positive[i + 1])
        for i in range(first, n - 1)
    ]
    if not candidates:
        return None
    ratio, below, above = max(candidates)
    return (below, above) if ratio >= min_gap else None


def plot_observed_expected(
    ticks: Sequence[str],
    groups: Sequence[str],
    observed: Sequence[float],
    expected: Sequence[float],
    y_title: str,
    tick_rows: Sequence[str] | None = None,
    facet_groups: bool = False,
    width: int | None = None,
    height: int | None = None,
    file_prefix: str | None = None,
) -> go.Figure:
    """
    Paired bars of observed and expected counts, one pair per category.

    ``groups`` names the heading each tick sits under. By default consecutive
    ticks sharing a heading are drawn as one block along a single axis, with
    the heading beneath them; with ``facet_groups`` each heading becomes its
    own row, labeled in the right margin, and the rows share both axes so
    that the bars line up vertically.

    A fold change is printed over each pair whose expected count reaches
    ``MIN_EXPECTED_TO_LABEL``. Like every figure here it has no title, and
    the test behind it belongs in the text rather than beneath the plot.

    ``tick_rows`` names the lines of tick labels that run over several
    (``<br>``), one name per line, set to their left: which study each row
    of arrows is.

    The size follows the number of ticks and rows unless ``width`` and
    ``height`` are given.
    """
    if not (len(ticks) == len(groups) == len(observed) == len(expected)):
        raise ValueError("ticks, groups, observed and expected must be the same length")
    observed, expected = list(observed), list(expected)
    if facet_groups:
        fig = _plot_faceted(list(ticks), list(groups), observed, expected, y_title, width, height)
    else:
        fig = _plot_in_line(list(ticks), list(groups), observed, expected, y_title, tick_rows, width, height)
    save_image(fig, file_prefix)
    return fig


def _add_pair(fig: go.Figure, x, observed, expected, row: int, showlegend: bool) -> None:
    """The observed and expected bars for one panel."""
    fig.add_trace(
        go.Bar(x=x, y=observed, name="Observed", marker_color=PRIMARY, showlegend=showlegend),
        row=row, col=1,
    )
    fig.add_trace(
        go.Bar(x=x, y=expected, name="Expected if independent", marker_color=EXPECTED_COLOR, showlegend=showlegend),
        row=row, col=1,
    )


def _label_fold_change(fig: go.Figure, x, observed: float, expected: float, row: int) -> None:
    if expected < MIN_EXPECTED_TO_LABEL:
        return
    fig.add_annotation(
        x=x, y=max(observed, expected), text=f"{observed / expected:.2f}&#215;", showarrow=False,
        yshift=10, font=dict(size=10, color="#444444"), row=row, col=1,
    )


def _finish(
    fig: go.Figure, y_title: str,
    width: int | None, height: int | None, plot_width: int, plot_height: int,
    margin: dict, y_title_shift: int,
) -> None:
    """
    The size, legend, margins, and one y-axis title for every panel.
    ``plot_width`` and ``plot_height`` size the data region when ``width``
    or ``height`` is not given.
    """
    margin = dict(margin, t=margin.get("t", 50))
    fig.update_layout(
        template=TEMPLATE,
        width=width or max(MIN_WIDTH, margin["l"] + margin["r"] + plot_width),
        height=height or margin["t"] + margin["b"] + plot_height,
        # Hung from the top of the canvas, clear of the label over the first row
        legend=dict(TOP_LEGEND, yref="container", y=1.0, yanchor="top"),
        margin=margin,
    )
    fig.update_xaxes(tickfont=dict(size=13))
    # Drawn as an annotation so that it spans every panel; sized and colored
    # as the template draws an axis title
    fig.add_annotation(
        x=0, xref="paper", y=0.5, yref="paper", xshift=y_title_shift, textangle=-90,
        text=y_title, showarrow=False, font=dict(size=14, color="#222222"),
    )


def _plot_in_line(ticks, groups, observed, expected, y_title, tick_rows, width, height) -> go.Figure:
    """Every category along one axis, with group headings beneath and an axis break where the counts demand it."""
    # Positions, not the tick strings themselves: a categorical axis collapses
    # repeated labels onto one position, and the same label can legitimately
    # appear under more than one group heading
    positions = list(range(len(ticks)))
    axis_break = find_axis_break(observed + expected)
    rows = 2 if axis_break else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.24, 0.76] if axis_break else [1.0],
    )
    for row in range(1, rows + 1):
        _add_pair(fig, positions, observed, expected, row, showlegend=(row == 1))
    fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.05)
    fig.update_xaxes(
        tickmode="array", tickvals=positions, ticktext=ticks, range=[-0.5, len(positions) - 0.5],
    )

    ceiling = max(observed + expected)
    if axis_break:
        below, above = axis_break
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(range=[above * 0.93, ceiling * 1.14], row=1, col=1)
        fig.update_yaxes(range=[0, below * 1.32], row=2, col=1)
        # Slashes across the axis at the discontinuity, so that the two panels
        # are not read as one continuous scale
        gap = (fig.layout.yaxis2.domain[1] + fig.layout.yaxis.domain[0]) / 2
        for offset in (-0.007, 0.007):
            fig.add_shape(
                type="line", xref="paper", yref="paper",
                x0=-0.010, x1=0.010,
                y0=gap + offset - 0.015, y1=gap + offset + 0.015,
                line=dict(color="#333333", width=1.3),
            )
    else:
        fig.update_yaxes(range=[0, ceiling * 1.2], row=1, col=1)

    for i, (obs, exp) in enumerate(zip(observed, expected)):
        row = 1 if (axis_break and max(obs, exp) >= axis_break[1]) else rows
        _label_fold_change(fig, i, obs, exp, row)

    for i in range(len(groups) - 1):
        if groups[i] != groups[i + 1]:
            fig.add_vline(x=i + 0.5, line=dict(color="#e2e2e2", width=1), row="all", col=1)

    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            heading = groups[start]
            # Printed at half a page, a heading over few ticks meets its neighbor
            if len(heading) > HEADING_CHARS_PER_TICK * (i - start):
                heading = _two_lines(heading)
            fig.add_annotation(
                x=(start + i - 1) / 2, y=0, yref="paper", text=f"<b>{heading}</b>",
                showarrow=False, yshift=-55, xanchor="center", yanchor="top",
                font=dict(size=11, color="#333333"),
            )
            start = i

    # Each line of the tick labels named to its left, right-aligned against
    # the axis, the left margin widened to hold the longest name
    margin_left = 85
    for line, name in enumerate(tick_rows or []):
        fig.add_annotation(
            x=0, xref="paper", y=0, yref="paper", xanchor="right", xshift=-8,
            yshift=-TICK_LINE_FIRST_PX - TICK_LINE_PX * line, yanchor="middle",
            text=name, showarrow=False, font=dict(size=11, color="#333333"),
        )
        margin_left = max(margin_left, int(CHARACTER_PX * len(name)) + 20)

    fig.update_yaxes(title_text="", row=rows, col=1)
    _finish(
        fig, y_title, width, height,
        plot_width=IN_LINE_TICK_PX * len(ticks), plot_height=IN_LINE_PLOT_PX,
        margin=dict(b=98, l=margin_left, r=30), y_title_shift=-68,
    )
    return fig


def _plot_faceted(ticks, groups, observed, expected, y_title, width, height) -> go.Figure:
    """
    One row per group, each labeled above it.

    The rows share a y axis so that a bar in one row can be read against a bar
    in another, which is the comparison a contingency table is for. No axis
    break here: a discontinuity already costs the reader one leap of scale,
    and combining it with faceted rows asks for two.
    """
    order = list(dict.fromkeys(groups))
    ceiling = max(observed + expected) * 1.22
    fig = make_subplots(
        rows=len(order), cols=1, shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=ROW_GAP_PX / (len(order) * FACETED_ROW_PX + (len(order) - 1) * ROW_GAP_PX),
    )
    for row, group in enumerate(order, start=1):
        members = [i for i, g in enumerate(groups) if g == group]
        _add_pair(
            fig, [ticks[i] for i in members], [observed[i] for i in members], [expected[i] for i in members],
            row, showlegend=(row == 1),
        )
        fig.update_yaxes(range=[0, ceiling], row=row, col=1)
        for position, i in enumerate(members):
            _label_fold_change(fig, position, observed[i], expected[i], row)

        # Row label above the row, set level: turned on end beside the row
        # it runs longer than the row once the panel is drawn small for
        # print. Anchored to this row's own axis domain rather than to the
        # figure, so that it tracks the row wherever the layout puts it.
        axis = "y" if row == 1 else f"y{row}"
        fig.add_annotation(
            x=0, xref="paper", y=1.0, yref=f"{axis} domain", yshift=2,
            text=f"<b>{group}</b>", showarrow=False,
            xanchor="left", yanchor="bottom", font=dict(size=11, color="#333333"),
        )

    fig.update_layout(barmode="group", bargap=0.42, bargroupgap=0.05)
    fig.update_yaxes(nticks=4)
    # Tick marks only under the last row; elsewhere they reach into the label below
    fig.update_xaxes(ticks="")
    fig.update_xaxes(ticks="outside", row=len(order), col=1)
    widest_row = max(groups.count(group) for group in order)
    _finish(
        fig, y_title, width, height,
        plot_width=FACETED_TICK_PX * widest_row,
        plot_height=len(order) * FACETED_ROW_PX + (len(order) - 1) * ROW_GAP_PX,
        margin=dict(b=55, l=72, r=20, t=50 + ROW_GAP_PX), y_title_shift=-58,
    )
    return fig


def plot_observed_expected_heatmap(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    x_title: str,
    y_title: str,
    width: int,
    height: int,
    file_prefix: str | None = None,
) -> go.Figure:
    """
    A contingency table as a grid of cells, ``observed``'s rows up (the
    first at the bottom) and its columns across: each cell colored by how
    many times more (warm) or fewer (cool) counts it holds than ``expected``,
    on a log scale centered on 1x, and labeled with its observed and expected
    counts.

    The color scale runs across the top, where the paired bars put their
    legend, and labels of several words break over two lines: beside the
    paired bars the grid gets half a page, and a scale or a label down the
    side takes the width its cells' text needs.
    """
    observed = observed.rename(index=_two_lines, columns=_two_lines)
    expected = expected.rename(index=_two_lines, columns=_two_lines)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_fold = np.log2(observed / expected.where(expected > 0))
    # A cell beyond the scale, or empty where some were expected, sits at its
    # end, and its label gives the exact fold
    limit = HEATMAP_LOG2_LIMIT
    z = log_fold.clip(-limit, limit)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=list(observed.columns), y=list(observed.index),
        colorscale="RdBu_r", zmid=0, zmin=-limit, zmax=limit, xgap=2, ygap=2, hoverinfo="skip",
        # The title at the cells' text size: at the axis titles' it outweighs them
        colorbar=dict(
            title=dict(text="Observed / expected", font_size=12, side="top"),
            orientation="h", x=0.5, xanchor="center", y=1, yanchor="bottom", ypad=12,
            thickness=10, lenmode="pixels", len=COLORBAR_PX,
            tickvals=list(range(-limit, limit + 1)),
            ticktext=[f"{2 ** v}×" if v >= 0 else f"1/{2 ** -v}×" for v in range(-limit, limit + 1)],
        ),
    ))
    for r in observed.index:
        for c in observed.columns:
            cell = log_fold.loc[r, c]
            fold = f"{2.0 ** cell:.2f}×" if np.isfinite(cell) else ("0×" if expected.loc[r, c] > 0 else "")
            # Dark cells take white text
            dark = np.isfinite(z.loc[r, c]) and abs(z.loc[r, c]) > 0.6 * limit
            fig.add_annotation(
                x=c, y=r, showarrow=False,
                text=f"<b>{observed.loc[r, c]:,}</b><br>{expected.loc[r, c]:,.1f} expected<br>{fold}",
                font=dict(size=12, color="#ffffff" if dark else "#1a1a1a"),
            )
    fig.update_layout(
        template=SIMPLE_TEMPLATE, width=width, height=height,
        margin=dict(t=COLORBAR_BAND_PX, b=75),
        # Clear of tick labels that run over two lines
        xaxis=dict(title=dict(text=x_title, standoff=14), side="bottom", showline=False, ticks=""),
        yaxis=dict(title=y_title, showline=False, ticks=""),
    )
    save_image(fig, file_prefix)
    return fig


def _two_lines(label: str) -> str:
    """``label`` broken at the space nearest its middle, or as it is if it is one word."""
    spaces = [i for i, character in enumerate(label) if character == " "]
    if not spaces:
        return label
    middle = min(spaces, key=lambda i: (abs(i - len(label) / 2), i))
    return f"{label[:middle]}<br>{label[middle + 1:]}"
