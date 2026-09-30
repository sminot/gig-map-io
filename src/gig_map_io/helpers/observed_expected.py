"""
Observed counts against the counts expected if two variables were independent.

A contingency table asks one question — does this combination happen more
often than chance? — but reading that off a grid of cells shaded by their
departure from expectation is work. Drawn as paired bars, one category per
tick, the answer is the difference in height between two bars standing next to
each other.

This is shared by the study-concordance figure and the bin-presence figure so
that the two look and read the same way.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .save_image import save_image

OBSERVED_COLOR = "#2c6fbb"
EXPECTED_COLOR = "#c3cedb"

#: Below this expected count a fold change says more about rounding than about
#: the data, and the chi-squared approximation is unreliable
MIN_EXPECTED_TO_LABEL = 5


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


def _top_margin(subtitle: str) -> int:
    """Room for the title block, which grows if the subtitle is wrapped."""
    return 120 + 18 * subtitle.count("<br>")


def plot_observed_expected(
    ticks: Sequence[str],
    groups: Sequence[str],
    observed: Sequence[float],
    expected: Sequence[float],
    title: str,
    subtitle: str,
    y_title: str,
    caption: str | None = None,
    facet_groups: bool = False,
    width: int = 800,
    height: int = 500,
    file_prefix: str | None = None,
) -> go.Figure:
    """
    Paired bars of observed and expected counts, one pair per category.

    ``groups`` names the heading each tick sits under. By default consecutive
    ticks sharing a heading are drawn as one block along a single axis, with
    the heading beneath them; with ``facet_groups`` each heading becomes its
    own row, labelled in the right margin, and the rows share both axes so
    that the bars line up vertically.

    A fold change is printed over each pair whose expected count reaches
    ``MIN_EXPECTED_TO_LABEL``.
    """
    if facet_groups:
        return _plot_faceted(
            ticks=ticks, groups=groups, observed=list(observed), expected=list(expected),
            title=title, subtitle=subtitle, y_title=y_title, caption=caption,
            width=width, height=height, file_prefix=file_prefix,
        )

    if not (len(ticks) == len(groups) == len(observed) == len(expected)):
        raise ValueError("ticks, groups, observed and expected must be the same length")

    observed, expected = list(observed), list(expected)
    # Positions, not the tick strings themselves: a categorical axis collapses
    # repeated labels onto one position, and the same label can legitimately
    # appear under more than one group heading
    positions = list(range(len(ticks)))
    axis_break = find_axis_break(observed + expected)
    rows = 2 if axis_break else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.32, 0.68] if axis_break else [1.0],
    )
    for row in range(1, rows + 1):
        fig.add_trace(
            go.Bar(x=positions, y=observed, name="Observed",
                   marker_color=OBSERVED_COLOR, showlegend=(row == 1)),
            row=row, col=1,
        )
        fig.add_trace(
            go.Bar(x=positions, y=expected, name="Expected if independent",
                   marker_color=EXPECTED_COLOR, showlegend=(row == 1)),
            row=row, col=1,
        )

    fig.update_layout(
        barmode="group", bargap=0.32, bargroupgap=0.05, template="plotly_white",
        width=width, height=height,
        title=dict(text=f"{title}<br><sub>{subtitle}</sub>", x=0.5),
        legend=dict(orientation="h", y=1.0, x=0.5, xanchor="center", yanchor="bottom"),
        margin=dict(b=130 if caption else 105, t=_top_margin(subtitle), l=85),
    )
    fig.update_xaxes(
        tickmode="array", tickvals=positions, ticktext=list(ticks),
        range=[-0.5, len(positions) - 0.5], tickfont=dict(size=13),
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
        if exp < MIN_EXPECTED_TO_LABEL:
            continue
        row = 1 if (axis_break and max(obs, exp) >= axis_break[1]) else rows
        fig.add_annotation(
            x=i, y=max(obs, exp), text=f"{obs / exp:.2f}&#215;", showarrow=False,
            yshift=10, font=dict(size=10, color="#444444"), row=row, col=1,
        )

    for i in range(len(groups) - 1):
        if groups[i] != groups[i + 1]:
            fig.add_vline(x=i + 0.5, line=dict(color="#e2e2e2", width=1), row="all", col=1)

    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            fig.add_annotation(
                x=(start + i - 1) / 2, y=0, yref="paper", text=f"<b>{groups[start]}</b>",
                showarrow=False, yshift=-62, xanchor="center",
                font=dict(size=11, color="#333333"),
            )
            start = i

    # One axis title spanning both panels, rather than one centred on the lower
    fig.update_yaxes(title_text="", row=rows, col=1)
    fig.add_annotation(
        x=0, xref="paper", y=0.5, yref="paper", xshift=-68, textangle=-90,
        text=y_title, showarrow=False, font=dict(size=13, color="#2a3f5f"),
    )
    if caption:
        fig.add_annotation(
            x=0.5, xref="paper", y=0, yref="paper", yshift=-96, xanchor="center",
            showarrow=False, font=dict(size=11, color="#555555"), text=caption,
        )

    save_image(fig, file_prefix)
    return fig


def _plot_faceted(
    ticks: Sequence[str],
    groups: Sequence[str],
    observed: list[float],
    expected: list[float],
    title: str,
    subtitle: str,
    y_title: str,
    caption: str | None,
    width: int,
    height: int,
    file_prefix: str | None,
) -> go.Figure:
    """
    One row per group, labelled in the right margin.

    The rows share a y axis so that a bar in one row can be read against a bar
    in another, which is the comparison a contingency table is for. No axis
    break here: a discontinuity already costs the reader one leap of scale,
    and combining it with faceted rows asks for two.
    """
    order, seen = [], set()
    for group in groups:
        if group not in seen:
            order.append(group)
            seen.add(group)

    ceiling = max(observed + expected) * 1.22
    fig = make_subplots(
        rows=len(order), cols=1, shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.08,
    )

    for row, group in enumerate(order, start=1):
        members = [i for i, g in enumerate(groups) if g == group]
        labels = [ticks[i] for i in members]
        fig.add_trace(
            go.Bar(x=labels, y=[observed[i] for i in members], name="Observed",
                   marker_color=OBSERVED_COLOR, showlegend=(row == 1)),
            row=row, col=1,
        )
        fig.add_trace(
            go.Bar(x=labels, y=[expected[i] for i in members],
                   name="Expected if independent", marker_color=EXPECTED_COLOR,
                   showlegend=(row == 1)),
            row=row, col=1,
        )
        fig.update_yaxes(range=[0, ceiling], row=row, col=1)

        for position, i in enumerate(members):
            if expected[i] < MIN_EXPECTED_TO_LABEL:
                continue
            fig.add_annotation(
                x=position, y=max(observed[i], expected[i]),
                text=f"{observed[i] / expected[i]:.2f}&#215;", showarrow=False,
                yshift=10, font=dict(size=10, color="#444444"), row=row, col=1,
            )

        # Row label in the margin, in plotly's own facet style. Anchored to
        # this row's own axis domain rather than to the figure, so that it
        # tracks the row wherever the layout puts it.
        axis = "y" if row == 1 else f"y{row}"
        fig.add_annotation(
            x=1.0, xref="paper", xshift=16, y=0.5, yref=f"{axis} domain",
            text=f"<b>{group}</b>", showarrow=False, textangle=90,
            xanchor="left", yanchor="middle", font=dict(size=11, color="#333333"),
        )

    fig.update_layout(
        barmode="group", bargap=0.42, bargroupgap=0.05, template="plotly_white",
        width=width, height=height,
        title=dict(text=f"{title}<br><sub>{subtitle}</sub>", x=0.5),
        legend=dict(orientation="h", y=1.0, x=0.5, xanchor="center", yanchor="bottom"),
        margin=dict(b=80 if caption else 60, t=_top_margin(subtitle), l=85, r=98),
    )
    fig.update_xaxes(tickfont=dict(size=13))
    fig.add_annotation(
        x=0, xref="paper", y=0.5, yref="paper", xshift=-58, textangle=-90,
        text=y_title, showarrow=False, font=dict(size=13, color="#2a3f5f"),
    )
    if caption:
        fig.add_annotation(
            x=0.5, xref="paper", y=0, yref="paper", yshift=-52, xanchor="center",
            showarrow=False, font=dict(size=11, color="#555555"), text=caption,
        )

    save_image(fig, file_prefix)
    return fig
