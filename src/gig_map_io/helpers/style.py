"""
The one look every figure shares.

Registers a plotly template carrying the fonts, title placement, legend and
axis styling, and the qualitative palette, and names the colours that mean
the same thing wherever they appear: which organism a point belongs to, and
whether a group is the case-like or the control-like side of a contrast.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

#: Plotly template names. ``TEMPLATE`` has a light grid, ``SIMPLE_TEMPLATE``
#: has axis lines only, for heatmaps and trees where a grid would compete
#: with the marks.
TEMPLATE = "plotly_white+gig_map"
SIMPLE_TEMPLATE = "simple_white+gig_map"

#: Raster export resolution, as a multiple of the figure's layout size.
PNG_SCALE = 2
MATPLOTLIB_DPI = 200

#: Colour of a series when nothing distinguishes it from another.
PRIMARY = "#2c6fbb"
NEUTRAL = "#8a8f98"

#: Ten distinguishable hues (Paul Tol's colour-blind safe set, led by
#: ``PRIMARY``) for organisms, studies and any other categorical series.
QUALITATIVE = [
    PRIMARY, "#CC6677", "#117733", "#DDCC77", "#AA4499",
    "#44AA99", "#999933", "#882255", "#88CCEE", "#661100",
]

#: For a categorical series with more levels than ``QUALITATIVE`` can tell
#: apart, such as participants or community types.
LARGE_QUALITATIVE = px.colors.qualitative.Dark24

#: The two sides of a contrast: cases, or whatever is being tested for, in a
#: warm colour; controls in a cool one (Okabe-Ito vermillion and blue).
CASE_COLOR = "#D55E00"
CONTROL_COLOR = "#0072B2"

#: Reference lines: the zero line is barely there, thresholds are dashed.
ZERO_LINE = dict(line_color="#777777", line_width=1, line_dash="solid")
THRESHOLD_LINE = dict(line_color="#c0392b", line_width=1, line_dash="dash")

#: Marker opacity for scatter plots dense enough that points overlap.
DENSE_MARKER_OPACITY = 0.7

#: A horizontal legend centred above the plot area.
TOP_LEGEND = dict(orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom")

#: The thresholds that make a bin significant: the smallest effect size and
#: the largest FDR-adjusted q-value.
ESTIMATE_THRESH = 0.25
FDR_THRESH = 0.2

pio.templates["gig_map"] = go.layout.Template(
    layout=dict(
        font=dict(family="Helvetica, Arial, sans-serif", size=13, color="#222222"),
        title=dict(x=0.5, xanchor="center", font=dict(size=16)),
        colorway=QUALITATIVE,
        legend=dict(font=dict(size=12), title=dict(font=dict(size=13))),
        xaxis=dict(
            title=dict(font=dict(size=14), standoff=10),
            ticks="outside", showline=True, linecolor="#444444", linewidth=1,
            gridcolor="#ececec", zeroline=False,
        ),
        yaxis=dict(
            title=dict(font=dict(size=14), standoff=10),
            ticks="outside", showline=True, linecolor="#444444", linewidth=1,
            gridcolor="#ececec", zeroline=False,
        ),
        margin=dict(l=70, r=30, t=70, b=60),
        coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=14)),
    ),
)
pio.templates.default = TEMPLATE


def organism_order(organisms: Sequence[str]) -> List[str]:
    """
    Organisms in the one order every figure lists them in: alphabetical by
    display name, so "R. gnavus" precedes "Roseburia".
    """
    return sorted(set(organisms))


def organism_colors(organisms: Sequence[str]) -> Dict[str, str]:
    """
    A colour for each organism, assigned by name so that the same organism is
    drawn the same way in every figure whatever subset of organisms it shows.
    """
    return {
        name: QUALITATIVE[i % len(QUALITATIVE)]
        for i, name in enumerate(organism_order(organisms))
    }


def group_colors(order: Sequence[str]) -> Dict[str, str]:
    """
    Colours for the levels of a sample group, in the order the study declares
    them. A two-level group is a contrast, with the first level the case-like
    one; anything else takes the qualitative palette in order.
    """
    order = list(order)
    if len(order) == 2:
        return {order[0]: CASE_COLOR, order[1]: CONTROL_COLOR}
    palette = QUALITATIVE if len(order) <= len(QUALITATIVE) else LARGE_QUALITATIVE
    return {name: palette[i % len(palette)] for i, name in enumerate(order)}


def legend_above(fig: go.Figure, offset_px: int = 30) -> None:
    """
    Put the legend above the plot, clear of the facet titles that plotly
    express draws at the top of the first facet. Needs the figure's height to
    be set already.
    """
    fig.update_layout(legend=dict(TOP_LEGEND, y=1 + offset_px / fig.layout.height))
