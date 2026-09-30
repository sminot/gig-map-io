"""
Helper function to add reference lines to a plot.
"""

import plotly.graph_objects as go


def make_lines(
    val: float,
    fig: go.Figure,
    pos=True,
    neg=True,
    hline=True,
    vline=True,
    **line_kwargs
) -> None:
    """
    Add reference lines at ``val`` (and ``-val``) to a plot.

    Parameters
    ----------
    val : float
        The value of the line.
    fig : go.Figure
        The figure to add the lines to.
    pos : bool
        Whether to add a line at the positive value.
    neg : bool
        Whether to add a line at the negative value.
    hline : bool
        Whether to add a horizontal line.
    vline : bool
        Whether to add a vertical line.
    line_kwargs :
        Line styling (``line_color``, ``line_dash``, ``line_width``) and
        subplot placement (``row``, ``col``), as accepted by ``add_hline``.
        See ``style.ZERO_LINE`` and ``style.THRESHOLD_LINE``.
    """
    to_plot = []
    if pos:
        to_plot.append(val)
    if neg:
        to_plot.append(-val)
    to_plot = sorted(set(to_plot))

    for val in to_plot:
        if hline:
            fig.add_hline(y=val, **line_kwargs)
        if vline:
            fig.add_vline(x=val, **line_kwargs)
