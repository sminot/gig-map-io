"""
Helper function to add threshold lines to a plot.
"""

import plotly.graph_objects as go


def make_lines(
    val: float,
    color: str,
    fig: go.Figure,
    pos=True,
    neg=True,
    hline=True,
    vline=True,
    **line_kwargs
) -> None:
    """
    Add threshold lines to a plot.

    Parameters
    ----------
    val : float
        The value of the line.
    color : str
        The color of the line.
    fig : go.Figure
        The figure to add the lines to.
    pos : bool
        Whether to add a positive line.
    neg : bool
        Whether to add a negative line.
    hline : bool
        Whether to add a horizontal line.
    vline : bool
        Whether to add a vertical line.
    """

    line_kwargs = dict(line_dash="dash", line_width=2, **line_kwargs)

    to_plot = []
    if pos:
        to_plot.append(val)
    if neg:
        to_plot.append(-val)
    to_plot = list(set(to_plot))

    for val in to_plot:
        if hline:
            fig.add_hline(y=val, line_color=color, **line_kwargs)
        if vline:
            fig.add_vline(x=val, line_color=color, **line_kwargs)
