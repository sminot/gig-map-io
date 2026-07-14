from plotly import graph_objects as go


def log_scale_axis(fig: go.Figure, axis: str) -> None:
    """
    Set the axis to a log scale with custom tick labels.

    Parameters
    ----------
    fig: plotly.graph_objects.Figure
    axis: str

    Returns
    -------
    None
    """
    vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    text = ["1", "10", "100", "1k", "10k", "100k", "1M", "10M", "100M", "1B"]

    # Update the x tick labels so that 4 -> 10**4, 5 -> 10**5, etc.
    fig.update_layout(
        **{axis: dict(
            tickmode="array",
            tickvals=vals,
            ticktext=text
        )}
    )
