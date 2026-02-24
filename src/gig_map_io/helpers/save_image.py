import plotly.graph_objects as go
from pathlib import Path


def save_image(
    fig: go.Figure,
    file_prefix: str | None = None,
    as_html = True,
    as_pdf = True,
    as_png = True,
    as_json = True
):
    """
    Helper function used to optionally save an image as PDF and HTML.
    """
    if file_prefix is None:
        return

    # Make sure that the folder exists
    Path(file_prefix).parent.mkdir(parents=True, exist_ok=True)

    if as_html:
        fig.write_html(file_prefix + ".html")
    if as_pdf:
        fig.write_image(file_prefix + ".pdf")
    if as_png:
        fig.write_image(file_prefix + ".png")
    if as_json:
        fig.write_json(file_prefix + ".json")
