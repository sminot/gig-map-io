import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path


def save_image(
    fig: go.Figure | plt.Figure,
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

    if isinstance(fig, go.Figure):
        _save_image_plotly_figure(fig, file_prefix, as_html, as_pdf, as_png, as_json)
    elif isinstance(fig, plt.Figure):
        _save_image_matplotlib_figure(fig, file_prefix, as_pdf, as_png)
    else:
        raise ValueError(f"Unsupported figure type: {type(fig)}")


def _save_image_plotly_figure(
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


def _save_image_matplotlib_figure(
    fig: plt.Figure,
    file_prefix: str | None = None,
    as_pdf = True,
    as_png = True
):
    """
    Helper function used to optionally save an image as PDF and HTML.
    """
    if file_prefix is None:
        return

    # Make sure that the folder exists
    Path(file_prefix).parent.mkdir(parents=True, exist_ok=True)

    if as_pdf:
        fig.savefig(file_prefix + ".pdf")
    if as_png:
        fig.savefig(file_prefix + ".png")