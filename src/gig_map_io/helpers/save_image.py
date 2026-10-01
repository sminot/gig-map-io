"""
Write a figure in every format an analysis keeps.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from .style import MATPLOTLIB_DPI, PNG_SCALE


def save_image(fig: go.Figure | plt.Figure, file_prefix: str | None = None) -> None:
    """
    Write ``fig`` under ``file_prefix``: a plotly figure as HTML, PDF, PNG,
    a whitespace-trimmed PNG and the JSON specification; a matplotlib figure
    as PDF and PNG. Nothing is written without a prefix, so a plotting method
    can take one optionally.
    """
    if file_prefix is None:
        return
    Path(file_prefix).parent.mkdir(parents=True, exist_ok=True)

    if isinstance(fig, go.Figure):
        fig.write_html(file_prefix + ".html")
        fig.write_image(file_prefix + ".pdf")
        fig.write_image(file_prefix + ".png", scale=PNG_SCALE)
        trim_png(file_prefix + ".png", file_prefix + ".trimmed.png")
        fig.write_json(file_prefix + ".json")
    elif isinstance(fig, plt.Figure):
        fig.savefig(file_prefix + ".pdf", bbox_inches="tight")
        fig.savefig(file_prefix + ".png", bbox_inches="tight", dpi=MATPLOTLIB_DPI)
    else:
        raise ValueError(f"Unsupported figure type: {type(fig)}")


def trim_png(input_path: str, output_path: str) -> None:
    """Write ``input_path`` with its white margins cropped away."""
    with Image.open(input_path) as img:
        rgb = np.array(img.convert("RGB"))
        content = ~np.all(rgb == 255, axis=2)
        rows, cols = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
        if not len(rows):
            raise ValueError(f"{input_path} is entirely blank")
        img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)).save(output_path)
