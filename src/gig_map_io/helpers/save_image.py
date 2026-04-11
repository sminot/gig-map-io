import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np


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


def trim_png(input_path: str, output_path: str | None = None, bg_color: tuple | None = None) -> Image.Image:
    """
    Removes blank/whitespace from the edges of a PNG file.

    Handles both transparent PNGs (trims fully transparent pixels) and
    opaque PNGs (trims pixels matching the background color).

    Args:
        input_path:  Path to the source PNG file.
        output_path: Where to save the trimmed image. If None, the image
                     is returned but not saved.
        bg_color:    Background color to treat as "blank" for opaque images,
                     as an (R, G, B) tuple. Defaults to white (255, 255, 255).
                     Ignored for images with an alpha channel (transparent
                     pixels are trimmed instead).

    Returns:
        The trimmed PIL Image object.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError:        If the image is entirely blank.
    """
    with Image.open(input_path) as img:
        original_mode = img.mode
        width, height = img.size

        if bg_color is None:
            bg_color = (255, 255, 255)
        rgb = np.array(img.convert("RGB"))
        bg = np.array(bg_color[:3], dtype=np.uint8)
        # Mask = True where pixel does NOT match background
        mask = ~np.all(rgb == bg, axis=2)

        rows = np.any(mask, axis=1)  # True for rows containing content
        cols = np.any(mask, axis=0)  # True for cols containing content

        if not rows.any():
            raise ValueError("The image is entirely blank — nothing to trim.")

        top    = int(rows.argmax())
        bottom = int(len(rows) - rows[::-1].argmax() - 1)
        left   = int(cols.argmax())
        right  = int(len(cols) - cols[::-1].argmax() - 1)

        result = img.crop((left, top, right + 1, bottom + 1))

    if output_path:
        result.save(output_path)
        print(f"Saved trimmed image → {output_path}")
        print(f"  Original : {width} x {height} px")
        print(f"  Trimmed  : {result.width} x {result.height} px")

    return result


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
        png_path = file_prefix + ".png"
        fig.write_image(png_path)
        trim_png(png_path, file_prefix + ".trimmed.png")
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
        fig.savefig(file_prefix + ".pdf", bbox_inches="tight")
    if as_png:
        fig.savefig(file_prefix + ".png", bbox_inches="tight")