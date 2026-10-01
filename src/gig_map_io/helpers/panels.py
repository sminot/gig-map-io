"""
Assemble the PDFs that several analyses wrote into one multi-panel figure.

A figure is described by a small specification -- normally read from a JSON
file checked in beside the output -- naming the panel PDFs in reading order
and how many columns to lay them out in. Panels wrap into rows and each is
lettered.

A plotly panel is drawn for the screen, several hundred pixels wide with
13 px text, and shrinking that PDF into a 55 mm cell leaves the text
unreadable. So when the figure's exact specification (the ``.json``
``save_image`` writes) sits beside the PDF, the panel is drawn again on a
canvas as many times wider than its cell as its text is larger than the
point size wanted, and that page is scaled down into the cell: the text
lands at the point size asked for and everything else keeps the
proportions it had on screen. A PDF with no specification beside it, such
as a matplotlib figure drawn in inches, is placed as it is. Either way the
content stays vector, so the assembled figure prints at any size and its
text stays selectable.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from pypdf import PageObject, PdfReader, PdfWriter, Transformation

#: PDF user space is in points
POINTS_PER_MM = 72 / 25.4

#: Layout defaults, in millimetres: a journal double-column width, the gap
#: between cells, the page edge, and the label size in points
DEFAULT_WIDTH_MM = 170.0
DEFAULT_COLUMNS = 3
DEFAULT_GUTTER_MM = 4.0
DEFAULT_MARGIN_MM = 2.0
DEFAULT_LABEL_SIZE_PT = 10.0

#: Text size in a re-rendered panel, in points: what journals ask for
DEFAULT_FONT_PT = 7.0

#: Plotly's own default canvas, for a specification that leaves its size out
PLOTLY_DEFAULT_WIDTH = 700.0
PLOTLY_DEFAULT_HEIGHT = 450.0
PLOTLY_DEFAULT_FONT = 12.0

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Placement:
    """One panel's page and where it lands, in points from the bottom left."""

    page: PageObject
    label: str
    scale: float
    x: float
    y: float
    cell_x: float
    cell_top: float


def compose_panels(spec: Dict[str, Any], root: str | Path, output: str | Path) -> Path:
    """
    Lay the panels named in ``spec`` out on one page and write it to ``output``.

    Parameters
    ----------
    spec:
        ``panels``: the panels in reading order, each a dict with ``pdf`` (a
        path relative to ``root``), an optional ``label`` (otherwise lettered
        A, B, C ... in order) and an optional ``span`` (how many columns it
        takes, default 1). ``width_mm``, ``columns``, ``gutter_mm``,
        ``margin_mm`` and ``label_size_pt`` set the layout;
        ``max_panel_height_mm`` caps how tall a panel may be drawn, so that a
        tall, narrow panel is shrunk and centered in its cell rather than
        stretching its row. ``font_pt`` is the text size a plotly panel is
        rendered at, default 7, and a panel may carry its own; a panel may
        also carry ``height_mm`` to be rendered that tall instead of keeping
        the proportions it was drawn with, which is what gives a legend room
        in a narrow cell.
    root:
        The directory the panel paths are relative to.
    output:
        Where the assembled PDF is written.
    """
    panels = spec.get("panels", [])
    if not panels:
        raise ValueError("The figure specification names no panels")

    width = float(spec.get("width_mm", DEFAULT_WIDTH_MM)) * POINTS_PER_MM
    columns = int(spec.get("columns", DEFAULT_COLUMNS))
    gutter = float(spec.get("gutter_mm", DEFAULT_GUTTER_MM)) * POINTS_PER_MM
    margin = float(spec.get("margin_mm", DEFAULT_MARGIN_MM)) * POINTS_PER_MM
    max_height = spec.get("max_panel_height_mm")
    max_height = float(max_height) * POINTS_PER_MM if max_height is not None else None
    label_size = float(spec.get("label_size_pt", DEFAULT_LABEL_SIZE_PT))
    font_pt = float(spec.get("font_pt", DEFAULT_FONT_PT))

    column_width = (width - 2 * margin - (columns - 1) * gutter) / columns
    rows = _wrap_rows(panels, columns)

    placements: List[Placement] = []
    top = margin
    for row in rows:
        row_height = 0.0
        cell_x = margin
        for index, panel in row:
            span = int(panel.get("span", 1))
            cell_width = span * column_width + (span - 1) * gutter
            panel_height = panel.get("height_mm")
            page = _panel_page(
                Path(root) / panel["pdf"],
                cell_width,
                max_height,
                float(panel.get("font_pt", font_pt)),
                float(panel_height) * POINTS_PER_MM if panel_height is not None else None,
            )
            page_width, page_height = _page_size(page)
            scale = cell_width / page_width
            if max_height is not None and page_height * scale > max_height:
                scale = max_height / page_height
            placed_width, placed_height = page_width * scale, page_height * scale
            placements.append(Placement(
                page=page,
                label=panel.get("label", LABELS[index]),
                scale=scale,
                # Centered in a cell it does not fill, but always hung from the top
                x=cell_x + (cell_width - placed_width) / 2,
                y=top + placed_height,
                cell_x=cell_x,
                cell_top=top,
            ))
            row_height = max(row_height, placed_height)
            cell_x += cell_width + gutter
        top += row_height + gutter
    height = top - gutter + margin

    writer = PdfWriter()
    sheet = writer.add_blank_page(width=width, height=height)
    for placement in placements:
        # The placement was measured from the top; PDF counts from the bottom
        sheet.merge_transformed_page(
            placement.page,
            Transformation().scale(placement.scale, placement.scale).translate(
                placement.x - _page_origin(placement.page)[0] * placement.scale,
                height - placement.y - _page_origin(placement.page)[1] * placement.scale,
            ),
        )
    sheet.merge_page(_label_overlay(placements, width, height, label_size))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)
    return output


def _panel_page(
    pdf: Path, width: float, max_height: float | None, font_pt: float, height: float | None
) -> PageObject:
    """
    The page to place for one panel: the plotly figure beside ``pdf``
    rendered at ``width`` points with ``font_pt`` text when its
    specification exists, otherwise the first page of ``pdf`` itself.
    """
    spec = pdf.with_suffix(".json")
    if not spec.exists():
        return PdfReader(pdf).pages[0]
    return render_at_print_size(pio.read_json(spec), width, max_height, font_pt, height)


def render_at_print_size(
    fig: go.Figure, width: float, max_height: float | None, font_pt: float, height: float | None = None
) -> PageObject:
    """
    ``fig`` as a PDF page that, scaled to ``width`` points wide, shows its
    base font at ``font_pt``. The page is drawn larger by the ratio of the
    figure's font size to ``font_pt``, with every pixel size the figure
    gives left as it is, so the caller's scaling brings the text to
    ``font_pt`` and keeps everything else in the proportions it had on
    screen. The height keeps the figure's proportions unless ``height`` is
    given, and ``max_height`` cuts either down.
    """
    layout = fig.to_dict()["layout"]
    template_layout = (layout.get("template") or {}).get("layout", {})
    base_font = (
        (layout.get("font") or {}).get("size")
        or (template_layout.get("font") or {}).get("size")
        or PLOTLY_DEFAULT_FONT
    )
    original_width = layout.get("width") or PLOTLY_DEFAULT_WIDTH
    original_height = layout.get("height") or PLOTLY_DEFAULT_HEIGHT
    if height is None:
        height = width * original_height / original_width
        if max_height is not None and height > max_height:
            height = max_height
            width = max_height * original_width / original_height
    elif max_height is not None:
        height = min(height, max_height)

    enlargement = base_font / font_pt
    page = go.Figure(fig).update_layout(width=width * enlargement, height=height * enlargement)
    return PdfReader(io.BytesIO(page.to_image(format="pdf"))).pages[0]


def _wrap_rows(panels: Sequence[Dict[str, Any]], columns: int) -> List[List[tuple]]:
    """Reading order into rows, a panel moving to the next row when its span does not fit."""
    rows: List[List[tuple]] = [[]]
    used = 0
    for index, panel in enumerate(panels):
        span = int(panel.get("span", 1))
        if span > columns:
            raise ValueError(f"Panel {index} spans {span} columns of a {columns}-column figure")
        if used + span > columns:
            rows.append([])
            used = 0
        rows[-1].append((index, panel))
        used += span
    return rows


def _page_size(page: PageObject) -> tuple:
    box = page.cropbox
    return float(box.width), float(box.height)


def _page_origin(page: PageObject) -> tuple:
    box = page.cropbox
    return float(box.left), float(box.bottom)


def _label_overlay(
    placements: Sequence[Placement], width: float, height: float, size: float
) -> PageObject:
    """
    A transparent page of the same size carrying the panel letters, each at
    the top-left corner of its panel's cell. Drawn with matplotlib so that no
    further dependency is needed, with the text kept as text.
    """
    with matplotlib.rc_context({
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    }):
        fig = plt.figure(figsize=(width / 72, height / 72))
        fig.patch.set_alpha(0)
        axes = fig.add_axes([0, 0, 1, 1])
        axes.set_xlim(0, width)
        axes.set_ylim(0, height)
        axes.axis("off")
        for placement in placements:
            axes.text(
                placement.cell_x + 0.3 * size,
                height - placement.cell_top - 0.3 * size,
                placement.label,
                fontsize=size,
                fontweight="bold",
                ha="left",
                va="top",
            )
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf", transparent=True)
        plt.close(fig)
    buffer.seek(0)
    return PdfReader(buffer).pages[0]
