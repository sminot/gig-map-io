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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from pypdf.generic import ArrayObject, FloatObject, NameObject

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
PLOTLY_DEFAULT_MARGIN = dict(l=80, r=80, t=100, b=80)

#: Room a legend entry's symbol and padding take, in the figure's own pixels
LEGEND_SYMBOL_PX = 50.0
#: Height of one entry of a vertical legend, in the figure's own pixels
LEGEND_ENTRY_PX = 24.0

#: Narrowest data region a panel is drawn with, in the figure's own pixels
MIN_DATA_PX = 120.0

#: Plotly's marker diameter where a trace gives none, in pixels
PLOTLY_DEFAULT_MARKER_PX = 6.0
#: Smallest diameter a marker is shrunk to for print, in the figure's own pixels
MIN_PRINT_MARKER_PX = 3.0
#: Width of one character of text, as a fraction of the font size
CHARACTER_WIDTH = 0.55
#: An axis is drawn this much longer than its title, so the title clears its ends
AXIS_TITLE_PADDING = 1.15

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
        in a narrow cell, ``aspect`` to give its data region that height over
        width instead of the shape it had on screen, ``plot_height_mm`` to give
        the data region that height, so that panels side by side can match,
        and ``"legend": false``
        to be drawn without its legend, where another panel's legend already
        names the same traces.
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
                bool(panel.get("legend", True)),
                panel.get("aspect"),
                float(panel["plot_height_mm"]) * POINTS_PER_MM if "plot_height_mm" in panel else None,
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
        transformation = Transformation().scale(placement.scale, placement.scale).translate(
            placement.x - _page_origin(placement.page)[0] * placement.scale,
            height - placement.y - _page_origin(placement.page)[1] * placement.scale,
        )
        _move_patterns(placement.page, transformation.ctm)
        sheet.merge_transformed_page(placement.page, transformation)
    sheet.merge_page(_label_overlay(placements, width, height, label_size))
    # Merging copies a panel's objects many times over and leaves the copies
    # behind: a box plot drawing thousands of outliers grew from 0.1 to 15 MB
    writer.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)
    return output


def _panel_page(
    pdf: Path, width: float, max_height: float | None, font_pt: float, height: float | None, legend: bool,
    aspect: float | None, plot_height: float | None,
) -> PageObject:
    """
    The page to place for one panel: the plotly figure beside ``pdf``
    rendered at ``width`` points with ``font_pt`` text, and without its
    legend unless ``legend``, when its specification exists, otherwise the
    first page of ``pdf`` itself.
    """
    spec = pdf.with_suffix(".json")
    if not spec.exists():
        if not legend:
            raise ValueError(f"{pdf} has no plotly specification beside it, so its legend cannot be removed")
        return PdfReader(pdf).pages[0]
    fig = pio.read_json(spec)
    if not legend:
        fig.update_layout(showlegend=False)
    return render_at_print_size(fig, width, max_height, font_pt, height, aspect, plot_height)


def render_at_print_size(
    fig: go.Figure, width: float, max_height: float | None, font_pt: float, height: float | None = None,
    aspect: float | None = None,
    plot_height: float | None = None,
) -> PageObject:
    """
    ``fig`` as a PDF page that, scaled to ``width`` points wide, shows its
    base font at ``font_pt``. The page is drawn larger by the ratio of the
    figure's font size to ``font_pt``, with every pixel size the figure
    gives left as it is, so the caller's scaling brings the text to
    ``font_pt``.

    A narrow cell leaves less room beside the text than the screen did, and
    keeping the whole canvas's proportions would flatten the data region:
    the margins hold text of a fixed size and so take a larger share of the
    width than of the height. Instead the data region keeps the shape it had
    on screen, and the page is as tall as that region plus the margins --
    or taller, where a y axis would otherwise be shorter than its title.
    ``aspect`` replaces the screen shape with that height over width,
    ``height`` overrides both, and ``max_height`` caps any of them.
    ``plot_height`` (in points, like ``width``) instead fixes the data
    region's height, measured as drawn. Scatter
    markers shrink with the data region (see the comment where they do).
    """
    layout = fig.to_dict()["layout"]
    template_layout = (layout.get("template") or {}).get("layout", {})
    base_font = (
        (layout.get("font") or {}).get("size")
        or (template_layout.get("font") or {}).get("size")
        or PLOTLY_DEFAULT_FONT
    )
    enlargement = base_font / font_pt
    canvas_width = width * enlargement

    original_width = layout.get("width") or PLOTLY_DEFAULT_WIDTH
    original_height = layout.get("height") or PLOTLY_DEFAULT_HEIGHT
    margin = {**PLOTLY_DEFAULT_MARGIN, **(template_layout.get("margin") or {}), **(layout.get("margin") or {})}
    legend_width, legend_height = _side_legend_px(fig, base_font)
    across, down = margin["l"] + margin["r"] + legend_width, margin["t"] + margin["b"]
    requested_aspect = aspect
    # The measured region takes this share of the height the margins leave, so
    # a change of canvas height reaches it in that proportion: a facet of two
    # rows grows by less than half of what the canvas does
    bottom, top = (layout.get("yaxis") or {}).get("domain") or (0.0, 1.0)
    share = top - bottom
    if height is None:
        if aspect is None:
            aspect = (original_height - down) / (original_width - across)
        # A cell too narrow for the margins alone still gets a data region
        data_width = max(canvas_width - across, MIN_DATA_PX)
        data_height = max(data_width * aspect, _y_title_px(layout, template_layout, base_font))
        height = (down + max(data_height, legend_height)) / enlargement
        if max_height is not None and height > max_height:
            # Too tall for its cell: narrow the panel, so the data region
            # keeps its shape, and let the cell center it
            data_width = max((max_height * enlargement - down) / aspect, MIN_DATA_PX)
            canvas_width = across + data_width
            height = max_height
        if requested_aspect is not None:
            # Margins grow to fit a legend or a long axis title, so the region
            # drawn is not the one estimated: measure it and correct the height
            drawn_width, drawn_height = _data_region_px(fig, canvas_width, height * enlargement)
            region_height = drawn_width * requested_aspect
            # A side legend taller than the region hangs below it, into a
            # bottom margin grown to hold it
            overhang = max(legend_height - region_height, 0.0)
            if overhang:
                # Plotly otherwise caps a vertical legend at the region's height
                fig = go.Figure(fig).update_layout(margin_b=margin["b"] + overhang, legend_maxheight=legend_height)
            height = (height * enlargement + (region_height - drawn_height) / share + overhang) / enlargement
            if max_height is not None:
                height = min(height, max_height)
    elif max_height is not None:
        height = min(height, max_height)
    if plot_height is not None:
        drawn_height = _data_region_px(fig, canvas_width, height * enlargement)[1]
        height += (plot_height * enlargement - drawn_height) / share / enlargement

    # Text keeps its size relative to the canvas and the data region shrinks,
    # so markers drawn at their screen size crowd it; they shrink with the
    # square root of the region, enough to part them while staying visible
    shrink = (max(canvas_width - across, MIN_DATA_PX) / (original_width - across)) ** 0.5
    if shrink < 1:
        # Legend symbols would shrink with the markers they stand for
        fig = _scale_markers(go.Figure(fig), shrink).update_layout(legend_itemsizing="constant")

    page = go.Figure(fig).update_layout(width=canvas_width, height=height * enlargement)
    return PdfReader(io.BytesIO(page.to_image(format="pdf"))).pages[0]


def _data_region_px(fig: go.Figure, width: float, height: float) -> tuple:
    """The width and height of ``fig``'s main data region, in pixels, as drawn at ``width`` by ``height``."""
    svg = fig.to_image(format="svg", width=width, height=height).decode()
    # The main subplot's clip rectangle is exactly its data region
    region = re.search(r'<clipPath id="clip[^"]*xyplot"[^>]*>\s*<rect[^>]*>', svg)
    if region is None:
        raise ValueError("The figure has no xy subplot to measure")
    size = dict(re.findall(r'\b(width|height)="([0-9.]+)"', region.group()))
    return float(size["width"]), float(size["height"])


def _y_title_px(layout: dict, template_layout: dict, base_font: float) -> float:
    """
    How tall a data region must be, in the figure's own pixels, for every
    y axis to be at least as long as its title: an axis taking a fraction of
    the region's height needs the title's length over that fraction. Titles
    drawn as rotated annotations, such as one spanning several facets, need
    the whole height.
    """
    template_size = (((template_layout.get("yaxis") or {}).get("title") or {}).get("font") or {}).get("size")
    needed = 0.0
    for key, axis in layout.items():
        if not key.startswith("yaxis") or not isinstance(axis, dict):
            continue
        title = axis.get("title") or {}
        size = (title.get("font") or {}).get("size") or template_size or base_font
        bottom, top = axis.get("domain") or (0.0, 1.0)
        needed = max(needed, _text_px(title.get("text"), size) / (top - bottom))
    for annotation in layout.get("annotations") or []:
        if abs(annotation.get("textangle") or 0) == 90:
            size = (annotation.get("font") or {}).get("size") or base_font
            needed = max(needed, _text_px(annotation.get("text"), size))
    return needed * AXIS_TITLE_PADDING


def _text_px(text: str | None, font_px: float) -> float:
    """Roughly how long ``text`` runs at ``font_px``: its longest line, with markup removed."""
    if not text:
        return 0.0
    lines = re.sub(r"<(?!br)[^>]*>", "", text).split("<br>")
    return CHARACTER_WIDTH * font_px * max(len(re.sub(r"<[^>]*>", "", line)) for line in lines)


def _scale_markers(fig: go.Figure, factor: float) -> go.Figure:
    """``fig`` with every scatter marker's diameter multiplied by ``factor``."""
    for trace in fig.data:
        if trace.type not in ("scatter", "scattergl") or "markers" not in (trace.mode or "markers"):
            continue
        marker = trace.marker
        if marker.sizeref is not None:
            # Sized by a column: an area scale needs the square of the factor
            # The smallest markers keep their size, chosen to stay visible:
            # only the large ones crowd a small region
            marker.sizeref = marker.sizeref / (factor ** 2 if marker.sizemode == "area" else factor)
        elif marker.size is None:
            marker.size = max(PLOTLY_DEFAULT_MARKER_PX * factor, MIN_PRINT_MARKER_PX)
        elif isinstance(marker.size, (int, float)):
            marker.size = max(marker.size * factor, MIN_PRINT_MARKER_PX)
        else:
            marker.size = [max(size * factor, MIN_PRINT_MARKER_PX) for size in marker.size]
    return fig


def _side_legend_px(fig: go.Figure, font_px: float) -> tuple:
    """
    Roughly how wide and how tall a legend down the right margin is, in the
    figure's own pixels. Plotly widens the margin to fit it, so its width is
    not in the margin the figure declares, and it hangs from the top of the
    data region, so that region must be at least as tall. Zero for a
    horizontal or hidden legend.
    """
    layout = fig.layout
    if layout.showlegend is False or layout.legend.orientation == "h":
        return 0.0, 0.0
    names = list(dict.fromkeys(
        str(trace.name) for trace in fig.data if trace.name and trace.showlegend is not False
    ))
    if not names:
        return 0.0, 0.0
    title = layout.legend.title.text or ""
    longest = max(len(text) for text in names + [title])
    # A symbol and padding, then the text at about 0.55 of the font size per character
    width = LEGEND_SYMBOL_PX + 0.55 * font_px * longest
    height = LEGEND_ENTRY_PX * len(names) + (LEGEND_ENTRY_PX if title else 0)
    return width, height


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


def _move_patterns(page: PageObject, ctm: tuple) -> None:
    """
    Carry ``page``'s patterns along with a transformation of its content. A
    pattern -- a color bar's gradient -- is placed in the page's own
    coordinates, not the content's, so merging moves the drawing and leaves
    the gradient behind, showing a slice of it or one flat color.
    """
    patterns = (page.get("/Resources") or {}).get("/Pattern")
    if patterns is None:
        return
    for pattern in patterns.get_object().values():
        pattern = pattern.get_object()
        a, b, c, d, e, f = [float(v) for v in pattern.get("/Matrix", [1, 0, 0, 1, 0, 0])]
        A, B, C, D, E, F = ctm
        # The pattern's matrix followed by the content's, rows as PDF writes them
        pattern[NameObject("/Matrix")] = ArrayObject(FloatObject(v) for v in (
            a * A + b * C, a * B + b * D,
            c * A + d * C, c * B + d * D,
            e * A + f * C + E, e * B + f * D + F,
        ))


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
