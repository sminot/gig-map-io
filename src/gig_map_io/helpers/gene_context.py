"""
A bin's genes in the context of the genomes that carry them.

The gene map shows the genes of one bin on a single line. The context map
puts that line over one row per genome: the stretch of the contig carrying
the bin, with every gene on it drawn as an arrow, so that what surrounds the
bin -- and whether that neighbourhood is the same from genome to genome --
can be read off directly.

Genes are coloured by identity: the same gene is the same colour in every
row, so conserved synteny appears as columns of matching colour, and genes
found in only one of the rows shown are grey. The bin's own genes carry a
black outline.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from .clustering import linkage_order

#: Twenty hues for the genes shared between rows, assigned by position so
#: that neighbouring genes differ; genes unique to one row are grey
SHARED_COLORS = [plt.get_cmap("tab20")(i) for i in range(20)]
UNIQUE_COLOR = "#d9d9d9"
BIN_OUTLINE = "#111111"

#: Row height and the arrow's share of it, in inches; the title band above
#: the labels; and how tall one character of upright label is per point of
#: font size (an inch per 72 points, less for a typical glyph)
ROW_HEIGHT_IN = 0.32
TITLE_HEIGHT_IN = 0.4
LABEL_INCH_PER_POINT_CHAR = 0.0085
#: Upright labels are spaced this many font sizes apart
LABEL_PITCH = 1.3


def fit_labels(n: int, font_size: float, width_pt: float) -> tuple[float, float]:
    """
    The font size and spacing (both in points) for ``n`` upright labels in a
    panel ``width_pt`` wide: the spacing follows the font size, and the font
    shrinks when the labels would otherwise overflow the panel.
    """
    font = min(font_size, width_pt / (n * LABEL_PITCH))
    return font, font * LABEL_PITCH
ARROW_HEIGHT = 0.62
#: An arrowhead is this long, or two fifths of the gene if that is shorter
ARROWHEAD_BP = 250


def draw_gene_labels(
    ax: plt.Axes,
    genes: pd.DataFrame,
    x0: float,
    x1: float,
    y: float,
    text_offset: float,
    font_size: float,
    arrow_height: float = 0.0,
) -> None:
    """
    Gene arrows along a line at ``y`` with each gene's label standing over it:
    labels sit at a fixed spacing (``LABEL_PITCH`` font sizes), centred over
    the genes, each tied to its gene by a leader line, which is what keeps
    them legible when genes are packed. The font shrinks if the labels would
    not otherwise fit between ``x0`` and ``x1``.

    ``genes`` has ``start``, ``stop`` and ``label`` columns in the axes' x
    units. ``text_offset`` is how far above the line the labels begin, in
    y units; ``arrow_height`` draws the arrows as filled blocks of that
    height (in y units) rather than as line arrows.
    """
    genes = genes.sort_values(["start", "label"]).reset_index(drop=True)
    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    font, pitch_pt = fit_labels(len(genes), font_size, width_pt)
    pitch = pitch_pt * (x1 - x0) / width_pt
    block = len(genes) * pitch
    centre = (genes[["start", "stop"]].min().min() + genes[["start", "stop"]].max().max()) / 2
    left = min(max(centre - block / 2, x0), x1 - block)
    for i, gene in genes.iterrows():
        if arrow_height > 0:
            ax.add_patch(gene_arrow(gene["start"], gene["stop"], y, arrow_height, "#555555", "#333333", 0.6))
        else:
            ax.annotate("", xytext=(gene["start"], y), xy=(gene["stop"], y), arrowprops=dict(arrowstyle="simple"))
        label_x = left + (i + 0.5) * pitch
        ax.annotate(
            "", xytext=(label_x, y + text_offset), xy=(np.mean([gene["start"], gene["stop"]]), y + text_offset / 5),
            arrowprops=dict(arrowstyle="-", linewidth=0.6, color="#666666"),
        )
        ax.text(label_x, y + text_offset, gene["label"], rotation=90, fontsize=font,
                horizontalalignment="center", verticalalignment="bottom")


def gene_arrow(start: float, stop: float, y: float, height: float, face, edge, edge_width: float) -> Polygon:
    """A gene as a block arrow from ``start`` to ``stop``, pointing in the direction of transcription."""
    length = abs(stop - start)
    head = min(ARROWHEAD_BP, 0.4 * length)
    direction = 1 if stop >= start else -1
    tip = stop
    shoulder = stop - direction * head
    half = height / 2
    points = [
        (start, y - half), (shoulder, y - half), (tip, y),
        (shoulder, y + half), (start, y + half),
    ]
    return Polygon(points, closed=True, facecolor=face, edgecolor=edge, linewidth=edge_width, zorder=3)


def order_rows(presence: pd.DataFrame) -> list:
    """
    Rows ordered by hierarchical clustering on which genes they carry, so
    that genomes with the same neighbourhood sit together.
    """
    return [presence.index[i] for i in linkage_order(presence.values.astype(bool), metric="jaccard", optimal=True)]


def plot_gene_context(
    bin_genes: pd.DataFrame,
    rows: Sequence[dict],
    window: tuple,
    title: str,
    width: float,
    font_size: float,
) -> plt.Figure:
    """
    Draw the context map.

    Parameters
    ----------
    bin_genes:
        The bin's genes with ``start``, ``stop`` (bp in the shared space) and
        ``label``.
    rows:
        One dict per genome: ``genome`` (display name), ``extent`` (the part
        of the contig inside the window, as (lo, hi)), and ``genes``, a
        DataFrame with ``gene``, ``start``, ``stop``, ``in_bin``.
    window:
        (lo, hi) of the region drawn, in bp.
    """
    lo, hi = window
    n_rows = len(rows)
    # The label panel is as tall as its longest label, which stands upright,
    # at the size the labels will actually be drawn
    label_font, _ = fit_labels(len(bin_genes), font_size, width * (0.98 - 0.16) * 72)
    label_height = 0.5 + bin_genes["label"].str.len().max() * label_font * LABEL_INCH_PER_POINT_CHAR
    rows_height = ROW_HEIGHT_IN * n_rows
    fig_height = TITLE_HEIGHT_IN + label_height + rows_height + 0.55 + 0.6
    fig = plt.figure(figsize=(width, fig_height))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[label_height, rows_height + 0.55], hspace=0.02,
        left=0.16, right=0.98, top=1 - TITLE_HEIGHT_IN / fig_height, bottom=0.6 / fig_height,
    )
    fig.text(0.16, 1 - 0.5 * TITLE_HEIGHT_IN / fig_height, title, fontsize=font_size + 3, weight="bold", va="center")
    top = fig.add_subplot(gs[0])
    main = fig.add_subplot(gs[1], sharex=top)

    # The bin alone, labelled, above the rows
    draw_gene_labels(
        top, bin_genes.assign(label=bin_genes["label"]), lo, hi, y=0.0,
        text_offset=0.08, font_size=font_size, arrow_height=0.06,
    )
    top.set_ylim(-0.06, 1.0)
    top.axis("off")

    # Colours by gene: shared genes take a hue in order of position
    everything = pd.concat([row["genes"].assign(row=i) for i, row in enumerate(rows)], ignore_index=True)
    counts = everything.groupby("gene")["row"].nunique()
    positions = everything.groupby("gene")["start"].median().sort_values()
    shared = [gene for gene in positions.index if counts[gene] > 1]
    colors = {gene: SHARED_COLORS[i % len(SHARED_COLORS)] for i, gene in enumerate(shared)}

    for i, row in enumerate(rows):
        y = n_rows - 1 - i
        main.plot(row["extent"], [y, y], color="#8a8f98", linewidth=1.0, zorder=1, solid_capstyle="butt")
        for _, gene in row["genes"].iterrows():
            face = colors.get(gene["gene"], UNIQUE_COLOR)
            if gene["in_bin"]:
                main.add_patch(gene_arrow(gene["start"], gene["stop"], y, ARROW_HEIGHT, face, BIN_OUTLINE, 1.1))
            else:
                main.add_patch(gene_arrow(gene["start"], gene["stop"], y, ARROW_HEIGHT, face, "#ffffff", 0.4))
    main.set_yticks(range(n_rows))
    main.set_yticklabels([row["genome"] for row in reversed(rows)], fontsize=font_size - 1)
    main.set_ylim(-0.7, n_rows - 0.3)
    main.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        main.spines[side].set_visible(False)

    # A kb scale along the bottom, counted from the start of the window
    ticks = np.arange(lo, hi + 1, _tick_step(hi - lo))
    main.set_xlim(lo, hi)
    main.set_xticks(ticks)
    main.set_xticklabels([f"{(t - lo) / 1000:g}" for t in ticks], fontsize=font_size - 1)
    main.set_xlabel("kb", fontsize=font_size)
    fig.text(
        0.98, 0.012, "Same colour: same gene.  Outlined: in the bin.  Grey: found in one of these genomes only.",
        ha="right", va="bottom", fontsize=font_size - 1, color="#555555",
    )
    return fig


def _tick_step(span: float) -> float:
    """A tick spacing in bp that gives at most eight ticks."""
    for step in (1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000):
        if span / step <= 8:
            return step
    return 1000000
