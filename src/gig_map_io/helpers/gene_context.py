"""
A bin's genes in the context of the genomes that carry them.

The gene map shows the genes of one bin on a single line. The context map
puts that line over one row per genome: the stretch of the contig carrying
the bin, with every gene on it drawn as an arrow, so that what surrounds the
bin -- and whether that neighborhood is the same from genome to genome --
can be read off directly.

Genes are colored by identity: the same gene is the same color in every
row, so conserved synteny appears as columns of matching color, and genes
found in only one of the rows shown are gray. The bin's own genes carry a
black outline.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Polygon

from .clustering import linkage_order

#: Twenty hues for the genes shared between rows, assigned by position so
#: that neighboring genes differ; genes unique to one row are gray
SHARED_COLORS = [plt.get_cmap("tab20")(i) for i in range(20)]
UNIQUE_COLOR = "#d9d9d9"

#: The key to the colors, printed beneath every map
KEY = "Same color: same gene.  Outlined: in the bin.  Gray: found in one of these genomes only."

#: What an arc over a row means, printed when there are any
LINK_KEY = "Arc: one contig, its pieces joined in the order they really run, which differs from the other rows."

#: What the rows below the dashed line are, printed when there are any
WITHOUT_BIN_KEY = "Below the dashed line: genomes with the bin's neighbors but not the bin, broken where it would sit."
BIN_OUTLINE = "#111111"

#: Row height and the arrow's share of it, in inches; the space above the
#: labels; the room the kb axis takes below the rows; and the footnote's
#: distance from the bottom edge
ROW_HEIGHT_IN = 0.22
TOP_PAD_IN = 0.05
AXIS_IN = 0.4
FOOTNOTE_PAD_IN = 0.03
#: The bin's own line above the rows, in inches: its arrows' height, how far
#: above it the labels begin, and the room below it
LABEL_ARROW_IN = 0.16
LABEL_OFFSET_IN = 0.3
LABEL_BASE_IN = 0.1
#: Gap between the row labels and the rows, in inches
ROW_LABEL_PAD_IN = 0.08
#: Upright labels are spaced this many font sizes apart
LABEL_PITCH = 1.3
#: Genes whose label starts with this go unlabeled when labeling every gene
#: would shrink the font: they say nothing about what the gene does
HYPOTHETICAL = "hypothetical protein"


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
) -> int:
    """
    Gene arrows along a line at ``y`` with each gene's label standing over it:
    labels sit at a fixed spacing (``LABEL_PITCH`` font sizes), centered over
    the genes, each tied to its gene by a leader line, which is what keeps
    them legible when genes are packed. If the labels would not fit between
    ``x0`` and ``x1`` at ``font_size``, the hypothetical proteins go
    unlabeled, and if the rest still do not fit the font shrinks.

    ``genes`` has ``start``, ``stop`` and ``label`` columns in the axes' x
    units. ``text_offset`` is how far above the line the labels begin, in
    y units; ``arrow_height`` draws the arrows as filled blocks of that
    height (in y units) rather than as line arrows. Returns how many genes
    were left unlabeled.
    """
    genes = genes.sort_values(["start", "label"]).reset_index(drop=True)
    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    labeled = genes
    if fit_labels(len(genes), font_size, width_pt)[0] < font_size:
        labeled = genes.loc[~genes["label"].str.startswith(HYPOTHETICAL)]
    font, pitch_pt = fit_labels(len(labeled), font_size, width_pt)
    pitch = pitch_pt * (x1 - x0) / width_pt
    block = len(labeled) * pitch
    center = (genes[["start", "stop"]].min().min() + genes[["start", "stop"]].max().max()) / 2
    left = min(max(center - block / 2, x0), x1 - block)
    for _, gene in genes.iterrows():
        if arrow_height > 0:
            ax.add_patch(gene_arrow(gene["start"], gene["stop"], y, arrow_height, "#555555", "#333333", 0.6))
        else:
            ax.annotate("", xytext=(gene["start"], y), xy=(gene["stop"], y), arrowprops=dict(arrowstyle="simple"))
    for i, (_, gene) in enumerate(labeled.iterrows()):
        label_x = left + (i + 0.5) * pitch
        ax.annotate(
            "", xytext=(label_x, y + text_offset), xy=(np.mean([gene["start"], gene["stop"]]), y + text_offset / 5),
            arrowprops=dict(arrowstyle="-", linewidth=0.6, color="#666666"),
        )
        ax.text(label_x, y + text_offset, gene["label"], rotation=90, fontsize=font,
                horizontalalignment="center", verticalalignment="bottom")
    return len(genes) - len(labeled)


def unlabeled_note(n_unlabeled: int) -> str:
    """The footnote line for genes ``draw_gene_labels`` left unlabeled."""
    return f"Unlabeled: {n_unlabeled} hypothetical protein{'s' if n_unlabeled != 1 else ''}."


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


def link_arc(x_from: float, x_to: float, y: float) -> PathPatch:
    """
    A curve over a row from one stretch of contig to the next, rising to
    the gap between rows, for a contig drawn in pieces.
    """
    base, peak = y + ARROW_HEIGHT / 2, y + 0.5
    path = Path(
        [(x_from, base), (x_from, peak), (x_to, peak), (x_to, base)],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
    )
    return PathPatch(path, facecolor="none", edgecolor="#333333", linewidth=1.0, zorder=3)


def order_rows(presence: pd.DataFrame) -> list:
    """
    Rows ordered by hierarchical clustering on which genes they carry, so
    that genomes with the same neighborhood sit together.
    """
    return [presence.index[i] for i in linkage_order(presence.values.astype(bool), metric="jaccard", optimal=True)]


def plot_gene_context(
    bin_genes: pd.DataFrame,
    rows: Sequence[dict],
    window: tuple,
    width: float,
    font_size: float,
    note: str | None = None,
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
        DataFrame with ``gene``, ``start``, ``stop``, ``in_bin``. A row may
        carry ``segments``, the stretches of contig to draw where it is not
        continuous, ``links`` joining those stretches in the contig's own
        order (drawn as arcs above the row), and ``without_bin`` for a genome
        shown for context that lacks the bin; those come last, below a dashed
        line.
    window:
        (lo, hi) of the region drawn, in bp.
    note:
        A line for the footnote above the key to the colors, such as how many
        of the bin's genes lie outside the window. Like every figure here
        the map has no title.
    """
    lo, hi = window
    n_rows = len(rows)
    rows_height = ROW_HEIGHT_IN * n_rows + 0.55
    # The height is a placeholder until the labels are drawn and measured
    fig = plt.figure(figsize=(width, 1.0))
    gs = fig.add_gridspec(2, 1, hspace=0, right=0.98)
    top = fig.add_subplot(gs[0])
    main = fig.add_subplot(gs[1], sharex=top)

    # Colors by gene: shared genes take a hue in order of position
    everything = pd.concat([row["genes"].assign(row=i) for i, row in enumerate(rows)], ignore_index=True)
    counts = everything.groupby("gene")["row"].nunique()
    positions = everything.groupby("gene")["start"].median().sort_values()
    shared = [gene for gene in positions.index if counts[gene] > 1]
    colors = {gene: SHARED_COLORS[i % len(SHARED_COLORS)] for i, gene in enumerate(shared)}

    for i, row in enumerate(rows):
        y = n_rows - 1 - i
        if row.get("without_bin") and not rows[i - 1].get("without_bin"):
            main.axhline(y + 0.5, color="#8a8f98", linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
        for segment in row.get("segments", [row["extent"]]):
            main.plot(segment, [y, y], color="#8a8f98", linewidth=1.0, zorder=1, solid_capstyle="butt")
        for x_from, x_to in row.get("links", []):
            main.add_patch(link_arc(x_from, x_to, y))
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

    # The left margin holds the longest row label, which sets the width the
    # bin's labels have to fit in
    renderer = fig.canvas.get_renderer()
    row_label_in = max(label.get_window_extent(renderer).width for label in main.get_yticklabels()) / fig.dpi
    gs.update(left=(row_label_in + ROW_LABEL_PAD_IN) / width)

    # The bin alone, labeled, above the rows, its y units inches; the panel
    # is as tall as the longest label drawn
    n_unlabeled = draw_gene_labels(
        top, bin_genes, lo, hi, y=0.0,
        text_offset=LABEL_OFFSET_IN, font_size=font_size, arrow_height=LABEL_ARROW_IN,
    )
    label_in = max(text.get_window_extent(renderer).height for text in top.texts) / fig.dpi
    top.set_ylim(-LABEL_BASE_IN, LABEL_OFFSET_IN + label_in)
    top.axis("off")
    label_height = LABEL_BASE_IN + LABEL_OFFSET_IN + label_in

    key = KEY
    if any(row.get("without_bin") for row in rows):
        key = f"{WITHOUT_BIN_KEY}\n{key}"
    if any(row.get("links") for row in rows):
        key = f"{LINK_KEY}\n{key}"
    notes = [note] if note else []
    if n_unlabeled:
        notes.append(unlabeled_note(n_unlabeled))
    footnote = fig.text(
        0.98, 0, "\n".join(notes + [key]), ha="right", va="bottom", fontsize=font_size - 1, color="#555555",
    )
    bottom_in = AXIS_IN + FOOTNOTE_PAD_IN + footnote.get_window_extent(renderer).height / fig.dpi
    fig_height = TOP_PAD_IN + label_height + rows_height + bottom_in
    fig.set_figheight(fig_height)
    footnote.set_y(FOOTNOTE_PAD_IN / fig_height)
    gs.set_height_ratios([label_height, rows_height])
    gs.update(top=1 - TOP_PAD_IN / fig_height, bottom=bottom_in / fig_height)
    return fig


def _tick_step(span: float) -> float:
    """A tick spacing in bp that gives at most eight ticks."""
    for step in (1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000):
        if span / step <= 8:
            return step
    return 1000000
