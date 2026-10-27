"""
Pangenome base class for gig-map-io.
"""

from functools import cached_property
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List
from Bio import Phylo
import pandas as pd
import numpy as np
from plotly import graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from plotly.subplots import make_subplots

from ..helpers.clustering import linkage_order
from ..helpers.save_image import save_image
from ..helpers.style import PRIMARY, SIMPLE_TEMPLATE, TEMPLATE
from ..helpers.coords import Coords
from ..helpers.gene_context import draw_gene_labels, order_rows, plot_gene_context, unlabeled_note
from ..helpers.genomes import genbank_duplicates
from .dataset import Dataset
from ..helpers.phylogeny import Phylogeny

logger = logging.getLogger(__name__)

#: The log2 odds ratios beyond these are drawn in the end boxes. The upper
#: edge is high because the pairs that sit together on a contig only stand
#: apart from the merely common ones above about 15.
LOG_ODDS_BINS = (-4, 16)

#: Width of a log2 odds ratio box
LOG_ODDS_STEP = 2

#: How gene co-occurrence can be scored against distance: the column of
#: calc_membership_vs_distance, its axis title, and how it is binned
MEMBERSHIP_METRICS = {
    "jaccard": ("membership", "Shared genomes (Jaccard)", lambda x: x.round(1)),
    "log_odds": (
        "log2_odds_ratio",
        "Co-occurrence across genomes (log2 odds ratio)",
        lambda x: (x / LOG_ODDS_STEP).round().mul(LOG_ODDS_STEP).clip(*LOG_ODDS_BINS),
    ),
}


#: A contig's bin genes stay in one collinear piece while each lands within
#: this of where the previous one put the rest: the larger of a fixed
#: allowance for small indels and a share of the bin's span
PIECE_TOLERANCE_BP = 2000
PIECE_TOLERANCE_FRACTION = 0.05

#: The widest a gene or context map is drawn, in inches: a full-width cell of
#: a 170 mm figure, so that a map drawn at the printed font size places 1:1
MAP_MAX_WIDTH_IN = 6.5
#: A gene map's width per gene drawn, and what it needs besides, in inches
GENE_MAP_IN_PER_GENE = 0.2
GENE_MAP_BASE_IN = 0.6


def _largest_cluster(genes: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """The longest run of genes, by count, in which no two neighbors are more than ``max_gap`` apart."""
    ordered = genes.sort_values("start")
    starts = ordered["start"].values
    breaks = np.flatnonzero(np.diff(starts) > max_gap) + 1
    runs = np.split(np.arange(len(ordered)), breaks)
    best = max(runs, key=lambda run: (len(run), -run[0]))
    return ordered.iloc[best]


def _fit_to_reference(reference: pd.DataFrame, contig: pd.DataFrame):
    """
    The orientation and shift that land a contig's genes closest to where
    ``reference`` puts them, as (sign, offset): a contig coordinate ``x``
    maps to ``sign * x + offset``. ``reference`` has a median ``start`` and
    ``stop`` per gene; ``contig`` has each gene's ``qstart`` and ``qend``.
    """
    shared = contig.join(reference, how="inner")
    best = None
    for sign in (1, -1):
        offsets = np.concatenate([
            shared["start"] - sign * shared["qstart"], shared["stop"] - sign * shared["qend"],
        ])
        fit = (float(np.std(offsets)), sign, float(np.median(offsets)))
        if best is None or fit[0] < best[0]:
            best = fit
    return best[1], best[2]


def _collinear_pieces(anchors: pd.DataFrame, consensus: pd.DataFrame, tolerance: float, min_genes: int = 3) -> tuple:
    """
    A contig's bin genes split into runs that each sit in the consensus order,
    for a contig that carries the bin rearranged -- a circular permutation
    of a prophage, say. Returns (sign, pieces): the contig's orientation
    against the consensus, and per piece in contig order its ``offset`` (a
    contig coordinate ``x`` lands at ``sign * x + offset``) and the contig
    interval ``lo``..``hi`` its genes span.

    Walking the genes in contig order, a gene joins the current piece while
    its offset stays within ``tolerance`` of the previous gene's; runs of
    fewer than ``min_genes`` are folded into the piece before them, so a
    stray homolog does not make a piece of its own. The orientation giving
    the fewest pieces wins, and between equals the one with fewer runs before
    folding.
    """
    shared = anchors[["qstart", "qend"]].join(consensus[["start", "stop"]], how="inner")
    shared = shared.assign(at=(shared["qstart"] + shared["qend"]) / 2).sort_values(["at", "qstart"])
    target = ((shared["start"] + shared["stop"]) / 2).values
    at = shared["at"].values
    best = None
    for sign in (1, -1):
        offsets = target - sign * at
        runs = [[0]]
        for i in range(1, len(offsets)):
            if abs(offsets[i] - offsets[runs[-1][-1]]) <= tolerance:
                runs[-1].append(i)
            else:
                runs.append([i])
        merged = []
        for run in runs:
            if merged and len(run) < min_genes:
                merged[-1].extend(run)
            else:
                merged.append(list(run))
        if len(merged) > 1 and len(merged[0]) < min_genes:
            merged[1] = merged[0] + merged[1]
            merged = merged[1:]
        pieces = [
            dict(offset=float(np.median(offsets[run])), lo=float(at[run].min()), hi=float(at[run].max()))
            for run in merged
        ]
        # Folding can leave both orientations a single piece; the right one is
        # the one whose genes already ran together before any folding
        score = (len(pieces), len(runs))
        if best is None or score < best[0]:
            best = (score, sign, pieces)
    return best[1], best[2]


def _in_consensus_order(sign: int, pieces: List[dict]) -> bool:
    """
    Whether a contig's pieces run in the consensus order, so that it differs
    from the consensus only by an insertion or deletion between them and
    can be placed in one piece like any other. Pieces that run out of order
    are a rearrangement.
    """
    ordered = sorted(pieces, key=lambda piece: piece["lo"])
    centers = [sign * (piece["lo"] + piece["hi"]) / 2 + piece["offset"] for piece in ordered]
    # In contig order, a collinear contig's pieces climb the consensus if it
    # runs the same way and descend if it is reversed; one piece is trivially in order
    return bool(np.all(sign * np.diff(centers) > 0))


def _place_in_pieces(on_contig: pd.DataFrame, pieces: List[dict]) -> tuple:
    """
    Every gene of a rearranged contig placed by the piece it falls in, as in
    :func:`_collinear_pieces`. Returns the genes, the stretch each piece
    covers, and the links between consecutive pieces in contig order (from
    the end of one to the start of the next), which the map draws as arcs.
    Genes beyond the outermost bin genes are left out: past a rearranged
    end the contig no longer follows the consensus at all.
    """
    pieces = sorted(pieces, key=lambda piece: piece["lo"])
    bounds = [pieces[0]["lo"]] + [(a["hi"] + b["lo"]) / 2 for a, b in zip(pieces, pieces[1:])] + [pieces[-1]["hi"]]
    at = ((on_contig["qstart"] + on_contig["qend"]) / 2).values
    index = np.searchsorted(bounds, at, side="right") - 1
    inside = (at >= bounds[0]) & (at <= bounds[-1])
    index = np.clip(index, 0, len(pieces) - 1)

    def place(x, piece):
        return piece["sign"] * x + piece["offset"]

    kept = on_contig.loc[inside]
    owners = [pieces[i] for i in index[inside]]
    genes = pd.DataFrame({
        "gene": kept["sseqid"].values,
        "start": [place(x, piece) for x, piece in zip(kept["qstart"], owners)],
        "stop": [place(x, piece) for x, piece in zip(kept["qend"], owners)],
        "bin": kept["bin"].values,
    })
    segments = [tuple(sorted(place(x, piece) for x in (lo, hi)))
                for piece, lo, hi in zip(pieces, bounds[:-1], bounds[1:])]
    links = [(place(a["hi"], a), place(b["lo"], b)) for a, b in zip(pieces, pieces[1:])]
    return genes, segments, links


def _mean_jaccard(genes: set, others: List[set]) -> float:
    """How much ``genes`` shares with each of ``others`` on average, as Jaccard similarity."""
    return float(np.mean([len(genes & other) / len(genes | other) for other in others if genes | other]))


def _short_genome_name(genome: str) -> str:
    """An assembly file name reduced to its accession and strain, e.g. GCF_000210575.1 ASM21057v1."""
    name = genome.replace("_genomic.fna.gz", "").replace(".fna.gz", "").replace(".fasta", "")
    parts = name.split("_", 2)
    if len(parts) == 3 and parts[0] in ("GCA", "GCF"):
        return f"{parts[0]}_{parts[1]} {parts[2]}"[:34]
    return name[:34]


class Pangenome(Dataset):
    """
    Representation of a pangenome generated by gig-map.

    This is a minimal base class; concrete subclasses or helpers can add
    convenience methods for common gig-map outputs (e.g., gene presence/absence,
    annotations, etc.).

    DataFrame attributes are cached after first access to avoid reloading
    CSV files on subsequent accesses.
    """

    def __repr__(self) -> str:
        return f"Pangenome(directory={self.directory})"

    @cached_property
    def gene_bins(self) -> pd.DataFrame:
        """
        Gene bins DataFrame loaded from bin_pangenome/gene_bins.csv.

        Returns
        -------
        DataFrame containing gene bin assignments.
        """
        path = self.directory / "bin_pangenome" / "gene_bins.csv"
        return (
            pd.read_csv(path)
            .assign(
                gene_label=lambda d: d.apply(
                    lambda r: f"{r['gene_id']} - {r['combined_name']}",
                    axis=1
                )
            )
        )

    @cached_property
    def ani_distances(self) -> pd.DataFrame:
        """
        ANI distances DataFrame loaded from distances.csv.gz.

        Returns
        -------
        DataFrame containing ANI distances.
        """
        path = self.directory / "distances.csv.gz"
        return pd.read_csv(path, index_col=0)

    @cached_property
    def align_genomes(self) -> pd.DataFrame:
        """
        Alignment of genomes from align/genomes.aln.csv.gz.
        """
        path = self.directory / "align" / "genomes.aln.csv.gz"
        df = pd.read_csv(path)

        # Add the bin label to the alignment
        df = df.assign(
            bin=df["sseqid"].apply(
                self.gene_bins.set_index("gene_id")["bin"].get
            )
        )
        return df

    @cached_property
    def bin_names(self) -> list[str]:
        """
        List of unique (non-null) bin names in gene_bins.
        """
        return self.gene_bins["bin"].dropna().unique().tolist()

    @cached_property
    def genome_content(self) -> pd.DataFrame:
        """
        Genome content DataFrame loaded from bin_pangenome/genome_content.long.csv.

        Returns
        -------
        DataFrame containing genome content data.
        """
        path = self.directory / "bin_pangenome" / "genome_content.long.csv"
        return pd.read_csv(path, low_memory=False)

    @cached_property
    def bin_presence_wide(self) -> pd.DataFrame:
        """
        Wide dataframe of bin presence/absence for each genome. Each column is a bin, each row is a genome.
        """
        return (
            self.genome_content
            .assign(present=(self.genome_content["prop_genes_detected"] >= 0.5).astype(int))
            .reindex(columns=["genome", "bin", "present"])
            .dropna()
            .drop_duplicates()
            .pivot(
                index="genome",
                columns="bin",
                values="present"
            )
            .fillna(0)
        )

    @cached_property
    def bin_size(self) -> pd.DataFrame:
        """
        Number of genes in each bin.
        """
        return self.gene_bins["bin"].value_counts()

    def core_genome(self, prop_genes_detected: float = 0.9) -> str:
        """
        The bin that best represents the core genome of this pangenome.

        Picks the bin detected near-completely (at least `prop_genes_detected`
        of its genes) in the largest number of genomes, breaking ties by the
        number of genes in the bin.
        """
        counts = (
            self.genome_content
            .query(f"prop_genes_detected >= {prop_genes_detected}")
            .groupby("bin")
            .size()
            .to_frame(name="n_genomes")
            .assign(n_genes=self.bin_size)
            .sort_values(by=["n_genomes", "n_genes"], ascending=False)
        )
        return counts.index.values[0]

    def bin_genome_heatmap(
        self,
        width: int = 500,
        height: int = 400,
        file_prefix: str | None = None
    ) -> go.Figure:
        """
        Heatmap of bin presence/absence for each genome.
        """

        # Perform hierarchical clustering on the rows and columns
        wide = self.bin_presence_wide
        wide_sorted = wide.iloc[linkage_order(wide.values, method="ward"), linkage_order(wide.values.T, method="ward")]

        # Set the edges of the bins so that the widths match the number of genes
        x = [0] + self.bin_size.reindex(wide_sorted.columns.values).cumsum().tolist()

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=wide_sorted.values,
                    x=x,
                    colorscale="blues",
                    showscale=False
                )
            ],
            layout=go.Layout(
                xaxis_title="Gene",
                yaxis_title="Genome",
                margin=dict(r=20),
                template=SIMPLE_TEMPLATE,
                width=width,
                height=height
            )
        )
        save_image(fig, file_prefix)

        return fig

    def calc_membership_vs_distance(
        self,
        min_n_genomes: int = 10,
        max_n_genomes: int = 100,
        n_genes: int = 100,
        max_distance: int = 1000000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        For a random subset of genes, how similar each pair's genome
        membership is, and how far apart the two typically sit in the genomes
        that carry both.

        The genes eligible are those found in between ``min_n_genomes`` and
        ``max_n_genomes`` genomes, and ``n_genes`` of them are drawn at random
        from that pool, with every pair of the drawn genes compared. The upper
        limit is what keeps the core genome out: genes carried by nearly
        every genome co-occur almost perfectly wherever they sit, and would
        swamp the relationship between co-occurrence and distance. The lower
        limit drops genes whose co-occurrence rests on too few genomes.

        Parameters
        ----------
        min_n_genomes: int
            Fewest genomes a gene may be found in and still be drawn.
        max_n_genomes: int
            Most genomes a gene may be found in and still be drawn.
        n_genes: int
            How many genes to draw from the eligible pool; numpy raises a
            ValueError if the pool holds fewer.
        max_distance: int
            Cap on a pair's distance in bp, and the distance given to a pair
            never found on the same contig.
        random_state: int
            Seed for the gene draw.

        Returns
        -------
        One row per pair of genes: ``membership``, the Jaccard similarity of
        the genomes carrying each; ``log2_odds_ratio``, the log2 odds of the
        two genes sharing a genome, from the 2x2 table over every genome in
        the pangenome with 0.5 added to each cell; and ``distance``, their
        mean separation on the contigs that carry both.
        """

        n_genomes_per_gene = (
            self.align_genomes
            .reindex(columns=["sseqid", "genome"])
            .drop_duplicates()
            ["sseqid"]
            .value_counts()
        )
        genes_passing_filter = n_genomes_per_gene.loc[lambda x: (x >= min_n_genomes) & (x <= max_n_genomes)].index.tolist()

        # Pick a random subset of the genes
        genes = np.random.default_rng(random_state).choice(
            genes_passing_filter, size=n_genes, replace=False
        )

        df = (
            self.align_genomes
            .loc[self.align_genomes["sseqid"].isin(genes)]
            .assign(
                gene_position=lambda d: (d["qstart"] + d["qend"]) / 2
            )
        )
        n_genomes = self.align_genomes["genome"].nunique()

        # Compare each pair of genes to compute the membership and distance
        output = []
        for gene1, df1 in df.groupby("sseqid"):
            for gene2, df2 in df.groupby("sseqid"):
                if gene1 <= gene2:
                    continue
                genomes1, genomes2 = set(df1["genome"]), set(df2["genome"])
                shared = len(genomes1 & genomes2)
                only1 = len(genomes1 - genomes2)
                only2 = len(genomes2 - genomes1)
                neither = n_genomes - shared - only1 - only2
                output.append(dict(
                    gene1=gene1,
                    gene2=gene2,
                    membership=shared / (shared + only1 + only2),
                    # 0.5 in every cell keeps the ratio finite for pairs never seen together
                    log2_odds_ratio=np.log2(
                        (shared + 0.5) * (neither + 0.5) / ((only1 + 0.5) * (only2 + 0.5))
                    ),
                    distance=self._compute_gene_distance(df1, df2, max_distance)
                ))

        return pd.DataFrame(output)

    def _compute_gene_distance(self, df1: pd.DataFrame, df2: pd.DataFrame, max_distance: int) -> float:
        """
        Compute the distance between two genes.
        """
        df = pd.concat([
            df1.assign(gene=lambda d: "gene1"),
            df2.assign(gene=lambda d: "gene2")
        ]).pivot(
            index=["genome", "qseqid"],
            columns="gene",
            values="gene_position"
        ).dropna()

        if df.shape[0] == 0:
            return max_distance
        else:
            return min(max_distance, np.abs(df["gene1"] - df["gene2"]).mean())

    def compare_membership_vs_distance(
        self,
        metric: str = "jaccard",
        n_genes: int = 100,
        min_n_genomes: int = 10,
        max_n_genomes: int = 100,
        width: int = 500,
        height: int = 400,
        random_state: int = 42,
        file_prefix: str | None = None,
        **kwargs: dict[str, Any]
    ) -> go.Figure:
        """
        For a random subset of genes, compute how similar they are in terms
        of their membership across all genomes, and then also compute their
        typical physical distance across those genomes (when they are both present).
        Then plot the relationship between these two metrics.

        ``metric`` is how membership is scored, and is a key of
        ``MEMBERSHIP_METRICS``: ``"jaccard"`` in steps of 0.1, or
        ``"log_odds"`` in steps of ``LOG_ODDS_STEP`` log2 units, the end
        boxes holding the tails.
        """
        column, axis_title, to_bins = MEMBERSHIP_METRICS[metric]
        df = (
            self.calc_membership_vs_distance(
                min_n_genomes, max_n_genomes, n_genes, random_state=random_state
            )
            .assign(
                distance_log10=lambda d: np.log10(d["distance"]),
                membership_bins=lambda d: to_bins(d[column])
            )
        )

        fig = px.box(
            data_frame=df,
            x="membership_bins",
            y="distance_log10",
            template=TEMPLATE,
            labels=dict(
                distance_log10="Mean distance between genes (bp)",
                membership_bins=axis_title
            ),
            width=width,
            height=height,
            **kwargs
        )
        fig.update_traces(marker=dict(size=4, opacity=0.6), line_width=1)
        if metric == "log_odds":
            ticks = list(range(LOG_ODDS_BINS[0], LOG_ODDS_BINS[1] + 1, LOG_ODDS_STEP))
            fig.update_xaxes(
                tickmode="array",
                tickvals=ticks,
                ticktext=[f"\u2264{t}" if t == LOG_ODDS_BINS[0] else f"\u2265{t}" if t == LOG_ODDS_BINS[1] else str(t)
                          for t in ticks],
            )
        fig.update_yaxes(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=["1", "10", "100", "1k", "10k", "100k", "1M"]
        )

        save_image(fig, file_prefix)
        return fig

    def bin_size_df(
        self,
        bins: int = 30,
    ) -> pd.DataFrame:
        """
        Histogram of bin sizes.
        """
        # Get the bin sizes
        vals = self.bin_size.copy().apply(np.log10)

        counts, bins = np.histogram(
            vals,
            bins=bins,
            weights=self.bin_size.copy()
        )

        bin_names = [
            (
                f"{10**bins[i-1]:.0f} - {10**bins[i]:.0f}"
            )
            for i in range(1, len(bins))
        ]
        bins = 0.5 * (bins[:-1] + bins[1:])

        return pd.DataFrame(dict(
            bin_size=bins,
            count=counts,
            bin_names=bin_names
        ))

    def bin_size_histogram(
        self,
        bins: int = 30,
        width: int = 500,
        height: int = 400,
        file_prefix: str | None = None
    ) -> go.Figure:

        df = self.bin_size_df(bins)

        fig = px.bar(
            data_frame=df,
            x="bin_size",
            y="count",
            labels=dict(
                bin_size="Pangenome bin size (genes)",
                count="Total gene content (genes)"
            ),
            template=TEMPLATE,
            hover_name="bin_names",
            width=width,
            height=height
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=["1", "10", "100", "1k", "10k", "100k"]
        )
        save_image(fig, file_prefix)

        return fig

    def rarefaction_curve_data(self, n_reps: int = 10, random_state: int = 42) -> pd.DataFrame:
        data = []
        for rep in range(n_reps):
            # Take the bin presence matrix
            bin_presence = self.bin_presence_wide.copy()
            # Shuffle the rows, in an order that is reproducible across runs
            bin_presence = bin_presence.sample(
                frac=1, random_state=random_state + rep
            ).reset_index(drop=True)
            # Compute the cumulative sum by column (bin)
            cs = bin_presence.cumsum()
            # For each row, count the number of genes recovered
            n_genes = (cs > 0) * self.bin_size
            data.append(pd.DataFrame(dict(
                n_genomes=list(range(1, bin_presence.shape[0] + 1)),
                n_genes=n_genes.sum(axis=1)
            )))
        return (
            pd.concat(data)
            .groupby("n_genomes")
            ["n_genes"]
            .describe()
            .reset_index()
        )

    def rarefaction_curve(
        self,
        n_reps: int = 10,
        width: int = 500,
        height: int = 400,
        random_state: int = 42,
        file_prefix: str | None = None
    ) -> go.Figure:
        """
        Rarefaction curve of the pangenome.

        Parameters
        ----------
        n_reps: int
            Number of rarefaction replicates to simulate.
        random_state: int
            Seed for the order genomes are accumulated in.

        Returns
        -------
        Plotly figure of the rarefaction curve.
        """
        # Simulate the number of genes recovered with different numbers of subsampled genomes
        rf = self.rarefaction_curve_data(n_reps, random_state=random_state)

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=rf["n_genomes"],
                    y=rf["50%"],
                    mode="lines",
                    line=dict(color=PRIMARY, width=2),
                    showlegend=False
                ),
                go.Scatter(
                    x=rf['n_genomes'],
                    y=rf['75%'],
                    mode='lines',
                    marker=dict(color=PRIMARY),
                    line=dict(width=0),
                    showlegend=False
                ),
                go.Scatter(
                    x=rf['n_genomes'],
                    y=rf['25%'],
                    marker=dict(color=PRIMARY),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(44, 111, 187, 0.25)',
                    fill='tonexty',
                    showlegend=False
                )
            ],
            layout=go.Layout(
                xaxis_title="Number of genomes",
                yaxis_title="Number of genes",
                yaxis_range=[0, None],
                template=TEMPLATE,
                width=width,
                height=height
            )
        )

        # Make the x axis log scale, ticked at the decades only
        fig.update_xaxes(
            type="log",
            dtick=1
        )

        save_image(fig, file_prefix)

        return fig

    def bin_gene_map(
        self,
        bin: str,
        width: float | None = None,
        height: float = 1.0,
        max_gap: int = 20000,
        min_bin_fraction: float = 0.5,
        remove_gene_id = True,
        remove_org_tag = True,
        text_offset = 0.025,
        font_size: float = 7.0,
        file_prefix: str | None = None
    ) -> plt.Figure:
        """
        Map of genes in a bin.

        ``width`` and ``height`` are in inches, and set the span of the arrows;
        the labels standing over them extend the drawing above that. The
        width defaults to ``GENE_MAP_IN_PER_GENE`` per gene drawn plus
        ``GENE_MAP_BASE_IN``, at most ``MAP_MAX_WIDTH_IN``, so that with
        ``font_size`` at the printed point size the map places into a figure
        unscaled.

        The genes are those of the context map's window, placed as it places
        them (``_bin_window``), and a footnote says how many of the bin's
        genes lie elsewhere. With a ``file_prefix`` the coordinates of the
        genes drawn are also written as CSV.
        """
        n_genes = int(self.bin_size[bin])
        *_, coords = self._bin_window(bin, max_gap, min_bin_fraction)

        # Customize the names
        coords = coords.assign(
            gene_label=coords['gene'].apply(self.gene_bins.set_index("gene_id")["gene_label"].get),
            label=lambda d: d['gene_label'].apply(
                lambda gene: self.customize_label(gene, remove_gene_id, remove_org_tag)
            )
        )

        max_x = np.max([coords['start'].max(), coords['stop'].max()])
        min_x = np.min([coords['start'].min(), coords['stop'].min()])
        span_x = max_x - min_x

        coords = coords.assign(
            start_x=lambda d: (d['start'] - min_x) / span_x,
            stop_x=lambda d: (d['stop'] - min_x) / span_x
        )

        if width is None:
            width = min(MAP_MAX_WIDTH_IN, GENE_MAP_IN_PER_GENE * len(coords) + GENE_MAP_BASE_IN)
        fig, ax = plt.subplots(figsize=(width, height))
        fig.subplots_adjust(left=0, right=1)
        n_unlabeled = draw_gene_labels(
            ax, coords[["start_x", "stop_x", "label"]].rename(columns={"start_x": "start", "stop_x": "stop"}),
            x0=0.0, x1=1.0, y=0.0, text_offset=text_offset, font_size=font_size,
            arrow_height=text_offset / 2,
        )
        ax.set_xlim(0, 1)
        # Room for the size label below the genes and the start of the labels above
        ax.set_ylim(-1.05 * text_offset, 1.2 * text_offset)

        # The vertial lines work with an increment scaled to the text offset
        vertical_increment = text_offset / 10.

        # Draw a line underneath the entire gene map
        ax.annotate(
            '',
            xytext=(0, -4 * vertical_increment),
            xy=(1, -4 * vertical_increment),
            arrowprops=dict(arrowstyle="|-|")
        )

        # Write the size underneath
        if span_x > 1000000:
            size_label = f"{span_x / 1000000:.1f} Mb"
        elif span_x > 1000:
            size_label = f"{span_x / 1000:.1f} Kb"
        else:
            size_label = f"{span_x:.0f} bp"
        ax.text(
            0.5,
            - (text_offset),
            size_label,
            fontsize=font_size,
            horizontalalignment='center',
            verticalalignment='bottom'
        )
        notes = []
        if len(coords) < n_genes:
            notes.append(f"{n_genes - len(coords)} of the bin's {n_genes} genes lie outside this map.")
        if n_unlabeled:
            notes.append(unlabeled_note(n_unlabeled))
        if notes:
            ax.text(0.5, -1.1 * text_offset, "\n".join(notes), fontsize=font_size - 1, color="#555555",
                    horizontalalignment="center", verticalalignment="top")

        ax.axis("off")

        save_image(fig, file_prefix)

        # Optionally save the relative gene coordinates as CSV
        if file_prefix is not None:
            coords.to_csv(file_prefix + ".csv", index=False)

        return fig

    def bin_context_map(
        self,
        bin: str,
        n_genomes: int = 10,
        flank: int = 8000,
        max_gap: int = 20000,
        min_bin_fraction: float = 0.5,
        max_without_bin: float = 1 / 3,
        min_flank_fraction: float = 0.5,
        width: float = MAP_MAX_WIDTH_IN,
        font_size: float = 7.0,
        remove_gene_id: bool = True,
        remove_org_tag: bool = True,
        genome_names: Dict[str, str] | None = None,
        file_prefix: str | None = None,
    ) -> plt.Figure:
        """
        The bin's genes with their neighborhood in the genomes that carry
        them: the gene map above one row per genome, each row the stretch of
        contig around the bin with every gene on it drawn as an arrow.

        Contigs carrying at least ``min_bin_fraction`` of the bin's genes are
        placed in one coordinate space, and the ``n_genomes`` whose contigs
        span most of the window -- the bin plus ``flank`` bp either side --
        are shown, one contig per genome, with the GenBank copy of an
        assembly also present as RefSeq dropped. Rows are ordered by
        clustering on the genes they carry. With a ``file_prefix`` the genes
        of every row are also written as CSV.

        A contig whose bin genes sit in a different order from the others --
        a prophage integrated at another point in its own sequence, so that
        its two halves are swapped -- is split into the pieces that each
        follow the consensus. Each piece is placed on its own, and an arc
        joins consecutive pieces in the contig's actual order.

        Below them, for context, come genomes that have the bin's
        neighborhood but not the bin, up to ``max_without_bin`` of all the
        rows shown. The neighborhood is the genes in the window outside the
        bin that at least half of the carrying rows share. A genome qualifies
        if it carries fewer than ``min_bin_fraction`` of the bin's genes and
        one of its contigs carries neighbors from both sides of the bin and
        at least ``min_flank_fraction`` of all of them; those whose genes
        overlap most with the carrying rows', bin genes aside, come first.
        Each side of such a contig is placed against the carrying rows'
        neighbors separately, since the bin's length is missing between
        them, and the row is drawn broken where the bin would sit.

        A bin's genes need not sit together: the window covers the largest
        run of them with no gap over ``max_gap`` bp (``_bin_window``, shared
        with the gene map), and the footnote says how many lie elsewhere.
        ``genome_names`` labels the rows (keyed by the
        genome's file name, as in ``helpers.ncbi.genome_names``); a genome
        it does not name is labeled by its accession.
        """
        aln = self.align_genomes
        n_bin_genes = int(self.bin_size[bin])
        in_bin, carrying, placed, bin_coords = self._bin_window(bin, max_gap, min_bin_fraction)
        consensus = placed.to_df().set_index("gene")
        n_elsewhere = n_bin_genes - len(bin_coords)
        bin_lo = float(min(bin_coords["start"].min(), bin_coords["stop"].min()))
        bin_hi = float(max(bin_coords["start"].max(), bin_coords["stop"].max()))
        window = (bin_lo - flank, bin_hi + flank)

        labels = self.gene_bins.set_index("gene_id")["gene_label"]
        bin_genes = bin_coords.assign(
            label=bin_coords["gene"].map(labels).map(
                lambda name: self.customize_label(name, remove_gene_id, remove_org_tag)
            )
        )

        # Place each carrying contig from its bin genes alone, which is enough
        # to know how much of the window it spans; the genomes whose contigs
        # span most of it are kept, one contig each
        # A contig whose bin genes break into pieces in a different order from
        # the consensus is kept and drawn piecewise, rather than forced onto
        # one line where none of it would line up
        # Only the bin genes in the window anchor a contig: one that lies
        # elsewhere sits megabases from the rest in the consensus, and as an
        # outlier it decides which orientation fits best
        tolerance = max(PIECE_TOLERANCE_BP, PIECE_TOLERANCE_FRACTION * (bin_hi - bin_lo))
        candidates = []
        for (genome, contig), n_carried in carrying.items():
            anchors = in_bin.loc[
                (in_bin["genome"] == genome) & (in_bin["qseqid"] == contig) & in_bin["sseqid"].isin(bin_coords["gene"])
            ].set_index("sseqid")
            if anchors.empty:
                continue
            sign, pieces = _collinear_pieces(anchors, consensus, tolerance)
            if _in_consensus_order(sign, pieces):
                transform = placed.fit(anchors["qstart"].to_dict(), anchors["qend"].to_dict())
                ends = sorted(transform(x) for x in (1, int(anchors["qlen"].iloc[0])))
                extent = (max(ends[0], window[0]), min(ends[1], window[1]))
                pieces = None
            else:
                for piece in pieces:
                    piece["sign"] = sign
                spans = [sorted(sign * x + piece["offset"] for x in (piece["lo"], piece["hi"])) for piece in pieces]
                extent = (max(min(lo for lo, _ in spans), window[0]), min(max(hi for _, hi in spans), window[1]))
                transform = None
            if extent[1] <= extent[0]:
                continue
            candidates.append({
                "genome": genome, "contig": contig, "n_carried": int(n_carried),
                "coverage": (extent[1] - extent[0]) / (window[1] - window[0]),
                "extent": extent, "transform": transform, "pieces": pieces,
            })
        candidates.sort(key=lambda row: (-row["coverage"], -row["n_carried"], row["genome"]))
        chosen, seen = [], set()
        for row in candidates:
            if row["genome"] not in seen:
                chosen.append(row); seen.add(row["genome"])
            if len(chosen) == n_genomes:
                break

        # Every gene on each chosen contig, mapped into the shared space
        on_contigs = aln.set_index(["genome", "qseqid"]).sort_index()
        for row in chosen:
            on_contig = on_contigs.loc[(row["genome"], row["contig"])].drop_duplicates("sseqid")
            transform, pieces = row.pop("transform"), row.pop("pieces")
            if pieces is None:
                genes = pd.DataFrame({
                    "gene": on_contig["sseqid"].values,
                    "start": on_contig["qstart"].map(transform).values,
                    "stop": on_contig["qend"].map(transform).values,
                    "bin": on_contig["bin"].values,
                })
            else:
                genes, row["segments"], row["links"] = _place_in_pieces(on_contig, pieces)
                row["rearranged"] = True
            genes = genes.loc[(genes[["start", "stop"]].max(axis=1) > window[0]) & (genes[["start", "stop"]].min(axis=1) < window[1])]
            row["genes"] = genes.assign(in_bin=genes["bin"] == bin).sort_values("start")

        presence = pd.DataFrame(
            {row["genome"]: {gene: 1 for gene in row["genes"]["gene"]} for row in chosen}
        ).T.fillna(0).sort_index()
        by_genome = {row["genome"]: row for row in chosen}
        ordered = [by_genome[genome] for genome in order_rows(presence)]
        # The most that keeps them within max_without_bin of all the rows; the
        # small allowance stops 10 * (1/3) / (2/3) rounding down to 4
        n_without = int(len(chosen) * max_without_bin / (1 - max_without_bin) + 1e-9)
        ordered += self._rows_without_bin(
            bin, chosen, window, (bin_lo, bin_hi), n_without, min_bin_fraction, min_flank_fraction,
        )
        for row in ordered:
            row["genome_label"] = (genome_names or {}).get(row["genome"]) or _short_genome_name(row["genome"])

        fig = plot_gene_context(
            bin_genes,
            [dict(row, genome=row["genome_label"]) for row in ordered],
            window,
            width=width,
            font_size=font_size,
            note=f"{n_elsewhere} of the bin's {n_bin_genes} genes lie outside this window." if n_elsewhere else None,
        )
        save_image(fig, file_prefix)
        if file_prefix is not None:
            pd.concat([
                row["genes"].assign(
                    genome=row["genome"], contig=row["contig"], carries_bin=not row.get("without_bin", False),
                    rearranged=row.get("rearranged", False),
                    name=lambda d: d["gene"].map(labels),
                )
                for row in ordered
            ]).to_csv(file_prefix + ".csv", index=False)
        return fig

    def _rows_without_bin(
        self,
        bin: str,
        carrying_rows: List[dict],
        window: tuple,
        bin_span: tuple,
        n_rows: int,
        min_bin_fraction: float,
        min_flank_fraction: float,
    ) -> List[dict]:
        """
        Up to ``n_rows`` context-map rows for genomes that have the bin's
        neighborhood but not the bin, placed in the carrying rows' space, as
        described in :meth:`bin_context_map`.
        """
        if n_rows < 1:
            return []
        bin_lo, bin_hi = bin_span
        around = pd.concat([row["genes"].loc[~row["genes"]["in_bin"]].assign(genome=row["genome"]) for row in carrying_rows])
        reference = around.groupby("gene").agg(
            start=("start", "median"), stop=("stop", "median"), n=("genome", "nunique"),
        )
        reference = reference.loc[reference["n"] >= len(carrying_rows) / 2]
        middle = (reference["start"] + reference["stop"]) / 2
        left, right = set(reference.index[middle < bin_lo]), set(reference.index[middle > bin_hi])
        if not left or not right:
            return []
        neighbors = left | right

        aln = self.align_genomes
        n_bin_genes = int(self.bin_size[bin])
        bin_per_genome = aln.loc[aln["bin"] == bin].groupby("genome")["sseqid"].nunique()
        carriers = set(bin_per_genome.loc[bin_per_genome >= min_bin_fraction * n_bin_genes].index)
        candidates = aln.loc[aln["sseqid"].isin(neighbors) & ~aln["genome"].isin(carriers)]
        duplicates = genbank_duplicates(candidates["genome"].unique())
        candidates = candidates.loc[~candidates["genome"].isin(duplicates)]
        found = candidates.groupby(["genome", "qseqid"])["sseqid"].apply(set)
        found = found.loc[[
            bool(genes & left) and bool(genes & right) and len(genes) >= min_flank_fraction * len(neighbors)
            for genes in found
        ]]
        if found.empty:
            return []

        carried_sets = [set(row["genes"].loc[~row["genes"]["in_bin"], "gene"]) for row in carrying_rows]
        on_contigs = aln.set_index(["genome", "qseqid"]).sort_index()
        placed = []
        for (genome, contig), genes in found.items():
            row = self._place_without_bin(
                on_contigs.loc[(genome, contig)].drop_duplicates("sseqid"), reference, left, right, window, bin,
            )
            if row is None:
                continue
            others = set(row["genes"].loc[~row["genes"]["in_bin"], "gene"])
            placed.append(dict(row, genome=genome, contig=contig, without_bin=True,
                               overlap=_mean_jaccard(others, carried_sets)))
        placed.sort(key=lambda row: (-row["overlap"], row["genome"], row["contig"]))
        chosen, seen = [], set()
        for row in placed:
            if row["genome"] not in seen:
                chosen.append(row); seen.add(row["genome"])
            if len(chosen) == n_rows:
                break
        return chosen

    @staticmethod
    def _place_without_bin(
        on_contig: pd.DataFrame, reference: pd.DataFrame, left: set, right: set, window: tuple, bin: str,
    ) -> dict | None:
        """
        One contig without the bin, placed in the shared space: each side
        fitted to the reference positions of that side's neighbors, split
        where the bin would sit. ``None`` when the two sides disagree on
        orientation or overlap, as for a rearranged neighborhood.
        """
        positions = on_contig.set_index("sseqid")[["qstart", "qend"]]
        left_sign, left_offset = _fit_to_reference(reference.loc[sorted(left)], positions)
        right_sign, right_offset = _fit_to_reference(reference.loc[sorted(right)], positions)
        if left_sign != right_sign:
            return None
        sign = left_sign
        # The junction: halfway between the innermost neighbor on each side,
        # in the contig's own coordinates
        middle = (positions["qstart"] + positions["qend"]) / 2
        inner_left = middle.loc[middle.index.intersection(sorted(left))]
        inner_right = middle.loc[middle.index.intersection(sorted(right))]
        inner_left = inner_left.max() if sign == 1 else inner_left.min()
        inner_right = inner_right.min() if sign == 1 else inner_right.max()
        if sign * (inner_right - inner_left) <= 0:
            return None
        junction = (inner_left + inner_right) / 2

        def place(x, on_left):
            return sign * x + (left_offset if on_left else right_offset)

        on_left = (sign * (middle - junction) < 0).values
        genes = pd.DataFrame({
            "gene": positions.index.values,
            "start": [place(x, s) for x, s in zip(positions["qstart"], on_left)],
            "stop": [place(x, s) for x, s in zip(positions["qend"], on_left)],
            "bin": on_contig["bin"].values,
        })
        genes = genes.loc[(genes[["start", "stop"]].max(axis=1) > window[0]) & (genes[["start", "stop"]].min(axis=1) < window[1])]
        length = int(on_contig["qlen"].iloc[0])
        ends = (1, length) if sign == 1 else (length, 1)
        left_end, right_end = place(ends[0], True), place(ends[1], False)
        segments = [
            (max(left_end, window[0]), min(place(junction, True), window[1])),
            (max(place(junction, False), window[0]), min(right_end, window[1])),
        ]
        segments = [(lo, hi) for lo, hi in segments if hi > lo]
        if not segments or place(junction, True) > place(junction, False):
            return None
        return {
            "genes": genes.assign(in_bin=genes["bin"] == bin).sort_values("start"),
            "extent": (segments[0][0], segments[-1][1]),
            "segments": segments,
        }

    def _bin_window(self, bin: str, max_gap: int, min_bin_fraction: float):
        """
        The bin's genes in one coordinate space, and the stretch of them that
        the gene map and the context map both draw.

        The contigs carrying at least ``min_bin_fraction`` of the bin's genes,
        less the GenBank copy of any assembly also present as RefSeq, are
        placed together; the window is the largest run of the genes they
        place with no gap over ``max_gap`` bp. A gene on none of those
        contigs lies outside it, as do the genes of every other run.

        Returns the bin's alignments, one per gene and contig; the gene count
        of each carrying contig; the ``Coords`` placing them; and the genes in
        the window with their coordinates.
        """
        aln = self.align_genomes
        in_bin = aln.loc[aln["bin"] == bin].drop_duplicates(["sseqid", "qseqid", "genome"])
        per_contig = in_bin.groupby(["genome", "qseqid"])["sseqid"].nunique()
        carrying = per_contig.loc[per_contig >= max(2, min_bin_fraction * int(self.bin_size[bin]))]
        if carrying.empty:
            raise ValueError(f"No contig carries {min_bin_fraction:.0%} of {bin}")
        duplicates = genbank_duplicates(carrying.index.get_level_values("genome"))
        carrying = carrying.loc[~carrying.index.get_level_values("genome").isin(duplicates)]

        placed = Coords(in_bin.set_index(["genome", "qseqid"]).loc[carrying.index].reset_index())
        return in_bin, carrying, placed, _largest_cluster(placed.to_df(), max_gap)

    @staticmethod
    def customize_label(label: str, remove_gene_id: bool, remove_org_tag: bool):
        if remove_gene_id:
            if label.startswith("Bin "):
                label = " - ".join(label.split(" - ", 2)[[0, 2]])
            else:
                label = label.split(" - ", 1)[1]
        if remove_org_tag:
            if " [" in label:
                label = label.rsplit(" [", 1)[0]
            if "MULTISPECIES: " in label:
                label = label.replace("MULTISPECIES: ", "")

        return label

    def bin_presence_heatmap(
        self,
        bins: str | List[str],
        width: int = 520,
        height: int = 640,
        tree_proportion: float = 0.9,
        horizontal_spacing: float = 0.04,
        file_prefix: str | None = None
    ) -> go.Figure:
        """
        Which genomes carry each of ``bins``, as a strip of cells beside the
        genomes' ANI tree, with the genomes in the tree's leaf order.
        ``tree_proportion`` is the share of the width given to the tree.
        """
        if isinstance(bins, str):
            bins = [bins]

        tree = self.ani_tree
        df = (
            self.genome_content
            .loc[self.genome_content["bin"].isin(bins)]
            .assign(present=1)
            .pivot_table(index="genome", columns="bin", values="present")
            .reindex(index=tree.leaves_list, columns=bins)
            .fillna(0)
            .astype(int)
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            horizontal_spacing=horizontal_spacing,
            column_widths=[tree_proportion, 1. - tree_proportion],
        )
        tree.plot_lines(fig, row=1, col=1)
        tree.plot_points(fig, mode="markers", row=1, col=1)
        fig.update_xaxes(title_text="ANI distance", row=1, col=1)
        # Above the strip, the bin names stay clear of the tree's last tick
        fig.update_xaxes(side="top", row=1, col=2)
        fig.add_heatmap(
            z=df.values,
            x=df.columns.values,
            y=list(range(df.shape[0])),
            colorscale="blues",
            showscale=False,
            row=1,
            col=2,
        )
        fig.update_layout(
            height=height,
            width=width,
            template=SIMPLE_TEMPLATE,
            margin=dict(r=20),
        )
        # Leaves sit at 0 .. n - 1; autorange would pad past them
        fig.update_yaxes(visible=False, range=[-0.5, df.shape[0] - 0.5])
        save_image(fig, file_prefix)
        return fig

    @cached_property
    def ani_tree(self) -> Phylogeny:
        # Get the ANI distances
        ani_dist = self.ani_distances.copy()

        # Make a distance matrix in BioPython format
        dm = DistanceMatrix(
            names=list(ani_dist.index.values),
            matrix=[
                l[:(i+1)]
                for i, l in enumerate(ani_dist.values.tolist())
            ]
        )

        # Compute a hash of the ani_dist DataFrame which can be used to identify the tree
        tree_hash = hashlib.sha256(ani_dist.to_csv().encode()).hexdigest()

        # Cache the NJ tree as Newick, using the hash as the filename
        cache_file = Path(f".cache/pangenome_ani_tree/{tree_hash}.nwk")
        if cache_file.exists():
            logger.info(f"Loading ANI tree from cache: {cache_file}")
            tree = Phylo.read(cache_file, "newick")
        else:
            logger.info(f"Computing ANI tree and caching to: {cache_file}")
            constructor = DistanceTreeConstructor()
            tree = constructor.nj(dm)
            tree.root_at_midpoint()
            cache_file.parent.mkdir(exist_ok=True, parents=True)
            Phylo.write(tree, cache_file, "newick")

        distances = pd.DataFrame(dm.matrix, index=dm.names, columns=dm.names)

        return Phylogeny(
            name="ANI",
            tree=tree,
            distances=distances
        )