import re
from functools import cached_property
from typing import Dict

from plotly.subplots import make_subplots
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from gig_map_io.helpers.enrichment import enrich_organisms, plot_enrichment
from gig_map_io.helpers.save_image import save_image
from gig_map_io.helpers.style import SIMPLE_TEMPLATE, TEMPLATE, organism_colors, organism_order

from .pangenome import Pangenome
from .dataset_dict import DatasetDict


#: A gene product name ends with the organism in brackets, and some begin with
#: qualifiers ("MULTISPECIES:", "MAG:", "MAG TPA:", "fOG:"). A colon inside a
#: name, as in "sodium:solute symporter", is part of it.
NAME_PREFIX = re.compile(r"^(?:(?:MULTISPECIES|MAG|TPA|fOG):?\s+)+")

#: A locus tag, as in "hypothetical protein DXA20_03275": unique to one gene
LOCUS_TAG = re.compile(r"^[A-Za-z][A-Za-z0-9]*_[0-9]+$")

#: Words that say nothing about function on their own. A term made only of
#: these (and of short tokens such as "A" or "II") is not a functional term
GENERIC_WORDS = frozenset({
    "protein", "proteins", "family", "superfamily", "domain", "domains",
    "domain-containing", "containing", "subunit", "subunits", "type", "like",
    "putative", "hypothetical", "uncharacterized", "predicted", "probable",
    "partial", "fragment", "component", "components", "chain", "region",
    "system", "related", "associated", "dependent", "involved", "unknown",
    "function", "conserved", "unnamed", "n-terminal", "c-terminal", "terminal",
})

#: Joining words, which leave a run hanging when they start or end it
FUNCTION_WORDS = frozenset({
    "of", "and", "or", "the", "in", "to", "for", "with", "by", "a", "an",
    "on", "from", "at", "as", "via",
})

#: A term may not begin or end on one of these: "conjugal transfer protein"
#: only repeats "conjugal transfer". "type" may begin one, as in "type VI
#: secretion".
EDGE_WORDS = FUNCTION_WORDS | (GENERIC_WORDS - {"type"})


def _strip_organism(name: str) -> str:
    """``name`` without its final bracketed group, matching nested brackets."""
    if not name.endswith("]"):
        return name
    depth = 0
    for i in range(len(name) - 1, -1, -1):
        if name[i] == "]":
            depth += 1
        elif name[i] == "[":
            depth -= 1
            if depth == 0:
                return name[:i].rstrip()
    return name


def _balanced(text: str) -> bool:
    """Whether every bracket in ``text`` is closed, in order."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack


def _informative(words: list) -> bool:
    """Whether a run of words could name a function."""
    if words[0].lower() in EDGE_WORDS or words[-1].lower() in EDGE_WORDS:
        return False
    # "VI secretion" is a fragment of "type VI secretion"; a short token may
    # end a run ("subunit A") but not start one
    if len(words) > 1 and len(words[0]) <= 2:
        return False
    if all(word.lower() in GENERIC_WORDS or len(word) <= 2 for word in words):
        return False
    # A bracketed qualifier on its own, such as "[Fe]" or "(ATP)"
    if len(words) == 1 and words[0][0] in "([{" and words[0][-1] in ")]}":
        return False
    return _balanced(" ".join(words))


def annotation_terms(name: str) -> set:
    """
    The functional terms in one gene product name: every run of consecutive
    words that could name a function.

    The organism in brackets at the end and a qualifier such as
    "MULTISPECIES:" at the start are removed, and so are locus tags. A double
    space is where a comma stood in the original name, and no term spans one.
    A run is dropped if it begins or ends on a joining word ("of", "in") or a
    generic one ("protein", "family", "subunit", "putative"; "type" may
    begin a run), if it starts on a token of two characters or fewer, if
    every word in it is generic or that short, if it is a bracketed
    qualifier alone ("[Fe]"), or if its
    brackets do not balance, which happens where a run cuts a chemical name
    in two.
    """
    name = NAME_PREFIX.sub("", _strip_organism(name))
    terms = set()
    for clause in re.split(r"\s{2,}", name):
        words = [word for word in clause.split() if not LOCUS_TAG.match(word)]
        for start in range(len(words)):
            for stop in range(start + 1, len(words) + 1):
                if _informative(words[start:stop]):
                    terms.add(" ".join(words[start:stop]))
    return terms


class PangenomeSet(DatasetDict):
    """
    Representation of a set of pangenomes.
    """

    @cached_property
    def pangenomes(self) -> Dict[str, Pangenome]:
        return {key: Pangenome(self.directory_dict[key]) for key in self.directory_dict.keys()}

    def __repr__(self) -> str:
        return f"PangenomeSet(directory_dict={self.directory_dict})"

    def __getitem__(self, key: str) -> Pangenome:
        return self.pangenomes[key]

    @cached_property
    def gene_bins(self) -> pd.DataFrame:
        return pd.concat([
            pangenome.gene_bins.assign(pangenome=pangenome_name)
            for pangenome_name, pangenome in self.pangenomes.items()
        ])

    @cached_property
    def genomes(self) -> list:
        """Every genome file name across the pangenomes, sorted."""
        return sorted({
            genome for pangenome in self.pangenomes.values() for genome in pangenome.align_genomes["genome"].unique()
        })

    @cached_property
    def core_genomes(self) -> Dict[str, str]:
        """The bin representing the core genome of each pangenome."""
        return {
            pangenome_name: pangenome.core_genome()
            for pangenome_name, pangenome in self.pangenomes.items()
        }

    def find_enriched_annotation_terms(
        self,
        features: pd.MultiIndex,
        min_count: int = 2,
        alternative: str = "greater",
        universe: pd.MultiIndex | None = None,
    ) -> pd.DataFrame:
        """
        Find annotation terms statistically over-represented in the given set of bins
        compared to the background of all other bins. What counts as a term is
        set out in :func:`annotation_terms`.

        Parameters
        ----------
        features : pd.MultiIndex
            (pangenome, bin) pairs making up the foreground set.
        min_count : int
            Minimum number of foreground bins a term must appear in to be tested.
        alternative : str
            Alternative hypothesis for Fisher's exact test ('greater', 'less', or 'two-sided').
        universe : pd.MultiIndex, optional
            The bins that could have been in the foreground. Defaults to every
            bin in the set; pass the bins an analysis actually covered to keep
            untested bins out of the background.

        Returns
        -------
        pd.DataFrame
            Columns: term, n_foreground, n_background, n_foreground_total,
                     n_background_total, odds_ratio, pvalue, qvalue
            Sorted by pvalue ascending.
        """

        # Drop rows where bin is NaN
        gb = self.gene_bins.dropna(subset=["bin"])

        # Build bin_terms: (pangenome, bin) -> set of terms
        bin_terms = (
            gb.groupby(["pangenome", "bin"])["combined_name"]
            .apply(lambda names: set().union(*[annotation_terms(n) for n in names]))
        )

        # Make sure that all of the features are in the pangenome set
        for (pangenome, feature) in features:
            if pangenome not in self.pangenomes:
                raise ValueError(f"Pangenome {pangenome} not found in pangenome set")
            if feature not in self.pangenomes[pangenome].bin_names:
                raise ValueError(f"Feature {feature} not found in pangenome {pangenome}")

        # Split into foreground and background
        fg_index = set(features)
        if universe is not None:
            considered = set(universe) | fg_index
            bin_terms = bin_terms.loc[[k in considered for k in bin_terms.index]]
        fg_bins = {k: v for k, v in bin_terms.items() if k in fg_index}
        bg_bins = {k: v for k, v in bin_terms.items() if k not in fg_index}

        n_fg_total = len(fg_bins)
        n_bg_total = len(bg_bins)

        # Collect all terms that appear in foreground bins
        fg_term_bins: dict = {}
        for bin_key, terms in fg_bins.items():
            for term in terms:
                fg_term_bins.setdefault(term, set()).add(bin_key)

        # Collect background term bin counts
        bg_term_bins: dict = {}
        for bin_key, terms in bg_bins.items():
            for term in terms:
                bg_term_bins.setdefault(term, set()).add(bin_key)

        # Run Fisher's exact test for each term with sufficient foreground support
        results = []
        for term, fg_set in fg_term_bins.items():
            a = len(fg_set)
            if a < min_count:
                continue
            b = len(bg_term_bins.get(term, set()))
            c = n_fg_total - a
            d = n_bg_total - b
            odds_ratio, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative=alternative)
            results.append({
                "term": term,
                "n_foreground": a,
                "n_background": b,
                "n_foreground_total": n_fg_total,
                "n_background_total": n_bg_total,
                "odds_ratio": odds_ratio,
                "pvalue": pvalue,
            })

        if not results:
            # Typed rather than bare, so that concatenating an empty result with
            # a populated one does not turn the numbers into objects
            return pd.DataFrame({
                "term": pd.Series(dtype=str),
                **{
                    name: pd.Series(dtype=int)
                    for name in ["n_foreground", "n_background",
                                 "n_foreground_total", "n_background_total"]
                },
                **{name: pd.Series(dtype=float) for name in ["odds_ratio", "pvalue", "qvalue"]},
            })

        df = pd.DataFrame(results)

        # Prune redundant shorter terms: drop term T if a longer super-term T' exists
        # such that T is a substring of T' and pvalue(T') <= pvalue(T)
        pvalue_map = dict(zip(df["term"], df["pvalue"]))
        terms_to_drop = set()
        all_terms = list(pvalue_map.keys())
        for term in all_terms:
            for other_term in all_terms:
                if other_term == term:
                    continue
                # other_term is a longer super-term containing term as a contiguous phrase
                if len(other_term) > len(term) and term in other_term and pvalue_map[other_term] <= pvalue_map[term]:
                    terms_to_drop.add(term)
                    break

        df = df[~df["term"].isin(terms_to_drop)].copy()

        # Apply FDR correction
        reject, qvalues, _, _ = multipletests(df["pvalue"].values, method="fdr_bh")
        df["qvalue"] = qvalues

        # Sorted by term as well as by p-value: many terms are carried by the
        # same handful of bins and so share a p-value exactly, and the order
        # they were collected in varies between processes.
        return df.sort_values(["pvalue", "term"]).reset_index(drop=True)

    @cached_property
    def all_bins(self) -> pd.MultiIndex:
        """Every (pangenome, bin) pair in the set."""
        return pd.MultiIndex.from_frame(
            self.gene_bins.dropna(subset=["bin"])[["pangenome", "bin"]].drop_duplicates(),
            names=["pangenome", "bin"],
        )

    def find_enriched_organisms(
        self,
        features: pd.MultiIndex | pd.DataFrame,
        alternative: str = "greater",
    ) -> pd.DataFrame:
        """
        Whether each organism contributed more bins to the given set than its
        share of every bin in the pangenomes. Fisher's exact test per
        organism, FDR-corrected across organisms.
        """
        index = features.index if isinstance(features, pd.DataFrame) else features
        return enrich_organisms(
            index, self.all_bins, organism_order(self.pangenomes), alternative
        )

    def plot_enriched_organisms(
        self,
        features: pd.MultiIndex | pd.DataFrame,
        qvalue_threshold: float = 0.2,
        width: int = 800,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The organism counterpart of :meth:`plot_enriched_annotation_terms`:
        how many of the bins each organism contributed and the log2 odds ratio
        against its share of the background, with the q-value beside each bar.
        """
        enrichment = self.find_enriched_organisms(features)
        return plot_enrichment(
            enrichment.assign(group="Candidate bins"),
            label="organism",
            axis_title="Organism",
            title="Organisms among the candidate bins",
            qvalue_threshold=qvalue_threshold,
            order=organism_order(enrichment["organism"]),
            width=width,
            height=None,
            file_prefix=file_prefix,
            mark="qvalue",
            show_legend=False,
        )

    def plot_enriched_annotation_terms(
        self,
        features: pd.MultiIndex,
        qvalue_threshold: float = 0.2,
        width: int = 800,
        height: int | None = None,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The terms passing the q-value threshold, drawn the same way as the
        per-study enrichment figures: how many of the bins carry each term,
        and the log2 odds ratio against the background, with the q-value
        printed beside it. Terms are ordered by odds ratio.

        Parameters
        ----------
        features : pd.MultiIndex
            The (pangenome, bin) pairs to test, against every other bin.
        qvalue_threshold : float
            Only show terms with qvalue < this threshold.
        height : int, optional
            Follows the number of terms unless given.
        """
        enrichment = self.find_enriched_annotation_terms(features)
        df = enrichment[enrichment["qvalue"] < qvalue_threshold].sort_values(["odds_ratio", "term"])
        return plot_enrichment(
            df.assign(group="Candidate bins"),
            label="term",
            axis_title="Annotation term",
            title="Annotations among the candidate bins",
            qvalue_threshold=qvalue_threshold,
            order=df["term"].tolist()[::-1],
            width=width,
            height=height,
            file_prefix=file_prefix,
            mark="qvalue",
            show_legend=False,
        )

    def bin_genome_heatmap(self,
        col_wrap: int = 3,
        width: int = 500,
        height: int = 400,
        horizontal_spacing: float = 0.05,
        vertical_spacing: float = 0.05,
        file_prefix: str | None = None
    ) -> go.Figure:
        """
        Heatmap of bin presence/absence for each genome, faceted by pangenome.
        """
        n_rows = -(-len(self.pangenomes) // col_wrap)
        fig = make_subplots(
            rows=n_rows,
            cols=col_wrap,
            shared_yaxes=False,
            shared_xaxes=False,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            subplot_titles=[pangenome for pangenome in self.pangenomes.keys()]
        )
        for i, pangenome in enumerate(self.pangenomes.keys()):
            for trace in self.pangenomes[pangenome].bin_genome_heatmap().data:
                fig.add_trace(
                    trace,
                    row=i // col_wrap + 1,
                    col=i % col_wrap + 1
                )

        fig.update_layout(
            height=height, width=width, template=SIMPLE_TEMPLATE,
            margin=dict(l=70, r=20, t=50, b=70),
        )

        # Left-align the subplot titles
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].update(x=0.02, xanchor='left', xref=f'x{i+1}')

        # One axis title per dimension, rather than one per facet
        fig.add_annotation(
            text="Gene", x=0.5, xref="paper", y=0, yref="paper", yshift=-45,
            showarrow=False, font=dict(size=14),
        )
        fig.add_annotation(
            text="Genome", x=0, xref="paper", y=0.5, yref="paper", xshift=-55,
            textangle=-90, showarrow=False, font=dict(size=14),
        )

        save_image(fig, file_prefix)
        return fig

    def bin_size_histogram(self,
        bins: int = 30,
        width: int = 500,
        height: int = 400,
        file_prefix: str | None = None
    ) -> go.Figure:
        """
        Histogram of bin sizes, faceted by pangenome.
        """
        # Set the boundaries of the bins to be the same for all pangenomes
        max_bin_size = max([pg.bin_size.max() for pg in self.pangenomes.values()])
        min_bin_size = min([pg.bin_size.min() for pg in self.pangenomes.values()])
        bins = np.linspace(np.log10(min_bin_size), np.log10(max_bin_size), bins + 1)

        df = pd.concat([
            pangenome.bin_size_df(bins).assign(pangenome=pangenome_name)
            for pangenome_name, pangenome in self.pangenomes.items()
        ])

        fig = px.bar(
            data_frame=df,
            x="bin_size",
            y="count",
            color="pangenome",
            color_discrete_map=organism_colors(self.pangenomes),
            labels=dict(
                bin_size="Pangenome bin size (genes)",
                count="Total gene content (genes)",
                pangenome="Organism"
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

    def rarefaction_curve(
        self,
        n_reps: int = 10,
        width: int = 500,
        height: int = 400,
        file_prefix: str | None = None
    ) -> go.Figure:
        """
        Rarefaction curve of the pangenomes, faceted by pangenome.
        """

        # Simulate the number of genes recovered with different numbers of subsampled genomes
        rf = pd.concat([
            pangenome.rarefaction_curve_data(n_reps).assign(pangenome=pangenome_name)
            for pangenome_name, pangenome in self.pangenomes.items()
        ]).rename(columns={"50%": "n_genes"})

        colors = organism_colors(self.pangenomes)
        fig = px.line(
            data_frame=rf,
            x="n_genomes",
            y="n_genes",
            color="pangenome",
            color_discrete_map=colors,
            labels=dict(
                n_genomes="Number of genomes",
                n_genes= "Number of genes",
                pangenome="Organism"
            ),
            template=TEMPLATE,
            width=width,
            height=height
        )
        fig.update_traces(line_width=2)
        point_df = rf.sort_values(by=["pangenome","n_genomes"]).groupby("pangenome").tail(1)
        for trace in px.scatter(
            data_frame=point_df, x="n_genomes", y="n_genes", color="pangenome",
            color_discrete_map=colors,
        ).data:
            trace.update(showlegend=False, marker_size=8)
            fig.add_trace(trace)
        fig.update_xaxes(type="log", dtick=1)
        fig.update_yaxes(range=[0, None])
        save_image(fig, file_prefix)
        return fig