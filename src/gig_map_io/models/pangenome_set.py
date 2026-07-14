from functools import cached_property
from logging import getLogger
import logging
from pathlib import Path
import sys
from typing import Dict

from plotly.subplots import make_subplots
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from gig_map_io.helpers.save_image import save_image

from .pangenome import Pangenome
from .dataset_dict import DatasetDict

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

class PangenomeSet(DatasetDict):
    """
    Representation of a set of pangenomes.
    """
    def __init__(self, directory_dict: Dict[str, str | Path]) -> None:
        super().__init__(directory_dict)

    @cached_property
    def pangenomes(self) -> Dict[str, Pangenome]:
        return {key: Pangenome(self.directory_dict[key]) for key in self.directory_dict.keys()}

    def __repr__(self) -> str:
        return f"PangenomeSet(directory_dict={self.directory_dict})"

    def __str__(self) -> str:
        return f"PangenomeSet(directory_dict={self.directory_dict})"

    def __format__(self, format_spec: str) -> str:
        return f"PangenomeSet(directory_dict={self.directory_dict})"

    def __getitem__(self, key: str) -> Pangenome:
        return self.pangenomes[key]

    @cached_property
    def gene_bins(self) -> pd.DataFrame:
        return pd.concat([
            pangenome.gene_bins.assign(pangenome=pangenome_name)
            for pangenome_name, pangenome in self.pangenomes.items()
        ])

    def find_enriched_annotation_terms(
        self,
        features: pd.MultiIndex,
        min_count: int = 2,
        alternative: str = "greater",
    ) -> pd.DataFrame:
        """
        Find annotation terms statistically over-represented in the given set of bins
        compared to the background of all bins in the PangenomeSet.

        Parameters
        ----------
        features : pd.MultiIndex
            MultiIndex with level names 'pangenome' and 'feature', where 'feature'
            corresponds to the 'bin' column in gene_bins.
        min_count : int
            Minimum number of foreground bins a term must appear in to be tested.
        alternative : str
            Alternative hypothesis for Fisher's exact test ('greater', 'less', or 'two-sided').

        Returns
        -------
        pd.DataFrame
            Columns: term, n_foreground, n_background, n_foreground_total,
                     n_background_total, odds_ratio, pvalue, qvalue
            Sorted by pvalue ascending.
        """

        def _ngrams(text: str) -> set:
            words = text.split()
            return {" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, len(words) + 1)}

        def _sanitize_combined_name(combined_name: str) -> str:
            if combined_name.endswith("]") and "[" in combined_name:
                combined_name = combined_name.rsplit("[", 1)[0]
            if combined_name.startswith("MULTISPECIES: "):
                combined_name = combined_name.replace("MULTISPECIES: ", "")
            return combined_name

        # Drop rows where bin is NaN
        gb = self.gene_bins.dropna(subset=["bin"])

        # Build bin_terms: (pangenome, bin) -> set of n-gram terms
        bin_terms = (
            gb.groupby(["pangenome", "bin"])["combined_name"]
            .apply(lambda names: set().union(*[_ngrams(_sanitize_combined_name(n)) for n in names]))
        )

        # Make sure that all of the features are in the pangenome set
        for (pangenome, feature) in features:
            if pangenome not in self.pangenomes:
                raise ValueError(f"Pangenome {pangenome} not found in pangenome set")
            if feature not in self.pangenomes[pangenome].bin_names:
                raise ValueError(f"Feature {feature} not found in pangenome {pangenome}")

        # Split into foreground and background
        fg_index = set(zip(
            features.get_level_values("pangenome"),
            features.get_level_values("feature")
        ))
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
            return pd.DataFrame(columns=[
                "term", "n_foreground", "n_background",
                "n_foreground_total", "n_background_total",
                "odds_ratio", "pvalue", "qvalue"
            ])

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

        return df.sort_values("pvalue").reset_index(drop=True)

    def plot_enriched_annotation_terms(
        self,
        features: pd.MultiIndex | pd.DataFrame,
        qvalue_threshold: float = 0.2,
        min_count: int = 2,
        alternative: str = "greater",
        width: int = 800,
        height: int = 500,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Horizontal bar plot of enriched annotation terms.

        Parameters
        ----------
        features : pd.MultiIndex or pd.DataFrame
            Either a MultiIndex (levels: pangenome, feature) passed directly to
            find_enriched_annotation_terms, or the DataFrame output of that method.
        qvalue_threshold : float
            Only show terms with qvalue < this threshold.
        min_count : int
            Passed to find_enriched_annotation_terms when features is a MultiIndex.
        alternative : str
            Passed to find_enriched_annotation_terms when features is a MultiIndex.
        """
        if isinstance(features, pd.MultiIndex):
            enrichment_df = self.find_enriched_annotation_terms(
                features, min_count=min_count, alternative=alternative
            )
        else:
            enrichment_df = features

        df = enrichment_df[enrichment_df["qvalue"] < qvalue_threshold].sort_values("odds_ratio")

        hover = {"qvalue": ":.2e", "odds_ratio": ":.2f", "n_foreground": True, "n_background": True}
        common = dict(orientation="h", template="plotly_white")

        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05)

        for trace in px.bar(
            data_frame=df, x="n_foreground", y="term", hover_data=hover,
            labels=dict(n_foreground="Foreground Bins", term="Annotation Term"),
            **common,
        ).data:
            fig.add_trace(trace, row=1, col=1)

        for trace in px.bar(
            data_frame=df, x="odds_ratio", y="term", hover_data=hover,
            labels=dict(odds_ratio="Odds Ratio", term="Annotation Term"),
            **common,
        ).data:
            fig.add_trace(trace, row=1, col=2)

        for trace in px.bar(
            data_frame=df, x="qvalue", y="term", hover_data=hover,
            labels=dict(qvalue="Q-value", term="Annotation Term"),
            **common,
        ).data:
            fig.add_trace(trace, row=1, col=3)

        fig.update_layout(
            width=width,
            height=height,
            showlegend=False,
            xaxis=dict(title="Bins"),
            xaxis2=dict(title="Odds Ratio"),
            xaxis3=dict(title="Q-value"),
            yaxis=dict(automargin=True),
            template="plotly_white",
        )
        save_image(fig, file_prefix)
        return fig

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
        fig = make_subplots(
            rows=len(self.pangenomes) // col_wrap + 1,
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

        fig.update_layout(height=height, width=width)

        # Left-align the subplot titles
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].update(x=0.02, xanchor='left', xref=f'x{i+1}')

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
            labels=dict(
                bin_size="Pangenome Bin Size (# of Genes)",
                count="Total Gene Content",
                pangenome="Pangenome"
            ),
            template="plotly_white",
            hover_name="bin_names",
            width=width,
            height=height
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=["1", "10", "100", "1k", "10k", "100k"]
        )
        # If save_image was provided, use the string as the file
        # prefix to write out HTML, PDF, PNG, and JSON
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

        fig = px.line(
            data_frame=rf,
            x="n_genomes",
            y="n_genes",
            color="pangenome",
            labels=dict(
                n_genomes="Number of Genomes",
                n_genes= "Number of Genes",
                pangenome="Pangenome"
            ),
            template="plotly_white",
            width=width,
            height=height
        )
        point_df = rf.sort_values(by=["pangenome","n_genomes"]).groupby("pangenome").tail(1)
        for trace in px.scatter(data_frame=point_df, x="n_genomes", y="n_genes", color="pangenome").data:
            trace.update(showlegend=False)
            fig.add_trace(trace)
        fig.update_xaxes(type="log")
        fig.update_yaxes(range=[0, None])
        save_image(fig, file_prefix)
        return fig