"""
Study: a named set of gig-map outputs that are analyzed together.

A study ties together the contrast-metagenomes, pangenome, and phylogeny
outputs for one comparison (e.g. "GvHD - Fredricks"), keyed by organism. It is
serialized as JSON holding relative paths, so that a study definition can be
checked in alongside the analysis code and re-resolved against a downloaded
copy of the data.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .contrast_metagenomes import ContrastMetagenomes
from .contrast_metagenomes_set import ContrastMetagenomesSet
from .pangenome import Pangenome
from .pangenome_set import PangenomeSet
from .phylogeny import PangenomePhylogeny
from .phylogeny_set import PangenomePhylogenySet
from .sample_group import SampleGroup
from ..helpers.save_image import save_image


class Study:
    """
    A named collection of gig-map workflow outputs analyzed as a unit.

    Parameters
    ----------
    name:
        Short slug identifying the study (matches the JSON file name).
    label:
        Display name used in figure titles and axis labels.
    parameter:
        Name of the association parameter tested in the contrasts. ``None``
        for studies that hold only pangenome/phylogeny outputs.
    contrasts, pangenomes, phylogenies:
        Maps of organism name to the output directory for that workflow,
        named relative to the root of the dataset collection rather than to
        any particular filesystem.
    datasets:
        Where that collection lives. Defaults to ``datasets`` in the current
        working directory; pass it explicitly to read the same definition
        against a copy of the data somewhere else.
    sample_groups:
        Named recodings of the contrast metadata into labelled categoricals
        (see :class:`~gig_map_io.models.sample_group.SampleGroup`).
    """

    def __init__(
        self,
        name: str,
        label: str | None = None,
        parameter: str | None = None,
        contrasts: Dict[str, str] | None = None,
        pangenomes: Dict[str, str] | None = None,
        phylogenies: Dict[str, str] | None = None,
        sample_groups: Dict[str, SampleGroup] | None = None,
        datasets: str | Path = "datasets",
    ) -> None:
        self.name = name
        self.label = label if label is not None else name
        self.parameter = parameter
        self.contrast_dirs = dict(contrasts or {})
        self.pangenome_dirs = dict(pangenomes or {})
        self.phylogeny_dirs = dict(phylogenies or {})
        self.sample_groups = dict(sample_groups or {})
        self.datasets = Path(datasets)

        if self.contrast_dirs and self.parameter is None:
            raise ValueError(f"Study {name!r} defines contrasts but no parameter")

    # --- Serialization ----------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path, datasets: str | Path = "datasets") -> "Study":
        """
        Read a study definition from a JSON file.

        The dataset directories named inside the file are resolved against
        ``datasets``, not against the location of the JSON file, so a
        definition stays valid wherever the data has been copied to.
        """
        with Path(path).open() as handle:
            spec = json.load(handle)

        return cls(
            name=spec["name"],
            label=spec.get("label"),
            parameter=spec.get("parameter"),
            contrasts=spec.get("contrasts"),
            pangenomes=spec.get("pangenomes"),
            phylogenies=spec.get("phylogenies"),
            sample_groups={
                key: SampleGroup.from_dict(val)
                for key, val in spec.get("sample_groups", {}).items()
            },
            datasets=datasets,
        )

    def to_json(self, path: str | Path) -> None:
        """Write this study definition to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    def to_dict(self) -> dict:
        spec: dict = {"name": self.name, "label": self.label}
        if self.parameter is not None:
            spec["parameter"] = self.parameter
        for key, val in [
            ("contrasts", self.contrast_dirs),
            ("pangenomes", self.pangenome_dirs),
            ("phylogenies", self.phylogeny_dirs),
        ]:
            if val:
                spec[key] = val
        if self.sample_groups:
            spec["sample_groups"] = {
                key: val.to_dict() for key, val in self.sample_groups.items()
            }
        return spec

    def __repr__(self) -> str:
        return (
            f"Study(name={self.name!r}, parameter={self.parameter!r}, "
            f"organisms={len(self.organisms)})"
        )

    # --- Reader objects ---------------------------------------------------

    def _resolve(self, dirs: Dict[str, str]) -> Dict[str, Path]:
        return {name: self.datasets / path for name, path in dirs.items()}

    @cached_property
    def contrasts(self) -> ContrastMetagenomesSet:
        if not self.contrast_dirs:
            raise ValueError(f"Study {self.name!r} defines no contrasts")
        return ContrastMetagenomesSet(self._resolve(self.contrast_dirs), self.parameter)

    @cached_property
    def pangenomes(self) -> PangenomeSet:
        if not self.pangenome_dirs:
            raise ValueError(f"Study {self.name!r} defines no pangenomes")
        return PangenomeSet(self._resolve(self.pangenome_dirs))

    @cached_property
    def phylogenies(self) -> PangenomePhylogenySet:
        if not self.phylogeny_dirs:
            raise ValueError(f"Study {self.name!r} defines no phylogenies")
        return PangenomePhylogenySet(self._resolve(self.phylogeny_dirs))

    @property
    def organisms(self) -> List[str]:
        """Organism names, in the order they appear in the study definition."""
        for dirs in (self.contrast_dirs, self.pangenome_dirs, self.phylogeny_dirs):
            if dirs:
                return list(dirs)
        return []

    def contrast(self, organism: str) -> ContrastMetagenomes:
        return self.contrasts[organism]

    def pangenome(self, organism: str) -> Pangenome:
        return self.pangenomes[organism]

    def phylogeny(self, organism: str) -> PangenomePhylogeny:
        return self.phylogenies[organism]

    # --- Sample metadata --------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self.contrasts.n_samples

    def sample_metadata(self, groups: Iterable[str] | None = None) -> pd.DataFrame:
        """
        Resolve the study's sample groups against the contrast metadata.

        Returns one column per sample group, indexed by sample.
        """
        metadata = self.contrasts.metadata
        names = list(groups) if groups is not None else list(self.sample_groups)
        missing = [name for name in names if name not in self.sample_groups]
        if missing:
            raise KeyError(f"Study {self.name!r} defines no sample group(s) {missing}")
        return pd.DataFrame(
            {name: self.sample_groups[name].resolve(metadata) for name in names}
        )

    def group_order(self, group: str) -> List[str]:
        """Category order declared for one sample group."""
        return list(self.sample_groups[group].order)

    # --- Association results ----------------------------------------------

    @property
    def association(self) -> pd.DataFrame:
        return self.contrasts.association

    def significant_bins(
        self,
        estimate_thresh: float = 0.25,
        fdr_thresh: float = 0.2,
        direction: str | None = None,
    ) -> pd.DataFrame:
        """
        Association results for the bins passing the significance thresholds.

        Parameters
        ----------
        estimate_thresh:
            Minimum absolute effect size.
        fdr_thresh:
            Maximum FDR-adjusted q-value.
        direction:
            ``"positive"`` or ``"negative"`` to keep only one sign of effect;
            ``None`` (default) keeps both.

        Returns
        -------
        DataFrame indexed by (pangenome, feature).
        """
        df = self.association.set_index(["pangenome", "feature"])
        df = df.loc[(df["Estimate"].abs() > estimate_thresh) & (df["qvalue"] < fdr_thresh)]
        if direction == "positive":
            df = df.loc[df["Estimate"] > 0]
        elif direction == "negative":
            df = df.loc[df["Estimate"] < 0]
        elif direction is not None:
            raise ValueError("direction must be 'positive', 'negative', or None")
        return df.sort_values(by="pvalue")

    @cached_property
    def tested_bins(self) -> pd.MultiIndex:
        """
        The (pangenome, bin) pairs the association analysis covered.

        This is the right background for asking what is over-represented among
        the bins it called significant: a bin that was never tested had no
        chance of being called.
        """
        return pd.MultiIndex.from_frame(
            self.association[["pangenome", "feature"]], names=["pangenome", "bin"]
        )

    def significant_bins_by_direction(
        self,
        estimate_thresh: float = 0.25,
        fdr_thresh: float = 0.2,
    ) -> Dict[str, pd.MultiIndex]:
        """
        The significant bins split by the direction of their effect, as a map
        of label to (pangenome, bin) pairs.
        """
        return {
            label: self.significant_bins(estimate_thresh, fdr_thresh, direction).index
            for label, direction in [
                ("Positively associated", "positive"),
                ("Negatively associated", "negative"),
            ]
        }

    # --- Enrichment among a set of bins ------------------------------------

    def find_enriched_organisms(
        self,
        features: pd.MultiIndex | pd.DataFrame,
        alternative: str = "greater",
    ) -> pd.DataFrame:
        """
        Test whether each organism contributed more bins to the given set than
        its share of the bins this study tested.

        Fisher's exact test per organism, FDR-corrected across organisms.

        Returns
        -------
        DataFrame with one row per organism: the number of its bins in the
        foreground and in the background, the totals, the odds ratio, and the
        p- and q-values.
        """
        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        index = features.index if isinstance(features, pd.DataFrame) else features
        foreground = set(index)
        universe = set(self.tested_bins)

        untested = foreground - universe
        if untested:
            raise ValueError(
                f"{len(untested)} of the given bins were not tested by study "
                f"{self.name!r}, e.g. {sorted(untested)[0]}"
            )

        foreground_counts = Counter(organism for organism, _ in foreground)
        universe_counts = Counter(organism for organism, _ in universe)
        n_foreground = len(foreground)
        n_background = len(universe) - n_foreground

        rows = []
        for organism in self.organisms:
            in_foreground = foreground_counts.get(organism, 0)
            in_background = universe_counts.get(organism, 0) - in_foreground
            odds_ratio, pvalue = stats.fisher_exact(
                [
                    [in_foreground, in_background],
                    [n_foreground - in_foreground, n_background - in_background],
                ],
                alternative=alternative,
            )
            rows.append({
                "organism": organism,
                "n_foreground": in_foreground,
                "n_background": in_background,
                "n_foreground_total": n_foreground,
                "n_background_total": n_background,
                "odds_ratio": odds_ratio,
                "pvalue": pvalue,
            })

        df = pd.DataFrame(rows)
        df["qvalue"] = multipletests(df["pvalue"], method="fdr_bh")[1]
        return df.sort_values("pvalue").reset_index(drop=True)

    def organism_enrichment(
        self,
        groups: Dict[str, pd.MultiIndex] | None = None,
        alternative: str = "greater",
        **thresholds: float,
    ) -> pd.DataFrame:
        """
        Organism enrichment for several sets of bins at once, tested
        independently and labelled by a ``group`` column.

        Defaults to the significant bins split by direction of effect.
        """
        groups = groups if groups is not None else self.significant_bins_by_direction(**thresholds)
        return pd.concat(
            [
                self.find_enriched_organisms(index, alternative=alternative).assign(group=label)
                for label, index in groups.items()
            ],
            ignore_index=True,
        )

    def annotation_enrichment(
        self,
        groups: Dict[str, pd.MultiIndex] | None = None,
        min_count: int = 2,
        alternative: str = "greater",
        **thresholds: float,
    ) -> pd.DataFrame:
        """
        Annotation term enrichment for several sets of bins at once, tested
        independently against the bins this study covered and labelled by a
        ``group`` column.

        Defaults to the significant bins split by direction of effect.
        """
        groups = groups if groups is not None else self.significant_bins_by_direction(**thresholds)
        return pd.concat(
            [
                self.pangenomes.find_enriched_annotation_terms(
                    index,
                    min_count=min_count,
                    alternative=alternative,
                    universe=self.tested_bins,
                ).assign(group=label)
                for label, index in groups.items()
            ],
            ignore_index=True,
        )

    def plot_organism_enrichment(
        self,
        enrichment: pd.DataFrame | Dict[str, pd.MultiIndex] | None = None,
        qvalue_threshold: float = 0.2,
        width: int = 800,
        height: int = 450,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        How many bins each organism contributed to each set, and how far that
        is from its share of the bins this study tested.

        Takes the output of :meth:`organism_enrichment`, a map of label to
        bins, or nothing at all (in which case the significant bins are split
        by direction of effect).
        """
        if not isinstance(enrichment, pd.DataFrame):
            enrichment = self.organism_enrichment(enrichment)

        return _enrichment_figure(
            enrichment,
            label="organism",
            axis_title="Organism",
            title=f"{self.label} - organisms among the associated bins",
            qvalue_threshold=qvalue_threshold,
            order=list(reversed(self.organisms)),
            width=width,
            height=height,
            file_prefix=file_prefix,
        )

    def plot_annotation_enrichment(
        self,
        enrichment: pd.DataFrame | Dict[str, pd.MultiIndex] | None = None,
        qvalue_threshold: float = 0.2,
        max_terms: int = 20,
        width: int = 900,
        height: int = 600,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Annotation terms over-represented in each set of bins, relative to the
        bins this study tested.

        Takes the output of :meth:`annotation_enrichment`, a map of label to
        bins, or nothing at all (in which case the significant bins are split
        by direction of effect).
        """
        if not isinstance(enrichment, pd.DataFrame):
            enrichment = self.annotation_enrichment(enrichment)

        significant = enrichment.loc[enrichment["qvalue"] < qvalue_threshold]
        keep = (
            significant.groupby("term")["pvalue"].min()
            .sort_values().head(max_terms).index
        )
        return _enrichment_figure(
            enrichment.loc[enrichment["term"].isin(keep)],
            label="term",
            axis_title="Annotation Term",
            title=f"{self.label} - annotations among the associated bins",
            qvalue_threshold=qvalue_threshold,
            order=None,
            width=width,
            height=height,
            file_prefix=file_prefix,
        )

    # --- Pangenome-level plots --------------------------------------------

    @cached_property
    def core_genomes(self) -> Dict[str, str]:
        """The bin representing the core genome of each organism."""
        return self.pangenomes.core_genomes

    def bin_genome_heatmap(self, **kwargs: Any) -> go.Figure:
        return self.pangenomes.bin_genome_heatmap(**kwargs)

    def rarefaction_curve(self, **kwargs: Any) -> go.Figure:
        return self.pangenomes.rarefaction_curve(**kwargs)

    def bin_size_histogram(self, **kwargs: Any) -> go.Figure:
        return self.pangenomes.bin_size_histogram(**kwargs)

    def plot_enriched_annotation_terms(self, features, **kwargs: Any) -> go.Figure:
        return self.pangenomes.plot_enriched_annotation_terms(features, **kwargs)

    def compare_membership_vs_distance(self, organism: str, **kwargs: Any) -> go.Figure:
        return self.pangenome(organism).compare_membership_vs_distance(**kwargs)

    def bin_gene_map(self, organism: str, bin: str, **kwargs: Any) -> go.Figure:
        return self.pangenome(organism).bin_gene_map(bin, **kwargs)

    def bin_presence_heatmap(self, organism: str, bins, **kwargs: Any) -> go.Figure:
        return self.pangenome(organism).bin_presence_heatmap(bins, **kwargs)

    # --- Contrast-level plots ---------------------------------------------

    def volcano_plot(self, **kwargs: Any) -> go.Figure:
        kwargs.setdefault("title", self.label)
        return self.contrasts.volcano_plot(**kwargs)

    def bin_abundance_heatmap(self, features, **kwargs: Any) -> go.Figure:
        return self.contrasts.bin_abundance_heatmap(features, **kwargs)

    def plot_bin_abundance(self, organism: str, bin: str, **kwargs: Any) -> go.Figure:
        return self.contrast(organism).plot_bin_abundance(bin, **kwargs)

    def _compare(self, method: str, comparitor: "Study", kwargs: dict) -> go.Figure:
        kwargs.setdefault("self_label", self.label)
        kwargs.setdefault("comparitor_label", comparitor.label)
        return getattr(self.contrasts, method)(comparitor.contrasts, **kwargs)

    def compare_sig_scatter(self, comparitor: "Study", **kwargs: Any) -> go.Figure:
        return self._compare("compare_sig_scatter", comparitor, kwargs)

    def compare_sig_categories(self, comparitor: "Study", **kwargs: Any) -> go.Figure:
        return self._compare("compare_sig_categories", comparitor, kwargs)

    def compare_association_scatter(self, comparitor: "Study", **kwargs: Any) -> go.Figure:
        return self._compare("compare_association_scatter", comparitor, kwargs)

    def compare_volcano_with_estimate(self, comparitor: "Study", **kwargs: Any) -> go.Figure:
        return self._compare("compare_volcano_with_estimate", comparitor, kwargs)

    # --- Bin descriptions -------------------------------------------------

    def describe_bins(
        self,
        features: pd.MultiIndex | pd.DataFrame,
        odds_by: str | None = None,
        ref_group: Any = 1,
        comp_group: Any = 0,
        threshold: float | str = 10.0,
    ) -> pd.DataFrame:
        """
        Summarize a set of pangenome bins: size, prevalence, and log2 odds
        ratio of detection between the contrast groups.

        Parameters
        ----------
        features:
            (pangenome, bin) pairs to describe.
        odds_by:
            Name of a sample group; the odds ratio is computed separately
            within each of its levels, one column per level. When ``None``,
            a single odds ratio is computed across all samples.
        ref_group, comp_group:
            Values of the contrast parameter to use as the reference and
            comparison groups. A positive log2 odds ratio means the bin is
            more often detected in ``comp_group``.
        threshold:
            RPKM at or above which a bin counts as detected.
        """
        index = features.index if isinstance(features, pd.DataFrame) else features
        subsets: Dict[str, pd.Index | None] = {"log2_odds_ratio": None}
        if odds_by is not None:
            labels = self.sample_metadata([odds_by])[odds_by]
            order = self.group_order(odds_by) or sorted(labels.dropna().unique())
            subsets = {level: labels.index[labels == level] for level in order}

        rows = []
        for organism, bin_id in index:
            pangenome = self.pangenome(organism)
            contrast = self.contrast(organism)
            row = {
                "n_genes": int(pangenome.bin_size[bin_id]),
                "n_genomes": int(pangenome.bin_presence_wide[bin_id].sum()),
                "n_genomes_total": pangenome.bin_presence_wide.shape[0],
            }
            for label, samples in subsets.items():
                row[label] = np.log2(
                    contrast.calc_odds_ratio(
                        metadata_col=self.parameter,
                        ref_group=ref_group,
                        comp_group=comp_group,
                        bin_id=bin_id,
                        samples=samples,
                        threshold=threshold,
                    )
                )
            rows.append(row)

        return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
            list(index), names=["pangenome", "bin"]
        ))


def _enrichment_figure(
    enrichment: pd.DataFrame,
    label: str,
    axis_title: str,
    title: str,
    qvalue_threshold: float,
    order: List[str] | None,
    width: int,
    height: int,
    file_prefix: str | None,
) -> go.Figure:
    """
    Two panels sharing a category axis: how many bins of each category fell in
    each group, and how enriched that is. Bars are grouped by the set the bins
    came from, so the two directions of effect can be read against each other.

    Odds ratios are shown on a log2 scale, which is symmetric about no
    enrichment, and marked where they clear the q-value threshold. A category
    with no bins at all in a group has nothing to say about enrichment, so no
    bar is drawn for it; one whose bins are *all* in the foreground has an
    infinite odds ratio and is drawn at the edge of the observed range.
    """
    df = enrichment.copy()
    log2_odds = np.log2(df["odds_ratio"].replace({0: np.nan, np.inf: np.nan}).astype(float))
    limit = float(np.nanmax(np.abs(log2_odds))) if log2_odds.notna().any() else 1.0

    unbounded = np.isinf(df["odds_ratio"]) & (df["n_foreground"] > 0)
    df["log2_odds_ratio"] = log2_odds.where(~unbounded, limit)
    # Nothing observed means no evidence either way, rather than depletion
    df.loc[df["n_foreground"] == 0, "log2_odds_ratio"] = np.nan
    df["mark"] = np.where(df["qvalue"] < qvalue_threshold, "*", "")

    category_orders = {"group": list(dict.fromkeys(enrichment["group"]))}
    if order is not None:
        category_orders[label] = order

    hover = {"qvalue": ":.2e", "odds_ratio": ":.3g", "n_foreground": True, "n_background": True}
    shared = dict(
        data_frame=df, y=label, color="group", orientation="h", barmode="group",
        template="plotly_white", hover_data=hover, category_orders=category_orders,
    )

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06)
    for trace in px.bar(x="n_foreground", **shared).data:
        fig.add_trace(trace, row=1, col=1)
    for trace in px.bar(x="log2_odds_ratio", text="mark", **shared).data:
        fig.add_trace(trace.update(showlegend=False, textposition="outside"), row=1, col=2)

    fig.add_vline(x=0, line_dash="dot", line_color="grey", row=1, col=2)
    fig.update_layout(
        width=width,
        height=height,
        title=title,
        template="plotly_white",
        legend_title_text="",
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(title="Bins"),
        xaxis2=dict(title=f"Log2 Odds Ratio (* q < {qvalue_threshold})"),
        yaxis=dict(title=axis_title, automargin=True),
    )
    save_image(fig, file_prefix)
    return fig
