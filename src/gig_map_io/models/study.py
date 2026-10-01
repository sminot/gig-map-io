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
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from plotly import graph_objects as go

from .contrast_metagenomes import ContrastMetagenomes
from .contrast_metagenomes_set import ContrastMetagenomesSet
from .pangenome import Pangenome
from .pangenome_set import PangenomeSet
from .phylogeny import PangenomePhylogeny
from .phylogeny_set import PangenomePhylogenySet
from .sample_group import SampleGroup
from ..helpers.enrichment import enrich_organisms, plot_enrichment
from ..helpers.ncbi import genome_names
from ..helpers.style import ESTIMATE_THRESH, FDR_THRESH, organism_order

#: The two sets the significant bins are split into, by the sign of their effect
DIRECTIONS = {"Positively associated": "positive", "Negatively associated": "negative"}


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


    @property
    def association(self) -> pd.DataFrame:
        return self.contrasts.association

    def significant_bins(
        self,
        estimate_thresh: float = ESTIMATE_THRESH,
        fdr_thresh: float = FDR_THRESH,
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

    def significant_bins_by_direction(self) -> Dict[str, pd.MultiIndex]:
        """
        The significant bins split by the direction of their effect, as a map
        of label to (pangenome, bin) pairs.
        """
        return {
            label: self.significant_bins(direction=direction).index
            for label, direction in DIRECTIONS.items()
        }


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
        index = features.index if isinstance(features, pd.DataFrame) else features
        return enrich_organisms(index, self.tested_bins, self.organisms, alternative)

    def organism_enrichment(self) -> pd.DataFrame:
        """
        Organism enrichment among the positively and among the negatively
        associated bins, tested independently and labelled by a ``group``
        column.
        """
        return pd.concat(
            [
                self.find_enriched_organisms(index).assign(group=label)
                for label, index in self.significant_bins_by_direction().items()
            ],
            ignore_index=True,
        )

    def annotation_enrichment(self) -> pd.DataFrame:
        """
        Annotation term enrichment among the positively and among the
        negatively associated bins, tested independently against the bins
        this study covered and labelled by a ``group`` column.
        """
        return pd.concat(
            [
                self.pangenomes.find_enriched_annotation_terms(index, universe=self.tested_bins).assign(group=label)
                for label, index in self.significant_bins_by_direction().items()
            ],
            ignore_index=True,
        )

    def plot_organism_enrichment(
        self,
        enrichment: pd.DataFrame,
        qvalue_threshold: float = 0.2,
        width: int = 800,
        height: int = 450,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The output of :meth:`organism_enrichment`: how many bins each organism
        contributed to each set, and how far that is from its share of the
        bins this study tested.
        """
        return plot_enrichment(
            enrichment,
            label="organism",
            axis_title="Organism",
            title=f"{self.label} - organisms among the associated bins",
            qvalue_threshold=qvalue_threshold,
            order=organism_order(self.organisms),
            width=width,
            height=height,
            file_prefix=file_prefix,
        )

    def plot_annotation_enrichment(
        self,
        enrichment: pd.DataFrame,
        qvalue_threshold: float = 0.2,
        max_terms: int = 20,
        width: int = 900,
        height: int | None = None,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The output of :meth:`annotation_enrichment`: the terms over-represented
        in each set of bins, relative to the bins this study tested, up to
        ``max_terms`` of them. The height follows the number of terms shown
        unless given.
        """
        significant = enrichment.loc[enrichment["qvalue"] < qvalue_threshold]
        # Break ties on the term itself, so that which terms make the cut does
        # not depend on the order they happened to be collected in
        keep = (
            significant.groupby("term")["pvalue"].min()
            .reset_index()
            .sort_values(["pvalue", "term"])
            .head(max_terms)["term"]
        )
        return plot_enrichment(
            enrichment.loc[enrichment["term"].isin(keep)],
            label="term",
            axis_title="Annotation term",
            title=f"{self.label} - annotations among the associated bins",
            qvalue_threshold=qvalue_threshold,
            order=None,
            width=width,
            height=height,
            file_prefix=file_prefix,
        )


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

    def plot_enriched_organisms(self, features, **kwargs: Any) -> go.Figure:
        return self.pangenomes.plot_enriched_organisms(features, **kwargs)

    def compare_membership_vs_distance(self, organism: str, **kwargs: Any) -> go.Figure:
        return self.pangenome(organism).compare_membership_vs_distance(**kwargs)

    def bin_gene_map(self, organism: str, bin: str, **kwargs: Any) -> go.Figure:
        return self.pangenome(organism).bin_gene_map(bin, **kwargs)

    def bin_context_map(self, organism: str, bin: str, **kwargs: Any):
        kwargs.setdefault("title", f"{organism} {bin} and its neighbourhood")
        return self.pangenome(organism).bin_context_map(bin, **kwargs)

    def genome_names(self, cache: str | Path) -> pd.DataFrame:
        """
        A display name for every genome in the study's pangenomes, built from
        NCBI's record of each assembly; see ``helpers.ncbi.genome_names``.
        """
        return genome_names(self.pangenomes.genomes, cache)

    def bin_presence_heatmap(self, organism: str, bins, **kwargs: Any) -> go.Figure:
        return self.pangenome(organism).bin_presence_heatmap(bins, **kwargs)


    def volcano_plot(self, **kwargs: Any) -> go.Figure:
        kwargs.setdefault("title", self.label)
        return self.contrasts.volcano_plot(**kwargs)

    def bin_abundance_heatmap(self, features, **kwargs: Any) -> go.Figure:
        """
        The sample annotations named in ``annotation_cols`` are shown with the
        display names and order the study's sample groups give them.
        """
        annotation_cols = kwargs.get("annotation_cols")
        if isinstance(annotation_cols, dict):
            labels, orders = {}, {}
            for column, display in annotation_cols.items():
                group_labels, group_order = self._metadata_group(column)
                if group_labels:
                    labels[display] = group_labels
                if group_order:
                    orders[display] = group_order
            kwargs.setdefault("annotation_labels", labels)
            kwargs.setdefault("annotation_orders", orders)
        return self.contrasts.bin_abundance_heatmap(features, **kwargs)

    def plot_bin_abundance(self, organism: str, bin: str, **kwargs: Any) -> go.Figure:
        """
        The metadata column used for ``color`` or ``facet_row`` is shown with
        the display names and order the study's sample groups give it.
        """
        columns = {kwargs.get(key) for key in ("color", "facet_row", "facet_col")} - {None}
        if len(columns) == 1:
            labels, order = self._metadata_group(next(iter(columns)))
            kwargs.setdefault("group_labels", labels)
            kwargs.setdefault("group_order", order or None)
        return self.contrast(organism).plot_bin_abundance(bin, **kwargs)

    def plot_bin_contingency(self, organism: str, bin: str, **kwargs: Any) -> go.Figure:
        kwargs.setdefault("metadata_col", self.parameter)
        kwargs.setdefault("group_labels", self._metadata_group(kwargs["metadata_col"])[0])
        return self.contrast(organism).plot_bin_contingency(bin, **kwargs)

    def _metadata_group(self, metadata_col: str) -> tuple[Dict[str, str], List[str]]:
        """
        Display names for the values of a metadata column, and the order to
        show them in, taken from whichever sample group reads that column.
        Lets a figure say "BSI" where the contrast says 1, without the
        analysis script repeating the mapping the study definition already
        carries. Empty when no sample group labels the column.
        """
        for group in self.sample_groups.values():
            for source in group.sources:
                if source.column == metadata_col and source.labels:
                    return dict(source.labels), list(group.order)
        return {}, []

    def _compare(self, method: str, comparator: "Study", kwargs: dict) -> go.Figure:
        kwargs.setdefault("self_label", self.label)
        kwargs.setdefault("comparator_label", comparator.label)
        return getattr(self.contrasts, method)(comparator.contrasts, **kwargs)

    def compare_sig_scatter(self, comparator: "Study", **kwargs: Any) -> go.Figure:
        return self._compare("compare_sig_scatter", comparator, kwargs)

    def compare_sig_categories(self, comparator: "Study", **kwargs: Any) -> go.Figure:
        return self._compare("compare_sig_categories", comparator, kwargs)

    def compare_association_scatter(self, comparator: "Study", **kwargs: Any) -> go.Figure:
        return self._compare("compare_association_scatter", comparator, kwargs)

    def compare_volcano_with_estimate(self, comparator: "Study", **kwargs: Any) -> go.Figure:
        return self._compare("compare_volcano_with_estimate", comparator, kwargs)


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

