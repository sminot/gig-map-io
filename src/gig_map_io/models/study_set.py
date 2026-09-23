"""
StudySet: analyses that span more than one study.

Where a :class:`~gig_map_io.models.study.Study` covers one contrast, a
``StudySet`` stacks the samples of several studies into a single table so that
community-level structure can be compared across them.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go

from ..helpers.clustering import leiden
from ..helpers.contingency import chi2_contingency_test
from ..helpers.ordination import tsne
from ..helpers.permanova import permanova
from ..helpers.positivity import plot_feature_positivity, plot_positivity_heatmap
from ..helpers.save_image import save_image
from .study import Study

#: Sample groups every study in a set is expected to define.
REQUIRED_SAMPLE_GROUPS = ("study", "disease", "participant")


class StudySet:
    """
    Several studies analyzed together over a shared sample-by-bin table.

    Parameters
    ----------
    studies:
        The studies to combine. Sample indices must be disjoint.
    """

    def __init__(self, studies: Iterable[Study]) -> None:
        self.studies = list(studies)
        if not self.studies:
            raise ValueError("StudySet requires at least one study")

    @classmethod
    def from_json(cls, *paths: str | Path, base: str | Path | None = None) -> "StudySet":
        """Read several study definitions from JSON files."""
        return cls([Study.from_json(path, base=base) for path in paths])

    def __repr__(self) -> str:
        return f"StudySet(studies={[study.name for study in self.studies]})"

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, name: str) -> Study:
        for study in self.studies:
            if study.name == name:
                return study
        raise KeyError(f"No study named {name!r} in this set")

    # --- Combined tables --------------------------------------------------

    @cached_property
    def rpkm(self) -> pd.DataFrame:
        """Bin RPKM for every sample, columns keyed by (organism, bin)."""
        return pd.concat([study.contrasts.rpkm for study in self.studies]).fillna(0)

    @cached_property
    def bin_size(self) -> pd.Series:
        """Number of genes in each bin of the combined table."""
        pangenomes = {}
        for study in self.studies:
            if study.pangenome_dirs:
                pangenomes.update(study.pangenomes.pangenomes)
        if not pangenomes:
            raise ValueError("No study in this set defines pangenomes")
        return pd.Series(
            [pangenomes[organism].bin_size[bin_id] for organism, bin_id in self.rpkm.columns],
            index=self.rpkm.columns,
        )

    @cached_property
    def weighted_rpkm(self) -> pd.DataFrame:
        """Bin RPKM scaled by the number of genes in each bin."""
        return self.rpkm * self.bin_size

    @cached_property
    def sample_metadata(self) -> pd.DataFrame:
        """
        Sample annotations for every study, with one column per shared sample
        group plus a ``study`` column naming the cohort each sample came from.
        """
        return pd.concat(
            [study.sample_metadata(REQUIRED_SAMPLE_GROUPS) for study in self.studies]
        )

    @cached_property
    def study_order(self) -> List[str]:
        """Cohort names in the order declared by the studies."""
        order: List[str] = []
        for study in self.studies:
            for name in study.group_order("study"):
                if name not in order:
                    order.append(name)
        return order

    def features(self, features: pd.MultiIndex | pd.DataFrame | None = None) -> pd.DataFrame:
        """The combined RPKM table, optionally restricted to a subset of bins."""
        if features is None:
            return self.rpkm
        index = features.index if isinstance(features, pd.DataFrame) else features
        return self.rpkm.reindex(columns=index)

    # --- Significance across studies --------------------------------------

    def significant_bins(
        self,
        estimate_thresh: float = 0.25,
        fdr_thresh: float = 0.2,
    ) -> pd.DataFrame:
        """
        Bins passing the significance thresholds in *every* study in the set.

        Association columns are suffixed with each study's name.
        """
        merged: pd.DataFrame | None = None
        for study in self.studies:
            df = study.significant_bins(estimate_thresh, fdr_thresh).add_suffix(f"_{study.name}")
            merged = df if merged is None else merged.join(df, how="inner")
        return merged

    # --- Community diversity ----------------------------------------------

    def tsne_plot(
        self,
        color: str,
        features: pd.MultiIndex | pd.DataFrame | None = None,
        width: int = 550,
        height: int = 400,
        show_legend: bool = True,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        t-SNE ordination of the combined samples, colored by a sample group.
        """
        coords = tsne(self.features(features)).merge(
            self.sample_metadata, left_index=True, right_index=True
        )
        fig = px.scatter(
            data_frame=coords,
            x="t-SNE 1",
            y="t-SNE 2",
            color=color,
            template="plotly_white",
            width=width,
            height=height,
            labels={"study": "Study", "disease": "Disease", "participant": "Participant"},
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(showlegend=show_legend)
        save_image(fig, file_prefix)
        return fig

    def permanova(
        self,
        features: pd.MultiIndex | pd.DataFrame | None = None,
        categories: Iterable[str] = ("disease", "participant"),
        file_prefix: str | None = None,
    ) -> pd.DataFrame:
        """
        Run PERMANOVA separately within each cohort, for each sample group.

        Returns the long-form results (one row per cohort per category). When
        ``file_prefix`` is given, a wide table is also written as CSV.
        """
        rpkm = self.features(features)
        metadata = self.sample_metadata

        results = pd.concat([
            permanova(
                scalars_df=rpkm.reindex(index=cohort_metadata.index),
                metadata_df=cohort_metadata.reindex(columns=list(categories)),
            ).assign(study=cohort)
            for cohort, cohort_metadata in metadata.groupby("study")
        ])

        if file_prefix is not None:
            path = Path(file_prefix + ".csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._permanova_wide(results).to_csv(path)

        return results

    def _permanova_wide(self, results: pd.DataFrame) -> pd.DataFrame:
        renamed = {
            "r_squared": "R^2",
            "p_value": "p-value",
            "f_statistic": "F statistic",
            "n_groups": "# Groups",
            "n_samples": "# Samples",
        }
        wide = (
            results
            .rename(columns=lambda v: renamed.get(v, v.title()))
            .pivot_table(
                columns="Category",
                index=["Study", "# Samples"],
                values=list(renamed.values()),
            )
        )
        wide.columns = pd.MultiIndex.from_tuples(
            [(metric, category) for category, metric in wide.columns],
            names=["Metric", "Category"],
        )
        return (
            wide
            .sort_index(axis=1)
            .reset_index()
            .set_index("Study")
            .reindex(index=self.study_order)
            .reset_index()
            .set_index(["Study", "# Samples"])
        )

    def compare_permanova(
        self,
        results: Dict[str, pd.DataFrame],
        metric: str = "r_squared",
        width: int = 700,
        height: int = 300,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Bar chart comparing PERMANOVA results from several feature subsets.

        ``results`` maps a label (e.g. "All bins") to the long-form output of
        :meth:`permanova`.
        """
        df = pd.concat([
            result.assign(Bins=label) for label, result in results.items()
        ])
        fig = px.bar(
            data_frame=df,
            x="study",
            y=metric,
            facet_col="category",
            color="Bins",
            barmode="group",
            template="plotly_white",
            facet_col_spacing=0.1,
            labels={
                "r_squared": "R^2",
                "p_value": "p-value",
                "study": "Study",
                "disease": "Disease",
                "participant": "Participant",
            },
            width=width,
            height=height,
            category_orders={"study": self.study_order},
        )
        if metric == "p_value":
            fig.add_hline(y=0.05, line_dash="dot", line_color="red")
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].title()))
        save_image(fig, file_prefix)
        return fig

    # --- Within-organism community types ----------------------------------

    def pangenome_clusters(
        self,
        organism: str,
        resolution: float = 1.0,
        metric: str = "braycurtis",
    ) -> pd.DataFrame:
        """
        Cluster the samples on one organism's bin composition.

        Samples with no detectable signal for the organism are dropped.
        Returns the t-SNE coordinates, the Leiden cluster label, and the
        sample metadata.
        """
        rpkm = self.weighted_rpkm[organism]
        rpkm = rpkm.loc[rpkm.max(axis=1) > 0]
        coords = tsne(rpkm).assign(cluster=leiden(rpkm, resolution=resolution, metric=metric))
        return pd.concat([coords, self.sample_metadata.reindex(index=rpkm.index)], axis=1)

    def cluster_disease_contingency(
        self,
        clusters: Dict[str, pd.DataFrame],
        width: int = 700,
        height: int = 450,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Heatmap of the association between community type and disease state,
        for each organism and cohort.
        """
        rows = []
        for organism, df in clusters.items():
            for cohort, cohort_df in df.groupby("study"):
                result = chi2_contingency_test(df=cohort_df, col_a="cluster", col_b="disease")
                rows.append({
                    "organism": organism,
                    "study": cohort,
                    "cramers_v": result["cramers_v"],
                    "label": f'V={result["cramers_v"]:.2} {_significance_stars(result["p_value"])}',
                })

        results = pd.DataFrame(rows)
        panels = {
            column: (
                results
                .pivot(index="organism", columns="study", values=column)
                .reindex(columns=self.study_order)
                .sort_index(axis=0, ascending=False)
            )
            for column in ("label", "cramers_v")
        }
        fig = go.Figure(
            data=go.Heatmap(
                x=panels["label"].columns.values,
                y=panels["label"].index.values,
                z=panels["cramers_v"],
                text=panels["label"],
                colorscale="Blues",
                colorbar_title="Cramer's V",
                texttemplate="%{text}",
            ),
            layout=dict(
                width=width,
                height=height,
                title="Chi2 Contingency: Cluster vs. Disease",
                title_x=0.5,
            ),
        )
        save_image(fig, file_prefix)
        return fig

    def cluster_tsne_plot(
        self,
        clusters: pd.DataFrame,
        width: int = 600,
        height: int = 500,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """t-SNE ordination of one organism's samples, colored by community type."""
        fig = px.scatter(
            data_frame=clusters,
            x="t-SNE 1",
            y="t-SNE 2",
            color="cluster",
            template="plotly_white",
            labels={"study": "Study", "disease": "Disease", "cluster": "Cluster"},
            category_orders={
                "study": self.study_order,
                "cluster": _sorted_clusters(clusters["cluster"]),
            },
            width=width,
            height=height,
        )
        save_image(fig, file_prefix)
        return fig

    def cluster_composition_bars(
        self,
        clusters: pd.DataFrame,
        width: int = 600,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Percentage of each cohort's cases and controls falling in each
        community type.
        """
        composition = (
            clusters
            .assign(count=1)
            .pivot_table(columns=["study", "disease"], index="cluster", values="count", aggfunc="sum")
            .fillna(0)
            .apply(lambda c: c / c.sum())
            .multiply(100.0)
            .melt(ignore_index=False)
            .reset_index()
        )
        fig = px.bar(
            data_frame=composition,
            x="cluster",
            y="value",
            color="disease",
            facet_col="study",
            facet_col_wrap=1,
            barmode="group",
            template="plotly_white",
            category_orders={
                "study": self.study_order,
                "cluster": _sorted_clusters(clusters["cluster"]),
            },
            labels={"value": "%", "cluster": "", "disease": "Disease"},
            width=width,
        )
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        save_image(fig, file_prefix)
        return fig

    # --- Feature positivity -----------------------------------------------

    def plot_feature_positivity(
        self,
        features: pd.MultiIndex | pd.DataFrame | None = None,
        samples: pd.Index | None = None,
        group: str = "disease",
        **kwargs: Any,
    ) -> go.Figure:
        """Samples ranked by how many of the given bins they carry."""
        rpkm, metadata = self._subset(features, samples)
        return plot_feature_positivity(rpkm, metadata[group], **kwargs)

    def plot_positivity_heatmap(
        self,
        features: pd.MultiIndex | pd.DataFrame | None = None,
        samples: pd.Index | None = None,
        groups: Iterable[str] = ("disease",),
        **kwargs: Any,
    ) -> go.Figure:
        """Per-sample detection of each bin, annotated by sample group."""
        rpkm, metadata = self._subset(features, samples)
        return plot_positivity_heatmap(rpkm, metadata.reindex(columns=list(groups)), **kwargs)

    def _subset(
        self,
        features: pd.MultiIndex | pd.DataFrame | None,
        samples: pd.Index | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rpkm = self.features(features)
        metadata = self.sample_metadata
        if samples is not None:
            rpkm = rpkm.reindex(index=samples)
            metadata = metadata.reindex(index=samples)
        rpkm = rpkm.set_axis(rpkm.columns.map(" - ".join), axis=1)
        return rpkm, metadata


def _significance_stars(p_value: float) -> str:
    for threshold, stars in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p_value < threshold:
            return stars
    return ""


def _sorted_clusters(values: pd.Series) -> List[str]:
    labels = values.drop_duplicates().tolist()
    labels.sort(key=lambda name: int(name.split(" ")[1]))
    return labels
