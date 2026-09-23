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
from ..helpers.supervised import fit_classifier
from .study import Study

#: Sample groups every study in a set is expected to define.
REQUIRED_SAMPLE_GROUPS = ("study", "cohort", "disease", "participant")


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

    def _order(self, group: str) -> List[str]:
        order: List[str] = []
        for study in self.studies:
            for name in study.group_order(group):
                if name not in order:
                    order.append(name)
        return order

    @cached_property
    def study_order(self) -> List[str]:
        """Study names in the order declared by the studies."""
        return self._order("study")

    @cached_property
    def cohort_order(self) -> List[str]:
        """Cohort names in the order declared by the studies."""
        return self._order("cohort")

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
            .assign(category=results["category"].str.title())
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

    # --- Bin abundance by cohort ------------------------------------------

    def bin_proportion_boxplot(
        self,
        features: pd.MultiIndex | pd.DataFrame,
        floor: float = 0.001,
        width: int = 500,
        height: int = 1200,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Abundance of each bin relative to its organism's core genome, split by
        cohort and by case/control status.

        Dividing by the core genome turns RPKM into the proportion of that
        organism's genomes in the sample which carry the bin, which is
        comparable across cohorts of differing sequencing depth.
        """
        index = features.index if isinstance(features, pd.DataFrame) else features

        panels = []
        for study in self.studies:
            core_genomes = study.core_genomes
            metadata = study.sample_metadata(["cohort", "disease"])
            for organism, bin_id in index:
                rpkm = study.contrast(organism).rpkm
                panels.append(pd.DataFrame({
                    "Cohort": metadata["cohort"],
                    "Group": metadata["disease"],
                    "Proportion": (rpkm[bin_id] / rpkm[core_genomes[organism]]).clip(lower=floor),
                    "Feature": f"{organism} - {bin_id}",
                }))

        fig = px.box(
            data_frame=pd.concat(panels),
            x="Cohort",
            y="Proportion",
            color="Group",
            template="plotly_white",
            log_y=True,
            facet_col="Feature",
            facet_col_wrap=1,
            category_orders={"Cohort": self.cohort_order, "Group": self._order("disease")},
            width=width,
            height=height,
            range_y=(floor, 1),
        )
        fig.for_each_annotation(
            lambda a: a.update(x=0.02, xanchor="left", text=a.text.split("=", 1)[1])
        )
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


    # --- Supervised models ------------------------------------------------

    def classify_by_organism(
        self,
        n_replicates: int = 10,
        n_interaction_features: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        """
        For each organism, fit a classifier per study separating cases from
        controls on that organism's bin abundance.

        Returns three tables: ``replicates`` (validation ROC-AUC per split),
        ``shap`` (mean absolute SHAP per bin, one column per study, plus their
        geometric mean), and ``interactions`` (pairwise SHAP interaction
        strengths among each model's most important bins).
        """
        organisms = [
            organism for organism in self.studies[0].organisms
            if all(organism in study.contrast_dirs for study in self.studies)
        ]

        replicates, shap_panels, interactions = [], [], []
        for organism in organisms:
            per_study = {}
            for study in self.studies:
                contrast = study.contrast(organism)
                result = fit_classifier(
                    contrast.abund,
                    contrast.metadata[study.parameter],
                    n_replicates=n_replicates,
                    n_interaction_features=n_interaction_features,
                )
                if result is None:
                    continue
                per_study[study.label] = result.shap
                replicates.extend(
                    {
                        "organism": organism,
                        "study": study.label,
                        "replicate": replicate,
                        "roc_auc": auc,
                        "n_samples": result.n_samples,
                    }
                    for replicate, auc in enumerate(result.replicate_auc)
                )
                if result.interactions is not None:
                    interactions.append(
                        result.interactions
                        .rename_axis(index="bin_a", columns="bin_b")
                        .stack()
                        .rename("mean_abs_interaction")
                        .reset_index()
                        .assign(organism=organism, study=study.label)
                    )

            # A bin can only be compared across studies if every study modelled it
            if len(per_study) == len(self.studies):
                shap_panels.append(
                    pd.DataFrame(per_study).fillna(0.0).rename_axis(index="bin")
                    .assign(organism=organism)
                    .reset_index()
                )

        shap = pd.concat(shap_panels, ignore_index=True)
        labels = [study.label for study in self.studies]
        shap["combined"] = np.exp(np.log(shap[labels].clip(lower=1e-12)).mean(axis=1))

        return {
            "replicates": pd.DataFrame(replicates),
            "shap": shap.sort_values("combined", ascending=False),
            "interactions": pd.concat(interactions, ignore_index=True),
        }

    def plot_classifier_performance(
        self,
        replicates: pd.DataFrame,
        width: int = 1000,
        height: int = 520,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """ROC-AUC per organism and study, with the spread across replicates."""
        summary = (
            replicates
            .groupby(["organism", "study"])
            .agg(auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", "std"))
            .reset_index()
        )
        fig = px.bar(
            data_frame=summary,
            x="organism",
            y="auc_mean",
            error_y="auc_std",
            color="study",
            barmode="group",
            template="plotly_white",
            labels={"organism": "Organism", "auc_mean": "ROC-AUC", "study": "Study"},
            category_orders={"study": [study.label for study in self.studies]},
            width=width,
            height=height,
        )
        fig.add_hline(
            y=0.5, line_dash="dash", line_color="grey",
            annotation_text="chance", annotation_position="bottom right",
        )
        fig.update_yaxes(range=[0.4, 1.05])
        fig.update_xaxes(tickangle=-35)
        save_image(fig, file_prefix)
        return fig

    def plot_shap_comparison(
        self,
        shap: pd.DataFrame,
        organism: str,
        n_labelled: int = 5,
        width: int = 650,
        height: int = 560,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Importance of one organism's bins in each study's model, against each
        other. Bins near the diagonal carry signal in both.
        """
        if len(self.studies) != 2:
            raise ValueError("plot_shap_comparison compares exactly two studies")
        x_label, y_label = (study.label for study in self.studies)

        df = shap.loc[shap["organism"] == organism].sort_values("combined", ascending=False)
        limit = max(df[x_label].max(), df[y_label].max()) * 1.08

        fig = px.scatter(
            data_frame=df,
            x=x_label,
            y=y_label,
            hover_name="bin",
            hover_data={"combined": ":.5f"},
            template="plotly_white",
            labels={
                x_label: f"mean |SHAP| - {x_label}",
                y_label: f"mean |SHAP| - {y_label}",
            },
            title=f"{organism} - bin importance in each study",
            width=width,
            height=height,
            range_x=[0, limit],
            range_y=[0, limit],
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=limit, y1=limit,
                      line=dict(color="lightgrey", dash="dash", width=1))
        top = df.head(n_labelled)
        fig.add_trace(go.Scatter(
            x=top[x_label], y=top[y_label], mode="text", text=top["bin"],
            textposition="top right", textfont=dict(size=9),
            showlegend=False, hoverinfo="skip",
        ))
        save_image(fig, file_prefix)
        return fig

    def plot_interaction_heatmap(
        self,
        interactions: pd.DataFrame,
        organism: str,
        study_label: str,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """Interaction strength between the most important bins of one model."""
        matrix = (
            interactions
            .loc[(interactions["organism"] == organism) & (interactions["study"] == study_label)]
            .pivot(index="bin_a", columns="bin_b", values="mean_abs_interaction")
        )
        matrix = matrix.reindex(index=matrix.columns)
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                colorscale="Magma",
                colorbar_title="mean |interaction|",
                hovertemplate="%{y} x %{x}: %{z:.4g}<extra></extra>",
            ),
            layout=dict(
                title=f"{organism} - {study_label}: bin interactions",
                xaxis=dict(tickangle=45),
                height=max(300, 28 * len(matrix) + 120),
                width=max(400, 28 * len(matrix) + 260),
            ),
        )
        save_image(fig, file_prefix)
        return fig


def _significance_stars(p_value: float) -> str:
    for threshold, stars in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p_value < threshold:
            return stars
    return ""


def _sorted_clusters(values: pd.Series) -> List[str]:
    labels = values.drop_duplicates().tolist()
    labels.sort(key=lambda name: int(name.split(" ")[1]))
    return labels
