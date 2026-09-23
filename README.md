# gig-map-io

Python library for reading the outputs of the [gig-map](https://github.com/fredhutch/gig-map)
workflow (genes-in-genomes map) and running the analyses built on them.

## Overview

- **Reader objects** — one per gig-map workflow, each reading that workflow's
  key outputs:
  - **`Pangenome`** — gene bins, genome content, gene coordinates, ANI distances
  - **`ContrastMetagenomes`** — association results, bin abundance, sample metadata
  - **`PangenomePhylogeny`** — per-bin gene trees

  Each has a `*Set` counterpart (`PangenomeSet`, `ContrastMetagenomesSet`,
  `PangenomePhylogenySet`) wrapping a dict of readers keyed by organism, so an
  analysis can span several pangenomes at once.

- **`Study`** — names the gig-map outputs belonging to one comparison, keyed by
  organism, along with the recoding that turns raw contrast metadata into
  labelled sample groups. A study is serialized as JSON holding relative paths,
  so a definition can be checked in next to the analysis code and re-resolved
  against a downloaded copy of the data. **`StudySet`** spans several studies
  for community-level analyses.

- **Analysis methods** on those objects, rather than free functions: volcano
  plots and contrast comparisons, pangenome summaries and gene maps, community
  ordination and PERMANOVA, Leiden community types, and supervised models with
  SHAP attribution. Every plotting method takes a `file_prefix` and writes PNG,
  trimmed PNG, PDF, HTML and JSON.

## Installation

```bash
pip install "gig-map-io @ git+https://github.com/sminot/gig-map-io.git"
```

For development:

```bash
git clone https://github.com/sminot/gig-map-io.git
cd gig-map-io
pip install -e ".[dev]"
```

## Quick start

### Reader objects

```python
from pathlib import Path
from gig_map_io import Pangenome, ContrastMetagenomes, PangenomePhylogeny

pangenome = Pangenome(directory=Path("/path/to/pangenome"))
pangenome.gene_bins
pangenome.genome_content
pangenome.core_genome()

contrast = ContrastMetagenomes(directory=Path("/path/to/contrast"), parameter="disease")
contrast.association
contrast.rpkm
contrast.metadata
```

### Studies

A study definition lists the dataset directories that belong together:

```json
{
  "name": "gvhd_combined",
  "label": "GvHD - Combined",
  "parameter": "disease",
  "contrasts": {"Alistipes": "datasets/Contrast - .../data"},
  "pangenomes": {"Alistipes": "datasets/Pangenome - .../data"},
  "phylogenies": {"Alistipes": "datasets/Phylogenies - .../data"},
  "sample_groups": {
    "disease": {
      "sources": [{"column": "disease", "labels": {"1": "Case", "0": "Control"}}],
      "order": ["Case", "Control"]
    }
  }
}
```

Paths are resolved relative to the current working directory (or to a `base`
you pass), so a definition travels with the data rather than with the machine
it was written on.

```python
from gig_map_io import Study

study = Study.from_json("studies/gvhd_combined.json")

study.volcano_plot(max_abs_estimate=2.5, file_prefix="out/volcano")
study.significant_bins(estimate_thresh=0.25, fdr_thresh=0.2, direction="negative")
study.describe_bins(bins, odds_by="cohort")
study.bin_gene_map("Alistipes", "Bin 100", file_prefix="out/gene_map")
```

Comparisons take another study, and label themselves from the two studies'
labels:

```python
ibd = Study.from_json("studies/ibd_combined.json")

study.compare_sig_scatter(ibd, file_prefix="out/scatter")
study.compare_sig_categories(ibd, file_prefix="out/categories")
study.compare_volcano_with_estimate(ibd, file_prefix="out/joint_volcano")
```

### Sample groups

Cohorts encode the same facts differently — one study's `batch_1` is another's
`study_name`, and the study a contrast was assembled around often leaves its
own samples unlabelled. A `SampleGroup` describes that recoding declaratively:
one or more source columns tried in order, the labels to map their values onto,
an optional `default` for nulls, and the category order to use in plots.

```python
study.sample_metadata(["study", "cohort", "disease", "participant"])
study.group_order("cohort")
```

### Several studies at once

```python
from gig_map_io import StudySet

studies = StudySet.from_json("studies/gvhd_combined.json", "studies/ibd_combined.json")

studies.significant_bins()                          # bins significant in every study
studies.tsne_plot("disease", file_prefix="out/tsne")
studies.permanova(file_prefix="out/permanova")
studies.pangenome_clusters("Alistipes")             # Leiden community types
studies.classify_by_organism()                      # XGBoost + SHAP per organism
```

## Reproducibility

Anything that subsamples takes a `random_state` and defaults to a fixed seed,
so re-running a figure reproduces it. Note that t-SNE and Leiden are both
sensitive to the order of rows and columns in the input, so a study definition
that lists its organisms in a stable order matters for more than tidiness.

## License

See the LICENSE file for details.
