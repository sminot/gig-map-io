# gig-map-io

Python library for parsing and analyzing the outputs of gig-map (genes-in-genomes map) workflow.

## Features

- **Core Classes**: `Pangenome`, `ContrastMetagenomes`, and `PangenomeBin` for working with gig-map outputs
- **Catalog**: Track and manage multiple pangenomes and contrasts with automatic persistence
- **CLI Tools**: Launch interactive marimo notebooks for data exploration
- **Simple API**: Direct filesystem access with pandas integration

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

## Quick Start

```python
from pathlib import Path
from gig_map_io import Pangenome, Catalog

# Work with a pangenome
pang = Pangenome(directory=Path("/path/to/pangenome"))
gpa = pang.gene_presence_absence()

# Track multiple outputs with Catalog
catalog = Catalog()
catalog.add_pangenome(Path("/path/to/pangenome1"), "pangenome1")
```

## License

See LICENSE file for details.
