"""
Entry marimo notebook for gig-map-io.

This notebook is a stub that can be extended with real analyses of gig-map
outputs. It is launched via the `gig-map-io launch-notebooks` CLI command.
"""

import marimo


app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        """
        # gig-map-io

        This is a placeholder marimo notebook shipped with the `gig-map-io`
        package. Extend this notebook with cells that load and explore outputs
        from the gig-map workflow using the core classes:

        - `Pangenome`
        - `ContrastMetagenomes`
        - `PangenomeBin`
        """
    )
    return


if __name__ == "__main__":
    app.run()

