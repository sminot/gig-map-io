"""
Links to the interactive Newick viewer.

The viewer reads its whole state from the URL fragment, so a tanglegram of two
trees can be shared as a single link with no server behind it.
"""

from __future__ import annotations

import json

import lzstring

NEWICK_VIEWER_URL = "https://sminot.github.io/newick-viewer/"


def newick_viewer_url(
    newick1: str,
    newick2: str,
    title: str,
    canvas_width: int = 700,
    canvas_height: int = 400,
) -> str:
    """Build a link that renders two Newick trees as a tanglegram."""
    state = {
        "newick1": newick1,
        "newick2": newick2,
        "tanglegram": True,
        "style": {
            "figureTitle": title,
            "canvasWidth": canvas_width,
            "canvasHeight": canvas_height,
        },
        "tanglegramStyle": {"showLeafLabels2": False, "heightScaler": -1},
    }
    encoded = lzstring.LZString().compressToEncodedURIComponent(json.dumps(state))
    return f"{NEWICK_VIEWER_URL}#s={encoded}"
