import logging
import re
from typing import Dict, Iterable, List

from Bio.Phylo.BaseTree import Tree, Clade
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .genomes import genbank_duplicates
from .save_image import save_image
from .style import SIMPLE_TEMPLATE, TOP_LEGEND

logger = logging.getLogger(__name__)

TANGLEGRAM_WIDTH = 720
TANGLEGRAM_HEIGHT_RANGE = (380, 700)
TANGLEGRAM_PX_PER_LEAF = 12
#: compare_several: the canvas, the fewest leaves a stacked tree's band is
#: sized for, and the gap between bands, which holds each one's title and axis
SEVERAL_TANGLEGRAM_SIZE = (620, 900)
MIN_BAND_LEAVES = 6
BAND_GAP = 0.12
#: The height a leaf joined to no stacked tree takes, against 1 for one that is
UNJOINED_LEAF_WEIGHT = 0.15
#: A leaf's point, and the larger triangle of a leaf standing for a collapsed clade
LEAF_MARKER_PX = 7
COLLAPSED_MARKER_PX = 11


class Phylogeny:
    """
    Helper object used to coordinate a phylogeny of genomes.

    NCBI publishes many assemblies under both a GCA_ (GenBank) and a GCF_
    (RefSeq) accession, and a pangenome built from both carries each such
    genome twice: two identical leaves. The GenBank copy is pruned on
    construction, so every tree drawn shows each genome once.
    """
    name: str
    tree: Tree
    #: Leaves standing for a clade collapsed by ``subtree``, and the leaves it held
    collapsed: Dict[str, List[str]]

    def __init__(self, name: str, tree: Tree) -> None:
        self.name = name
        self.tree = tree
        self.collapsed = {}
        leaves = {leaf.name: leaf for leaf in self.tree.get_terminals()}
        for leaf in sorted(genbank_duplicates(leaves)):
            self.tree.prune(leaves[leaf])

        # Find the coordinates
        self.find_coords()

        # Get the children of each node
        self.children = {}
        self._get_children(self.tree.clade)

    def find_coords(self, scale: float = 1.0, weights: Dict[str, float] | None = None) -> None:
        """
        Place every node. Each leaf takes a share of the height in proportion
        to its weight in ``weights``, 1 where it has none, so that leaves of
        less interest can be packed closer; ``span`` is the height taken.
        """
        # Get the X-Y position of each node (framing the whole tree from 0-1)
        self.coords = {}
        self._weights = weights or {}

        self._clade_ix = 0

        # Start at the root
        self.span = self._weight(self.tree.clade) * scale
        self._add_coord(self.tree.clade, -0.5, self.span - 0.5)

    def _weight(self, clade: Clade) -> float:
        return sum(self._weights.get(leaf.name, 1.0) for leaf in clade.get_terminals())

    def _add_coord(self, clade: Clade, start: float, stop: float) -> None:

        # Label unnamed nodes
        if clade.name is None:
            clade.name = f"clade ({self._clade_ix})"
            self._clade_ix += 1

        # If the clade is terminal, put it in the middle
        if clade.is_terminal():
            y=np.mean([start, stop])

        # An internal node sits at the mean of its children, each of which
        # takes a share of the span proportional to its leaves' weight
        else:
            span = stop - start
            previous_y = start
            child_ys = []
            for child in clade.clades:
                new_y = previous_y + (span * self._weight(child) / self._weight(clade))

                self._add_coord(
                    child,
                    previous_y,
                    new_y
                )
                child_ys.append(np.mean([previous_y, new_y]))
                previous_y = new_y

            # Calculate the y as the mean of the position of each child
            y = np.mean(child_ys)

        self.coords[clade.name] = dict(
            x=self.tree.depths().get(clade, 0),
            y=y
        )            

    def _get_children(self, clade):
        self.children[clade.name] = [child.name for child in clade.clades]
        for child in clade.clades:
            if not child.is_terminal():
                self._get_children(child)

    def plot_lines(self, fig, row=1, col=1):

        # For each internal node, draw a line to its children
        for parent, children in self.children.items():
            for child in children:
                fig.add_trace(
                    go.Scatter(
                        x=[
                            self.coords[parent]['x'],
                            self.coords[parent]['x'],
                            self.coords[child]['x']
                        ],
                        y=[
                            self.coords[parent]['y'],
                            self.coords[child]['y'],
                            self.coords[child]['y']
                        ],
                        mode="lines",
                        showlegend=False,
                        line=dict(color="black", width=1)
                    ),
                    row=row,
                    col=col
                )

    def plot_points(self, fig, mode: str, row=1, col=1):
        # Draw each terminal node
        fig.add_trace(
            go.Scatter(
                x=self._get_coord('x'),
                y=self._get_coord('y'),
                text=[node.name for node in self.tree.get_terminals()],
                mode=mode,
                showlegend=False,
                textposition="middle right",
                marker=dict(color="black", size=4),
                cliponaxis=False
            ),
            row=row,
            col=col
        )

    def plot_leaf_groups(
        self, fig, groups: Dict[str, str], colors: Dict[str, str], row=1, col=1, showlegend=False,
    ):
        """
        Each leaf as a point in the color of its group in ``groups``, one trace
        and legend entry per group. A leaf standing for a collapsed clade is a
        triangle, labeled with how many genomes it holds.
        """
        leaves = self.leaves_list
        for group, color in colors.items():
            members = [leaf for leaf in leaves if groups.get(leaf) == group]
            if not members:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[self.coords[leaf]["x"] for leaf in members],
                    y=[self.coords[leaf]["y"] for leaf in members],
                    text=[str(len(self.collapsed[leaf])) if leaf in self.collapsed else "" for leaf in members],
                    hovertext=members,
                    mode="markers+text",
                    textposition="middle right",
                    textfont=dict(size=10, color="#444444"),
                    name=group,
                    legendgroup=group,
                    showlegend=showlegend,
                    # Sizes given per point make plotly draw a bubble chart,
                    # which defaults to 70% opacity and a white outline
                    marker=dict(
                        color=color,
                        size=[COLLAPSED_MARKER_PX if leaf in self.collapsed else LEAF_MARKER_PX for leaf in members],
                        symbol=["triangle-left" if leaf in self.collapsed else "circle" for leaf in members],
                        opacity=1,
                        line_width=0,
                    ),
                    cliponaxis=False,
                ),
                row=row,
                col=col,
            )

    def _get_coord(self, kw: str):
        """Get a particular value for every item in the tree."""
        return [
            self.coords[node.name][kw]
            for node in self.tree.get_terminals()
        ]

    def _plot_tracer(self, fig, use_nodes, row=1, col=1):
        # Draw a line from each terminal node to the edge of the graph
        edge = np.max(self._get_coord("x"))
        for node_name in use_nodes:
            fig.add_trace(
                go.Scatter(
                    x=[self.coords[node_name]['x'], edge],
                    y=[self.coords[node_name]['y'], self.coords[node_name]['y']],
                    mode="lines",
                    showlegend=False,
                    line=dict(dash='dot', color="#b0b0b0", width=1),
                    cliponaxis=False
                ),
                row=row,
                col=col
            )

    def align_trees(self, comp: 'Phylogeny'):
        """Swap children at internal nodes wherever that brings the leaf order closer to ``comp``'s."""
        logger.info("Aligning %s to %s", self.name, comp.name)
        for _ in range(50):
            made_switch = False
            for node in self.tree.get_nonterminals():
                for i in range(len(node.clades) - 1):
                    for j in range(i + 1, len(node.clades)):
                        score = self._score_tree_alignment(comp)
                        node.clades[i], node.clades[j] = node.clades[j], node.clades[i]
                        if self._score_tree_alignment(comp) > score:
                            made_switch = True
                        else:
                            node.clades[i], node.clades[j] = node.clades[j], node.clades[i]
            if not made_switch:
                break

    def _score_tree_alignment(self, comp: 'Phylogeny'):
        self_leaf_order = self._leaf_order()
        comp_leaf_order = comp._leaf_order()
        shared = sorted(set(self_leaf_order.keys()) & set(comp_leaf_order.keys()))
        res = stats.spearmanr(
            [self_leaf_order[i] for i in shared],
            [comp_leaf_order[i] for i in shared]
        )
        return res.statistic

    def _leaf_order(self):
        return {
            node.name: i
            for i, node in enumerate(self.tree.get_terminals())
        }

    @property
    def n_leaves(self) -> int:
        return len(self.tree.get_terminals())

    @property
    def leaves_list(self) -> List[str]:
        return [node.name for node in self.tree.get_terminals()]

    def compare(
        self,
        comp: 'Phylogeny',
        file_prefix: str | None = None
    ):
        """
        A tanglegram: this tree on the left, ``comp`` on the right with its
        leaves reordered to follow this one, and the shared leaves joined.
        Both trees span the full height, so the height follows ``comp``'s
        leaf count: this tree is the core genome, hundreds of leaves that
        are read as clades, while ``comp`` is a bin's tree of a few dozen
        whose individual leaves need room.
        """
        # Sorted, not merely deduplicated: this order is the order the tracer
        # lines are added to the figure, and set iteration order varies between
        # processes, which made the saved figure specification differ run to run.
        shared_nodes = sorted(set(self._get_leafs(self.tree)) & set(comp._get_leafs(comp.tree)))
        if len(shared_nodes) < 3:
            raise ValueError("Not enough shared genomes to compare.")

        comp.align_trees(self)
        self.find_coords()
        comp.find_coords(scale=self.n_leaves / comp.n_leaves)

        fig = make_subplots(
            cols=3,
            rows=1,
            subplot_titles=(self.name, None, comp.name),
            shared_yaxes=True,
            horizontal_spacing=0.,
            vertical_spacing=0.,
            column_widths=[2, 1, 2]
        )

        self.plot_lines(fig)
        self.plot_points(fig, mode="markers")
        self._plot_tracer(fig, shared_nodes)

        comp.plot_lines(fig, row=1, col=3)
        comp.plot_points(fig, mode="markers", row=1, col=3)
        comp._plot_tracer(fig, shared_nodes, row=1, col=3)

        # Draw lines between each shared leaf
        for node_name in shared_nodes:
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[self.coords[node_name]['y'], comp.coords[node_name]['y']],
                    mode="lines",
                    showlegend=False,
                    line=dict(dash='dot', color="#b0b0b0", width=1),
                    cliponaxis=False
                ),
                row=1,
                col=2
            )

        blank_axis = dict(
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False
        )
        snp_rate_axis = dict(
            automargin=True,
            title_text="SNP rate",
            exponentformat="power",
            # Few enough to stay level under half of a printed tanglegram
            nticks=4,
        )
        margin = dict(l=50, r=30, b=70, t=60)

        fig.update_layout(
            template=SIMPLE_TEMPLATE,
            yaxis=blank_axis,
            yaxis2=blank_axis,
            yaxis3=blank_axis,
            xaxis=snp_rate_axis,
            xaxis2=blank_axis,
            xaxis3=dict(**snp_rate_axis, autorange="reversed"),
            # Leaves are drawn as points, not labels, so no label margin
            margin=margin,
            height=int(np.clip(
                margin["t"] + margin["b"] + TANGLEGRAM_PX_PER_LEAF * comp.n_leaves,
                *TANGLEGRAM_HEIGHT_RANGE
            )),
            width=TANGLEGRAM_WIDTH
        )

        save_image(fig, file_prefix)

        return fig


    def subtree(self, leaves: Iterable[str], collapse_others: bool = False) -> 'Phylogeny':
        """
        The clade at the most recent common ancestor of ``leaves``, as a tree
        of its own. It takes that clade from this tree rather than a copy,
        which for a tree of hundreds of leaves exceeds the recursion limit,
        so this tree should not be drawn afterward.

        With ``collapse_others``, every largest clade holding none of
        ``leaves`` becomes one leaf, at its leaves' mean depth, named for how
        many genomes it holds; ``collapsed`` maps each such leaf to them.
        """
        leaves = set(leaves)
        clade = self.tree.common_ancestor(sorted(leaves))
        # The clade's own branch is not part of it
        clade.branch_length = 0.0
        collapsed = {}
        if collapse_others:
            stack = [clade]
            while stack:
                node = stack.pop()
                for i, child in enumerate(node.clades):
                    members = sorted(leaf.name for leaf in child.get_terminals())
                    if len(members) > 1 and leaves.isdisjoint(members):
                        depths = child.depths()
                        name = f"{len(members)} genomes ({len(collapsed) + 1})"
                        node.clades[i] = Clade(
                            branch_length=float(np.mean([depths[leaf] for leaf in child.get_terminals()])),
                            name=name,
                        )
                        collapsed[name] = members
                    elif not child.is_terminal():
                        stack.append(child)
        phylogeny = Phylogeny(self.name, Tree(root=clade, rooted=True))
        phylogeny.collapsed = collapsed
        return phylogeny

    def compare_several(
        self,
        comps: List['Phylogeny'],
        leaf_groups: Dict[str, str],
        group_colors: Dict[str, str],
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Several tanglegrams sharing one tree: this tree on the left, drawn once,
        and each of ``comps`` stacked on the right in a band as tall as its
        leaves need, its leaves reordered to follow this tree and joined to it.
        Leaves on both sides take the color of their group in ``leaf_groups``
        (a species, say), in the colors and legend order of ``group_colors``.
        Leaves of this tree in none of ``comps`` take a fraction of the
        height (``UNJOINED_LEAF_WEIGHT``), so a tree of hundreds gives the
        joined ones room while still showing how many there are.
        Each of ``comps`` keeps its own distance axis: trees of different bins
        can differ in depth tenfold.
        """
        for comp in comps:
            if len(set(self.leaves_list) & set(comp.leaves_list)) < 3:
                raise ValueError(f"{comp.name} shares fewer than three genomes with {self.name}")
            comp.align_trees(self)
            comp.find_coords()
        # Leaves in none of comps are packed closer, so the shared ones have room
        joined = set().union(*(c.leaves_list for c in comps)) | set(self.collapsed)
        self.find_coords(weights={
            leaf: 1.0 if leaf in joined else UNJOINED_LEAF_WEIGHT for leaf in self.leaves_list
        })

        n = len(comps)
        fig = make_subplots(
            rows=n,
            cols=3,
            specs=[[{"rowspan": n}, {"rowspan": n}, {}]] + [[None, None, {}] for _ in range(n - 1)],
            column_widths=[2, 1, 1.4],
            row_heights=[max(comp.n_leaves, MIN_BAND_LEAVES) for comp in comps],
            horizontal_spacing=0.,
            vertical_spacing=BAND_GAP,
            subplot_titles=[self.name, "", *(comp.name for comp in comps)],
        )
        leaves = self.span
        self.plot_lines(fig)
        self.plot_leaf_groups(fig, leaf_groups, group_colors, showlegend=True)
        self._plot_tracer(fig, sorted(set(self.leaves_list) & set().union(*(c.leaves_list for c in comps))))
        fig.update_yaxes(range=[-0.5, leaves - 0.5], row=1, col=1)
        fig.update_yaxes(range=[-0.5, leaves - 0.5], row=1, col=2)
        fig.update_xaxes(range=[0, 1], row=1, col=2)

        # The joining lines run in the middle column, on this tree's scale; a
        # stacked tree's leaf is carried there through its band's place on the page
        middle = fig.layout[fig.get_subplot(1, 2).yaxis.plotly_name].domain
        for row, comp in enumerate(comps, start=1):
            comp.plot_lines(fig, row=row, col=3)
            comp.plot_leaf_groups(fig, leaf_groups, group_colors, row=row, col=3)
            shared = sorted(set(self.leaves_list) & set(comp.leaves_list))
            comp._plot_tracer(fig, shared, row=row, col=3)
            fig.update_yaxes(range=[-0.5, comp.n_leaves - 0.5], row=row, col=3)
            band = fig.layout[fig.get_subplot(row, 3).yaxis.plotly_name].domain
            for leaf in shared:
                page = band[0] + (comp.coords[leaf]["y"] + 0.5) / comp.n_leaves * (band[1] - band[0])
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[self.coords[leaf]["y"], (page - middle[0]) / (middle[1] - middle[0]) * leaves - 0.5],
                        mode="lines",
                        showlegend=False,
                        line=dict(dash="dot", color="#b0b0b0", width=1),
                        cliponaxis=False,
                    ),
                    row=1,
                    col=2,
                )

        snp_rate_axis = dict(automargin=True, exponentformat="power", nticks=3)
        fig.update_yaxes(visible=False)
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_xaxes(**snp_rate_axis, title_text="SNP rate", row=1, col=1)
        for row in range(1, n + 1):
            fig.update_xaxes(**snp_rate_axis, autorange="reversed", row=row, col=3)
        fig.update_xaxes(title_text="SNP rate", row=n, col=3)
        width, height = SEVERAL_TANGLEGRAM_SIZE
        fig.update_layout(
            template=SIMPLE_TEMPLATE,
            width=width,
            height=height,
            margin=dict(l=40, r=30, b=70, t=90),
            legend=dict(TOP_LEGEND, yref="container", y=1.0, yanchor="top"),
        )
        save_image(fig, file_prefix)
        return fig

    def _get_leafs(self, node: Tree):
        return [leaf.name for leaf in node.get_terminals()]

    def newick(self, rename: Dict[str, str] | None = None) -> str:
        """
        The tree as Newick, with leaves renamed through ``rename`` where it
        names them. The renaming is done in the text, since copying a tree of
        thousands of leaves exceeds the recursion limit.
        """
        newick = self.tree.format("newick")
        for leaf in self.tree.get_terminals():
            if rename and leaf.name in rename:
                newick = re.sub(
                    rf"(?<=[(,]){re.escape(leaf.name)}(?=:)",
                    _newick_safe(rename[leaf.name]).replace("\\", r"\\"),
                    newick,
                )
        return newick


def _newick_safe(name: str) -> str:
    """A leaf label with the characters Newick reserves replaced."""
    return name.replace("(", "[").replace(")", "]").replace(",", " ").replace(":", " ").replace(";", " ").replace("'", "")
