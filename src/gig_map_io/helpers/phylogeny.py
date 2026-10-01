import logging
import re
from typing import Dict, List

from Bio.Phylo.BaseTree import Tree, Clade
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .genomes import genbank_duplicates
from .save_image import save_image
from .style import SIMPLE_TEMPLATE

logger = logging.getLogger(__name__)


class Phylogeny:
    """
    Helper object used to coordinate a phylogeny.
    """
    name: str
    tree: Tree
    distances: Dict[str, Dict[str, float]]

    def __init__(
        self,
        name: str,
        tree: Tree,
        distances: pd.DataFrame | None = None,
    ) -> None:
        self.name = name
        self.tree = tree
        terminals = self.tree.get_terminals()

        if distances is None:
            self.distances = {
                n1.name: {
                    n2.name: self.tree.distance(n1, n2)
                    for n2 in terminals
                }
                for n1 in terminals
            }

        else:
            self.distances = {
                n: r.to_dict()
                for n, r in distances.iterrows()
            }

        # Find the coordinates
        self.find_coords()

        # Get the children of each node
        self.children = {}
        self._get_children(self.tree.clade)

    def find_coords(self, scale: float = 1.0) -> None:

        # Get the X-Y position of each node (framing the whole tree from 0-1)
        self.coords = {}

        self._clade_ix = 0

        # Start at the root
        self._add_coord(self.tree.clade, -0.5, (self.n_leaves * scale) - 0.5)

    def _add_coord(self, clade: Clade, start: float, stop: float) -> None:

        # Label unnamed nodes
        if clade.name is None:
            clade.name = f"clade ({self._clade_ix})"
            self._clade_ix += 1

        # If the clade is terminal, put it in the middle
        if clade.is_terminal():
            y=np.mean([start, stop])

        # An internal node sits at the mean of its children, each of which
        # takes a share of the span proportional to its leaves
        else:
            span = stop - start
            previous_y = start
            child_ys = []
            for child in clade.clades:
                new_y = previous_y + (span * len(child.get_terminals()) / len(clade.get_terminals()))

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

    def plot_lines(self, fig, row=1, col=1, y_offset=0):

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
                            self.coords[parent]['y'] + y_offset,
                            self.coords[child]['y'] + y_offset,
                            self.coords[child]['y'] + y_offset
                        ],
                        mode="lines",
                        showlegend=False,
                        line=dict(color="black", width=1)
                    ),
                    row=row,
                    col=col
                )

    def plot_points(self, fig, mode: str, row=1, col=1, y_offset=0.):
        # Draw each terminal node
        fig.add_trace(
            go.Scatter(
                x=self._get_coord('x'),
                y=self._get_coord('y', offset=y_offset),
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

    def _get_coord(self, kw: str, offset=0.):
        """Get a particular value for every item in the tree."""
        return [
            self.coords[node.name][kw] + offset
            for node in self.tree.get_terminals()
        ]

    def _plot_tracer(self, fig, use_nodes, row=1, col=1, y_offset=0.):
        # Draw a line from each terminal node to the edge of the graph
        edge = np.max(self._get_coord("x"))
        for node_name in use_nodes:
            fig.add_trace(
                go.Scatter(
                    x=[self.coords[node_name]['x'], edge],
                    y=[
                        self.coords[node_name]['y'] + y_offset,
                        self.coords[node_name]['y'] + y_offset
                    ],
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
        height: int,
        width: int,
        file_prefix: str | None = None
    ):
        """
        A tanglegram: this tree on the left, ``comp`` on the right with its
        leaves reordered to follow this one, and the shared leaves joined.
        """
        # Sorted, not merely deduplicated: this order is the order the tracer
        # lines are added to the figure, and set iteration order varies between
        # processes, which made the saved figure specification differ run to run.
        shared_nodes = sorted(set(self._get_leafs(self.tree)) & set(comp._get_leafs(comp.tree)))
        if len(shared_nodes) < 3:
            raise ValueError("Not enough shared genomes to compare.")

        comp.align_trees(self)
        self.find_coords()
        comp.find_coords()

        # Scale the comparison tree so that the shared leaves span the same height
        if len(shared_nodes) > 1:

            # Get the y-span for just the shared nodes
            self_shared_y = [self.coords[node]['y'] for node in shared_nodes]
            self_y_span = np.max(self_shared_y) - np.min(self_shared_y)

            comp_shared_y = [comp.coords[node]['y'] for node in shared_nodes]
            comp_y_span = np.max(comp_shared_y) - np.min(comp_shared_y)

            # Set the scale so that the spans will equal
            scale = self_y_span / comp_y_span

            # Regenerate the coordinates for the second tree
            comp.find_coords(scale=scale)

            # Set the offset so that the bottom node lines up
            y_offset = np.min([
                self.coords[node]['y']
                for node in shared_nodes
            ])

        else:

            # For every shared node, find the average y offset
            y_offset = np.mean([
                self.coords[node]['y'] - comp.coords[node]['y']
                for node in shared_nodes
            ])

        # Plot the two trees against each other

        # Set up a figure with two subplots
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

        comp.plot_lines(fig, row=1, col=3, y_offset=y_offset)
        comp.plot_points(fig, mode="markers", row=1, col=3, y_offset=y_offset)
        comp._plot_tracer(fig, shared_nodes, row=1, col=3, y_offset=y_offset)

        # Draw lines between each shared leaf
        for node_name in shared_nodes:
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[self.coords[node_name]['y'], comp.coords[node_name]['y'] + y_offset],
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

        fig.update_layout(
            template=SIMPLE_TEMPLATE,
            yaxis=blank_axis,
            yaxis2=blank_axis,
            yaxis3=blank_axis,
            xaxis=dict(
                automargin=True,
                title_text="SNP rate"
            ),
            xaxis2=blank_axis,
            xaxis3=dict(
                automargin=True,
                title_text="SNP rate",
                autorange="reversed"
            ),
            # Leaves are drawn as points, not labels, so no label margin
            margin=dict(l=50, r=30, b=70, t=60),
            height=height,
            width=width
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
    def dedup_refseq(self) -> None:
        """
        Drop the GenBank copy of any assembly that is also present as RefSeq.

        NCBI publishes the same assembly under both a GCA_ (GenBank) and a
        GCF_ (RefSeq) accession; a pangenome built from both ends up with two
        identical leaves, which clutters a tanglegram.
        """
        leaves = {leaf.name: leaf for leaf in self.tree.get_terminals()}
        for name in genbank_duplicates(leaves):
            self.tree.prune(leaves[name])


def _newick_safe(name: str) -> str:
    """A leaf label with the characters Newick reserves replaced."""
    return name.replace("(", "[").replace(")", "]").replace(",", " ").replace(":", " ").replace(";", " ").replace("'", "")
