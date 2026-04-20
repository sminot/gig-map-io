from Bio.Phylo.BaseTree import Tree, Clade
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List


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

        # If it's an internal node
        else:
            # See how much space we have to work with
            range = stop - start

            # Keep a pointer to the y coordinate which will increase
            # with each child
            previous_y = start

            child_ys = []

            # Iterate over each child (there may be more than 1)
            for child in clade.clades:
                # Calculate the new y based on its relative size
                new_y = previous_y + (range * len(child.get_terminals()) / len(clade.get_terminals()))

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

    def plot(self):
        # Set up a figure
        fig = make_subplots(rows=1, cols=1)

        self.plot_lines(fig)
        self.plot_points(fig, mode="markers+text")
        fig.update_layout(
            template="simple_white",
            yaxis=dict(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            xaxis=dict(
                automargin=True,
                title_text="SNP Rate"
            ),
            margin=dict(l=100, r=400, b=100, t=100),
            title_text=self.name,
        )
        return fig

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
                        line_color="black"
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
                marker_color="black",
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
                    line=dict(dash='dot', color="gray"),
                    cliponaxis=False
                ),
                row=row,
                col=col
            )

    def align_trees(self, comp: 'Phylogeny'):
        print(f"Aligning {self.name} to {comp.name}")

        made_switch = True
        for _ in range(50):
            made_switch = False
            for node in self.tree.get_nonterminals():
                # Try exchanging every pair of nodes
                for i in range(len(node.clades)-1):
                    for j in range(i, len(node.clades)):
                        score = self._score_tree_alignment(comp)
                        node.clades[i], node.clades[j] = node.clades[j], node.clades[i]
                        new_score = self._score_tree_alignment(comp)

                        if new_score > score:
                            made_switch = True
                            print(f"Kept new order - {node.name} {i} <-> {j}")
                        else:
                            node.clades[i], node.clades[j] = node.clades[j], node.clades[i]
            if not made_switch:
                break

    def _score_tree_alignment(self, comp: 'Phylogeny'):
        self_leaf_order = self._leaf_order()
        comp_leaf_order = comp._leaf_order()
        shared = list(set(self_leaf_order.keys()) & set(comp_leaf_order.keys()))
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

    def compare(self, comp: 'Phylogeny', height: int, width: int, scale_by: str, align_tree_a: bool):
        print(f"Calculating concordance: {self.name} vs. {comp.name}")
        concordance = self._calc_concordance(comp)

        # If there are fewer than 3 leafs, this cannot take place
        if concordance is None:
            raise ValueError("Not enough shared genomes to compare.")

        # Align the two trees against each other
        if align_tree_a:
            self.align_trees(comp)
        comp.align_trees(self)

        if align_tree_a:
            self.align_trees(comp)
            comp.align_trees(self)
            self.align_trees(comp)
            comp.align_trees(self)

        # Regenerate the coordinates
        print(f"Regenerating coordinates")
        self.find_coords()
        comp.find_coords()

        # Get the list of nodes which are found in common
        shared_nodes = list(set(self._get_leafs(self.tree)) & set(comp._get_leafs(comp.tree)))

        # If the user wants to scale the total trees to be the same, just adjust the comp coordinates
        if scale_by == "Total Span" and len(shared_nodes) > 1:

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
                    line=dict(dash='dot', color="gray"),
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
            template="simple_white",
            yaxis=blank_axis,
            yaxis2=blank_axis,
            yaxis3=blank_axis,
            xaxis=dict(
                automargin=True,
                title_text="SNP Rate"
            ),
            xaxis2=blank_axis,
            xaxis3=dict(
                automargin=True,
                title_text="SNP Rate",
                autorange="reversed"
            ),
            margin=dict(l=100, r=400, b=100, t=100),
            height=height,
            width=width
        )

        return fig


    def _calc_concordance(self, comp: 'Phylogeny'):
        """
        Concordance: Spearman correlation of distances for all shared nodes.
        Nodes are shared if both trees contain a node with the same set of leafs.
        """
        # Get the shared set of leafs for both trees
        shared_leafs = list(set(self._get_leafs(self.tree)) & set(self._get_leafs(comp.tree)))
        # If there are fewer than 3 shared leafs, return null
        if len(shared_leafs) < 3:
            return

        # Get the vector of pairwise distances for this bin
        dists1 = [
            self.distances[name1][name2]
            for name1 in shared_leafs
            for name2 in shared_leafs
            if name1 < name2
        ]
        # And the comparitor
        dists2 = [
            comp.distances[name1][name2]
            for name1 in shared_leafs
            for name2 in shared_leafs
            if name1 < name2
        ]

        # Calculate the spearman correlation
        r = stats.spearmanr(dists1, dists2)
        return r.statistic

    def _get_leafs(self, node: Tree):
        return [leaf.name for leaf in node.get_terminals()]

    def _get_node_terminals(self, tree: Tree, shared_leafs: set):
        nodes = [
            set(self._get_leafs(node))
            for node in tree.get_nonterminals()
            if len(node.get_terminals()) > 1
        ]

        # Only keep the shared leafs (genomes)
        nodes = [
            frozenset(node & shared_leafs)
            for node in nodes
            if len(node & shared_leafs) > 1
        ]

        return set(nodes)

    def newick(self) -> str:
        """
        Return the Newick string for the tree.
        """
        return self.tree.format("newick")