"""
CompareMultipleMetagenomes: combine multiple contrast analyses across pangenomes.

This model does not read from a single directory. It holds a mapping from
pangenome name to ContrastMetagenomes, for combining batches of analysis where
different pangenomes were used to compare the same groups of metagenomes.
"""

from __future__ import annotations

from typing import Iterator

from .contrast_metagenomes import ContrastMetagenomes


class CompareMultipleMetagenomes:
    """
    Combined view of multiple contrast-metagenomes analyses keyed by pangenome name.

    Each key in the input dict is the name of the pangenome used to create that
    contrast. Use this model to combine multiple batches of analysis where
    different pangenomes were used to compare the same groups of metagenomes.

    Parameters
    ----------
    contrasts : dict[str, ContrastMetagenomes]
        Mapping from pangenome name to ContrastMetagenomes instance. Each key
        denotes the pangenome used to create that contrast.
    """

    def __init__(self, contrasts: dict[str, ContrastMetagenomes]) -> None:
        self._contrasts = dict(contrasts)

    @property
    def pangenome_names(self) -> list[str]:
        """Ordered list of pangenome names (keys)."""
        return sorted(self._contrasts.keys())

    def __len__(self) -> int:
        return len(self._contrasts)

    def __contains__(self, pangenome_name: str) -> bool:
        return pangenome_name in self._contrasts

    def __getitem__(self, pangenome_name: str) -> ContrastMetagenomes:
        return self._contrasts[pangenome_name]

    def __iter__(self) -> Iterator[tuple[str, ContrastMetagenomes]]:
        return iter(self._contrasts.items())

    def get(self, pangenome_name: str, default: ContrastMetagenomes | None = None) -> ContrastMetagenomes | None:
        """Return the ContrastMetagenomes for the given pangenome name, or default."""
        return self._contrasts.get(pangenome_name, default)
