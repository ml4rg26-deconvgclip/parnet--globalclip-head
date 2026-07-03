"""BED file utilities for PARNET eCLIP tile datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenomicInterval:
    """Genomic interval parsed from a PARNET tile name (``chrN:start-end:strand``).

    Coordinates are 0-based half-open (BED convention), matching the tile name format
    used throughout PARNET datasets.

    Example:
        >>> gi = GenomicInterval.from_name("chr3:618239-620239:+")
        >>> gi.chrom, gi.start, gi.end, gi.strand
        ('chr3', 618239, 620239, '+')
        >>> gi.length
        2000
    """

    chrom: str
    start: int
    end: int
    strand: str
    name: str | None = None

    @classmethod
    def from_name(cls, name: str) -> "GenomicInterval":
        """Parse a tile name string into a GenomicInterval."""
        chrom, start, end, strand = parse_tile_name(name)
        return cls(chrom=chrom, start=start, end=end, strand=strand, name=name)

    @property
    def length(self) -> int:
        """Tile length in nucleotides (end - start)."""
        return self.end - self.start


def parse_tile_name(name: str) -> tuple[str, int, int, str]:
    """Parse a tile name string into its genomic coordinate components.

    The tile name format used throughout the PARNET eCLIP datasets is
    ``"chrN:start-end:strand"`` (e.g. ``"chr3:618239-620239:+"``).

    Args:
        name: Tile name string in ``"chrN:start-end:strand"`` format.

    Returns:
        Tuple of ``(chrom, start, end, strand)`` where ``start`` and ``end``
        are integers and ``strand`` is ``"+"`` or ``"-"``.

    Raises:
        ValueError: If ``name`` does not contain exactly two colons or the
            coordinate part cannot be split on ``"-"``.

    Example:
        >>> parse_tile_name("chr3:618239-620239:+")
        ('chr3', 618239, 620239, '+')
        >>> parse_tile_name("chrX:1000-1600:-")
        ('chrX', 1000, 1600, '-')
    """
    chrom, coords, strand = name.split(":")
    start, end = map(int, coords.split("-"))
    return chrom, start, end, strand
