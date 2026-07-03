"""Filter predicates for PARNET eCLIP dataset elements.

Every filter is a plain callable ``(ParnetDataElement) -> bool``. You can
compose built-in filters with your own by adding them to the list passed to
:class:`~parnet_demo_utils.datasets.FilteredMultiTaskDataset`.

Custom filters
--------------
Write any function or lambda that accepts a :class:`ParnetDataElement` and
returns ``True`` to keep the sample::

    def my_filter(element: ParnetDataElement) -> bool:
        return element["meta"]["name"].startswith("chr1:")

    filters = [
        partial(filter_min_read_count, min_read_count=3,
                tasks=["eCLIP"], track_indices=my_indices),
        my_filter,  # ← add custom filter here
    ]
    ds = FilteredMultiTaskDataset(data["train"], my_indices, filters)

Use ``functools.partial`` to bind parameters to built-in filters::

    from functools import partial
    f = partial(filter_min_read_count, min_read_count=5,
                tasks=["eCLIP", "control"], track_indices=[0, 3, 7])
    keeps = f(element)   # True / False
"""

from __future__ import annotations

from typing import Protocol

from .sparse_utils import ParnetDataElement, torch_sparse_to_dense


class FilterFunction(Protocol):
    """Protocol for filter predicates used by FilteredMultiTaskDataset."""

    def __call__(self, element: ParnetDataElement) -> bool:
        """Return True to keep, False to discard the sample."""
        ...


def filter_minimum_length(element: ParnetDataElement, *, min_length: int) -> bool:
    """Keep samples whose input sequence meets a minimum length.

    Useful when the source dataset contains variable-length tiles and a minimum
    window is required before centre-cropping.

    Args:
        element: A PARNET data element.
        min_length: Minimum acceptable sequence length (inclusive).

    Returns:
        True if ``len(element["inputs"]["sequence"]) >= min_length``.

    Example:
        >>> from functools import partial
        >>> f = partial(filter_minimum_length, min_length=2000)
        >>> f({"inputs": {"sequence": "ACG" * 700}, "outputs": {}, "meta": {}})
        True
    """
    return len(element["inputs"]["sequence"]) >= min_length


def filter_min_read_count(
    element: ParnetDataElement,
    *,
    min_read_count: int,
    tasks: list[str],
    track_indices: list[int],
) -> bool:
    """Keep samples where any selected track reaches a read-count threshold.

    Scans the specified output tasks and track indices. Returns ``True`` as
    soon as any ``(task, track)`` combination has a maximum value ≥
    ``min_read_count`` anywhere along the sequence.

    Args:
        element: A PARNET data element.
        min_read_count: Minimum read count that at least one selected track
            must reach (``max`` over the sequence position axis).
        tasks: Output task keys to check, e.g. ``["eCLIP"]`` or
            ``["eCLIP", "control"]``.
        track_indices: Indices into the track (first) dimension of the signal
            tensor. Must be valid indices for the full (pre-filter) dataset.

    Returns:
        True if any ``element["outputs"][task][track, :]`` has
        ``max >= min_read_count``.

    Example:
        >>> from functools import partial
        >>> f = partial(filter_min_read_count,
        ...             min_read_count=3,
        ...             tasks=["eCLIP"],
        ...             track_indices=[4, 7])
        >>> f(element)   # True if track 4 or 7 has ≥3 reads in eCLIP
    """
    for task in tasks:
        dense = torch_sparse_to_dense(element["outputs"][task])
        if dense[track_indices, :].max().item() >= min_read_count:
            return True
    return False
