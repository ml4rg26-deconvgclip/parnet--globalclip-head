"""Dataset class and item-level helpers for PARNET eCLIP preparation.

Minimal usage example::

    from functools import partial
    from parnet_demo_utils import (
        FilteredMultiTaskDataset,
        filter_min_read_count,
        center_crop_item,
    )

    filters = [
        partial(filter_min_read_count,
                min_read_count=3,
                tasks=["eCLIP"],
                track_indices=my_indices),
        # any additional (ParnetDataElement) -> bool callable goes here
    ]

    ds = FilteredMultiTaskDataset(
        base_list=data["train"],
        track_indices=my_indices,
        filters=filters,
    )
    sample = ds[0]  # ParnetDataElement with only the selected tracks
"""

from __future__ import annotations

import torch
import torch.utils.data

from .filters import FilterFunction
from .sparse_utils import (
    ParnetDataElement,
    torch_dense_to_sparse,
    torch_sparse_to_dense,
)


class FilteredMultiTaskDataset(torch.utils.data.Dataset):
    """Wraps a list of ParnetDataElements, applies filters, and slices tracks.

    Passing indices are precomputed at construction time so iteration is fast
    regardless of how many samples are discarded by the filters.

    The ``filters`` list is intentionally open: pass any combination of
    built-in predicates (from :mod:`parnet_demo_utils.filters`) and your own
    callables — anything matching ``(ParnetDataElement) -> bool`` works.

    Args:
        base_list: Full list of data elements (one split).
        track_indices: Track indices to retain in the output signal tensors.
            These are indices into the *full* (pre-filter) dataset track
            dimension.
        filters: List of filter predicates.  A sample is kept only if
            **all** predicates return ``True``.

    Example:
        >>> from functools import partial
        >>> from parnet_demo_utils import (
        ...     FilteredMultiTaskDataset,
        ...     filter_min_read_count,
        ... )
        >>> filters = [
        ...     partial(filter_min_read_count,
        ...             min_read_count=3,
        ...             tasks=["eCLIP"],
        ...             track_indices=[0, 1, 2]),
        ... ]
        >>> ds = FilteredMultiTaskDataset(raw_data, [0, 1, 2], filters)
        >>> len(ds)        # number of samples that passed all filters
        >>> sample = ds[0] # ParnetDataElement with 3-track signal
    """

    def __init__(
        self,
        base_list: list[ParnetDataElement],
        track_indices: list[int],
        filters: list[FilterFunction],
    ) -> None:
        self.base_list = base_list
        self.track_indices = track_indices
        self.index_map = [
            i for i, elem in enumerate(base_list) if all(f(elem) for f in filters)
        ]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> ParnetDataElement:
        """Return sample ``idx`` with signal tensors sliced to selected tracks.

        Args:
            idx: Index into the filtered dataset (not the original list).

        Returns:
            ParnetDataElement whose output signal tensors have shape
            ``(len(track_indices), SEQ_LEN)``.
        """
        elem = self.base_list[self.index_map[idx]]
        outputs = {
            task: torch_dense_to_sparse(
                torch_sparse_to_dense(sparse)[self.track_indices, :]
            )
            for task, sparse in elem["outputs"].items()
        }
        return ParnetDataElement(
            inputs=elem["inputs"],
            outputs=outputs,
            meta=elem["meta"],
        )


def center_crop_item(
    item: ParnetDataElement,
    crop_start: int,
    crop_end: int,
) -> ParnetDataElement:
    """Crop a data element's sequence string and signal tensors.

    Extracts ``sequence[crop_start:crop_end]`` and ``signal[:, crop_start:crop_end]``
    for every output task.  Useful when the source dataset has longer tiles
    (e.g. 2000 nt) and you need to extract a shorter central window (e.g. 600 nt).

    Args:
        item: A PARNET data element.
        crop_start: Start position of the crop window (inclusive).
        crop_end: End position of the crop window (exclusive).

    Returns:
        A new ParnetDataElement with sequence and signals cropped to
        ``[crop_start, crop_end)``.

    Example:
        >>> cropped = center_crop_item(element, crop_start=700, crop_end=1300)
        >>> len(cropped["inputs"]["sequence"])
        600
    """
    outputs_cropped = {
        task: torch_dense_to_sparse(
            torch_sparse_to_dense(sparse)[:, crop_start:crop_end]
        )
        for task, sparse in item["outputs"].items()
    }
    return ParnetDataElement(
        inputs={"sequence": item["inputs"]["sequence"][crop_start:crop_end]},
        outputs=outputs_cropped,
        meta=item["meta"],
    )
