"""Helpers for filtering and converting raw HuggingFace Arrow (HFDS) samples.

These functions operate on the raw dicts returned by ``hfds[split][i]`` —
i.e. before any conversion to ``ParnetDataElement``. They mirror the logic in
:mod:`parnet_demo_utils.filters` but work with Arrow-native sparse list dicts
(where ``indices`` and ``values`` are Python lists, not ``torch.Tensor``).

Typical usage in a prepare notebook::

    from parnet_demo_utils import hfds_sample_passes_filter, convert_hfds_sample

    for elem in hfds["train"]:
        seq_raw = elem["inputs"]["sequence"]
        # Handle old sparse one-hot format:
        if isinstance(seq_raw, dict):
            from parnet_demo_utils import sparse_onehot_to_seq
            seq_str = sparse_onehot_to_seq(seq_raw)
        else:
            seq_str = seq_raw.decode("utf-8") if isinstance(seq_raw, bytes) else seq_raw

        if not hfds_sample_passes_filter(
            elem,
            track_indices=my_indices,
            min_rc=3,
            output_keys=["eCLIP", "control"],
        ):
            continue

        converted = convert_hfds_sample(elem, track_indices=my_indices, seq_str=seq_str)
        # converted is an Arrow-serialisable dict ready for DatasetDict.from_list
"""

from __future__ import annotations

import torch

from .sparse_utils import dense_to_sparse_lists


def hfds_sample_passes_filter(
    elem: dict,
    *,
    track_indices: list[int],
    min_rc: int,
    output_keys: list[str],
) -> bool:
    """Keep an HFDS sample if any selected track meets a read-count threshold.

    Equivalent to :func:`~parnet_demo_utils.filters.filter_min_read_count` but
    operates on raw HFDS Arrow dicts (sparse list format) rather than on
    ``ParnetDataElement`` sparse tensor dicts.

    Args:
        elem: Raw HFDS sample dict with an ``"outputs"`` key containing sparse
            list dicts (``{"indices": list, "values": list, "size": list}``).
        track_indices: Track indices to check within the signal tensor.
        min_rc: Minimum read count that at least one selected track must reach.
        output_keys: Output keys to scan, e.g. ``["eCLIP"]`` or
            ``["eCLIP", "control"]``.

    Returns:
        True if any ``(key, track)`` pair has ``max >= min_rc``.

    Example:
        >>> passes = hfds_sample_passes_filter(
        ...     elem,
        ...     track_indices=[4, 7],
        ...     min_rc=3,
        ...     output_keys=["eCLIP"],
        ... )
    """
    for key in output_keys:
        sp = elem["outputs"][key]
        dense = torch.sparse_coo_tensor(
            torch.tensor(sp["indices"]),
            torch.tensor(sp["values"]),
            sp["size"],
        ).to_dense()  # (N_TRACKS, SEQ_LEN)
        if dense[track_indices, :].max().item() >= min_rc:
            return True
    return False


def convert_hfds_sample(
    elem: dict,
    *,
    track_indices: list[int],
    seq_str: str,
    output_keys: list[str] | None = None,
) -> dict:
    """Convert one HFDS sample to a filtered, Arrow-serialisable dict.

    Selects the requested tracks from each output task and stores the sequence
    as a plain DNA string (new HFDS convention).  The resulting dict can be
    passed to ``datasets.Dataset.from_list`` without further conversion.

    Args:
        elem: Raw HFDS sample dict with ``"outputs"`` (sparse list dicts),
            and ``"meta"`` keys.
        track_indices: Track indices to retain in the output signal tensors.
        seq_str: Pre-decoded DNA string for the sample's input sequence
            (caller is responsible for format detection; see module docstring).
        output_keys: Task keys to process.  ``None`` (default) processes every
            key present in ``elem["outputs"]``.

    Returns:
        Arrow-serialisable dict with keys ``"inputs"``, ``"outputs"``,
        ``"meta"``.  Signal tensors are stored as sparse list dicts.

    Example:
        >>> out = convert_hfds_sample(
        ...     elem,
        ...     track_indices=[0, 3, 7],
        ...     seq_str="ACGT...",
        ... )
        >>> out["inputs"]["sequence"]
        'ACGT...'
        >>> out["outputs"]["eCLIP"]["size"]
        [3, 600]
    """
    keys = output_keys if output_keys is not None else list(elem["outputs"].keys())
    outputs: dict[str, dict] = {}
    for key in keys:
        sp = elem["outputs"][key]
        dense = torch.sparse_coo_tensor(
            torch.tensor(sp["indices"]),
            torch.tensor(sp["values"]),
            sp["size"],
        ).to_dense()                        # (N_TRACKS, SEQ_LEN)
        selected = dense[track_indices, :]  # (n_selected, SEQ_LEN)
        outputs[key] = dense_to_sparse_lists(selected)

    name = elem["meta"]["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")

    return {
        "inputs":  {"sequence": seq_str},
        "outputs": outputs,
        "meta":    {"name": name},
    }
