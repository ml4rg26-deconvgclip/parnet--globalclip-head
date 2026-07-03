"""Sparse/dense tensor utilities for PARNET eCLIP datasets.

Handles the two storage formats used across dataset types:

- **sparse_tensor** format (``.pt`` / ``.pt.gz`` files): sparse dicts with
  ``torch.Tensor`` indices and values.
- **sparse_list** format (HuggingFace Arrow HFDS): sparse dicts with Python
  ``list`` indices and values (required for Arrow serialisation).

TypedDict type aliases are also defined here for use in type annotations
throughout the rest of the library and in notebooks.
"""

from __future__ import annotations

from typing import TypedDict

import torch

from .bed_utils import parse_tile_name


# ── Type aliases ──────────────────────────────────────────────────────────────


class SparseTensorDict(TypedDict):
    """Sparse COO tensor stored as a plain dict (used in .pt/.pt.gz files).

    Shape convention: ``size = (N_TRACKS, SEQ_LEN)``.
    """

    indices: torch.Tensor  # shape (2, nnz) — [track_index, position]
    values: torch.Tensor   # shape (nnz,)   — read counts at each (track, pos)
    size: torch.Size       # (N_TRACKS, SEQ_LEN)


class ParnetDataElementInputs(TypedDict):
    """Inputs dict for one eCLIP tile."""

    sequence: str  # DNA string, length = SEQ_LEN


ParnetDataElementOutputs = dict[str, SparseTensorDict]
"""Outputs dict for one eCLIP tile: arbitrary task names mapped to sparse tensors.

Keys are user-defined (e.g. ``"eCLIP"``, ``"control"``, ``"total"``).  No keys are
required — a dataset may omit ``"control"`` entirely.  The DataLoader layer
(``GzListDataset``, ``HFDSDataset``, ``PreloadedListDataset``) remaps the file-level
task key to ``"total"`` via its ``total_key=`` parameter before passing the batch to
the model.
"""


class ParnetDataElement(TypedDict):
    """One sample from a PARNET .pt/.pt.gz dataset."""

    inputs: ParnetDataElementInputs
    outputs: ParnetDataElementOutputs
    meta: dict  # at minimum: {"name": "chrN:start-end:strand"}


# ── Sparse ↔ dense conversion ─────────────────────────────────────────────────


def torch_sparse_to_dense(sparse: SparseTensorDict) -> torch.Tensor:
    """Reconstruct a dense tensor from a sparse COO dict (tensor format).

    Args:
        sparse: Dict with ``indices`` (Tensor), ``values`` (Tensor), and
            ``size`` (torch.Size or list).

    Returns:
        Dense float tensor of shape ``sparse["size"]``.

    Example:
        >>> signal = {
        ...     "indices": torch.tensor([[0, 0], [1, 5]]),
        ...     "values":  torch.tensor([3, 1]),
        ...     "size":    torch.Size([9, 600]),
        ... }
        >>> dense = torch_sparse_to_dense(signal)  # shape (9, 600)
    """
    return torch.sparse_coo_tensor(
        sparse["indices"], sparse["values"], sparse["size"]
    ).to_dense()


def torch_dense_to_sparse(dense: torch.Tensor) -> SparseTensorDict:
    """Convert a dense tensor to a sparse COO dict (tensor format).

    Only non-zero entries are stored.

    Args:
        dense: Dense tensor of any shape — typically ``(N_TRACKS, SEQ_LEN)``.

    Returns:
        SparseTensorDict with ``indices``, ``values``, and ``size``.

    Example:
        >>> t = torch.zeros(9, 600)
        >>> t[0, 42] = 5
        >>> sp = torch_dense_to_sparse(t)
        >>> sp["values"]
        tensor([5.])
    """
    indices = dense.nonzero(as_tuple=False).T  # (2, nnz)
    values = dense[indices[0], indices[1]]
    return SparseTensorDict(indices=indices, values=values, size=dense.shape)


def dense_to_sparse_lists(tensor: torch.Tensor) -> dict:
    """Convert a dense tensor to a sparse COO dict with Python list values.

    The list format is required for HuggingFace Arrow serialisation (HFDS).

    Args:
        tensor: Dense tensor of any shape.

    Returns:
        Dict with ``indices`` (list), ``values`` (list), and ``size`` (list).

    Example:
        >>> t = torch.zeros(9, 600)
        >>> t[2, 100] = 7
        >>> d = dense_to_sparse_lists(t)
        >>> d["values"]
        [7]
    """
    sp = tensor.to_sparse().coalesce()
    return {
        "indices": sp.indices().tolist(),
        "values":  sp.values().tolist(),
        "size":    list(tensor.shape),
    }


# ── Format-specific helpers ───────────────────────────────────────────────────

_BASES = "ACGT"


def sparse_onehot_to_seq(seq_dict: dict, *, n_threshold: float = 0.0) -> str:
    """Convert an old HFDS sparse one-hot sequence dict to a DNA string.

    The old ``parnet_data_v1_full`` HFDS stored sequences as sparse COO dicts.
    Shape is auto-detected from ``size``:

    - ``[SEQ_LEN, 4]`` (channels-last, PARNET HFDS default) → ``argmax(dim=1)``
    - ``[4, SEQ_LEN]`` (channels-first) → ``argmax(dim=0)``

    Both Python-list and ``torch.Tensor`` indices/values are accepted.

    A position is emitted as ``'N'`` when its maximum one-hot value is ≤
    ``n_threshold`` (default ``0.0`` = all-zeros only, matching the convention of
    ``parnet.utils.sequence_to_onehot``). Set ``n_threshold=0.3`` to handle
    uniform-0.25 ambiguity codes.

    Args:
        seq_dict: Sparse COO dict with ``indices``, ``values``, ``size``.
        n_threshold: Positions with ``max_value <= n_threshold`` are decoded as
            ``'N'``. Default ``0.0`` matches the PARNET all-zeros convention.

    Returns:
        DNA string of length ``SEQ_LEN``.

    Example:
        >>> # 3-nt sequence "ACN": third position all-zero → 'N'
        >>> seq = {
        ...     "indices": [[0, 1], [0, 1]],
        ...     "values":  [1, 1],
        ...     "size":    [3, 4],
        ... }
        >>> sparse_onehot_to_seq(seq)
        'ACN'
    """
    size = seq_dict["size"]
    # channels-last: size = [L, 4] with L != 4; channels-first: size = [4, L]
    channels_last = len(size) == 2 and size[1] == 4 and size[0] != 4
    dense = torch.sparse_coo_tensor(
        torch.tensor(seq_dict["indices"]),
        torch.tensor(seq_dict["values"]),
        size,
    ).to_dense()
    reduce_dim = 1 if channels_last else 0
    _max_vals = dense.max(dim=reduce_dim).values
    _idxs = dense.argmax(dim=reduce_dim)
    return "".join(
        _BASES[int(i)] if v > n_threshold else "N"
        for i, v in zip(_idxs.tolist(), _max_vals.tolist())
    )


def dense_onehot_to_seq(tensor: torch.Tensor, *, n_threshold: float = 0.0) -> str:
    """Convert a dense one-hot tensor to a DNA string.

    Shape is auto-detected:

    - ``(L, 4)`` channels-last → ``argmax(dim=1)``
    - ``(4, L)`` channels-first → ``argmax(dim=0)``

    A position is emitted as ``'N'`` when its maximum value is ≤ ``n_threshold``
    (default ``0.0`` = all-zeros, matching ``parnet.utils.sequence_to_onehot``).

    Args:
        tensor: Dense one-hot tensor of shape ``(L, 4)`` or ``(4, L)``.
        n_threshold: Positions with ``max_value <= n_threshold`` are decoded as
            ``'N'``. Default ``0.0`` matches the PARNET all-zeros convention.

    Returns:
        DNA string of length ``L``.

    Example:
        >>> import torch
        >>> t = torch.zeros(3, 4); t[0, 0] = 1; t[1, 1] = 1  # "ACN"
        >>> dense_onehot_to_seq(t)
        'ACN'
        >>> dense_onehot_to_seq(t.T)  # channels-first auto-detected
        'ACN'
    """
    t = tensor.float()
    channels_last = t.shape[-1] == 4 and t.shape[0] != 4
    reduce_dim = 1 if channels_last else 0
    _max_vals = t.max(dim=reduce_dim).values
    _idxs = t.argmax(dim=reduce_dim)
    return "".join(
        _BASES[int(i)] if v > n_threshold else "N"
        for i, v in zip(_idxs.tolist(), _max_vals.tolist())
    )


def classify_seq_padding(seq: str, *, strand: str | None = None) -> dict:
    """Classify the N-padding layout of a pre-padded DNA string.

    Counts leading and trailing ``'N'`` characters and maps to the parnet
    ``pad_side`` convention (0=both, 1=left, 2=right, -1=no edge padding).

    When ``strand="-"`` and only one side is padded, the physical layout is
    un-flipped to recover the original ``pad_side`` intent, matching the
    behaviour of ``parnet.data.datasets.ListDataset.__getitem__`` which swaps
    ``pad_side`` 1 ↔ 2 for reverse-strand tiles before inserting N's.

    Args:
        seq: DNA string (may contain ``'N'`` at edges and/or internally).
        strand: ``"+"`` or ``"-"`` from the tile name. If ``None``, the
            physical layout is returned without strand correction.

    Returns:
        Dict with keys:
        - ``left_pad_size``:  number of leading N's.
        - ``right_pad_size``: number of trailing N's.
        - ``pad_side``:       parnet convention integer (-1/0/1/2).
        - ``has_internal_n``: True if any non-edge position is ``'N'``.

    Example:
        >>> classify_seq_padding("NNNACGTNNN")
        {'left_pad_size': 3, 'right_pad_size': 3, 'pad_side': 0, 'has_internal_n': False}
        >>> classify_seq_padding("NNNACGT", strand="-")  # flip: physical left → intent right
        {'left_pad_size': 3, 'right_pad_size': 0, 'pad_side': 2, 'has_internal_n': False}
    """
    left  = len(seq) - len(seq.lstrip("N"))
    right = len(seq) - len(seq.rstrip("N"))
    inner = seq[left : len(seq) - right] if right else seq[left:]
    if left > 0 and right > 0:
        pad_side = 0
    elif left > 0:
        pad_side = 1
    elif right > 0:
        pad_side = 2
    else:
        pad_side = -1
    if strand == "-" and pad_side in (1, 2):
        pad_side = 3 - pad_side  # 1 ↔ 2
    return {
        "left_pad_size":  left,
        "right_pad_size": right,
        "pad_side":       pad_side,
        "has_internal_n": "N" in inner,
    }


def infer_pad_sizes(name: str, pad_side: int, window_len: int) -> tuple[int, int]:
    """Compute ``(left_pad, right_pad)`` from tile coordinates and ``pad_side``.

    Mirrors the logic in ``parnet.data.datasets.ListDataset.__getitem__``
    (lines 114–130), including the reverse-strand flip for ``pad_side`` ∈ {1, 2}.

    Use this to compute padding sizes without scanning the sequence — useful for
    2000nt-style datasets where sequences are stored at native length (not pre-padded)
    and also for cross-validating physical N counts in pre-padded 600nt sequences.

    Args:
        name:       Tile name ``"chrN:start-end:strand"``.
        pad_side:   Parnet convention: 0=both, 1=left, 2=right.
        window_len: Target sequence length (e.g. 600 or 2000).

    Returns:
        ``(left_pad, right_pad)`` — N's to prepend / append. Both 0 if tile fills
        the window.

    Example:
        >>> infer_pad_sizes("chrX:100-550:+", pad_side=0, window_len=600)
        (25, 25)
        >>> infer_pad_sizes("chrX:100-550:-", pad_side=1, window_len=600)
        (0, 500)
    """
    _, start, end, strand = parse_tile_name(name)
    total = window_len - (end - start)
    if total <= 0:
        return 0, 0
    if pad_side == 1:
        left, right = total, 0
    elif pad_side == 2:
        left, right = 0, total
    else:  # 0 = center
        right = total // 2
        left = total - right
    if strand == "-" and pad_side in (1, 2):
        left, right = right, left
    return left, right


def strip_element_padding(elem: dict) -> dict:
    """Strip parnet-inserted N-padding from a pre-padded element, returning native-length format.

    Converts a pre-padded element (sequence stored at full window length with leading /
    trailing N's added by parnet) into native-length format: the sequence shrinks to its
    genomic content and sparse signal indices are shifted accordingly.
    ``meta["pad_side"]`` is preserved so the wrapper dataset classes can re-pad at load
    time when passed ``length=<window_len>``.

    **Important**: padding sizes are derived from ``meta["name"]`` coordinates via
    :func:`infer_pad_sizes`, NOT by counting physical edge N's.  This is necessary
    because some reference genome regions start or end with many N's (assembly gaps,
    pericentromeric sequence) which are indistinguishable from parnet-inserted padding by
    physical inspection.  Coordinate-based sizes strip *only* the N's that parnet
    inserted, leaving reference genome N's intact.

    Elements with ``pad_side == -1`` (no parnet-inserted padding — tile fills the window) are
    returned unchanged.

    Operates on ``.pt`` tensor-format sparse dicts (``indices`` as ``torch.Tensor``).

    Args:
        elem: Sample dict with keys ``inputs``, ``outputs``, ``meta``.  Must have
            ``meta["pad_side"]`` and ``meta["name"]`` set (as written by the conversion
            scripts).

    Returns:
        New sample dict at native sequence length, or the original dict unchanged when
        ``pad_side == -1``.

    Example:
        >>> import torch
        >>> elem = {
        ...     "meta": {"name": "chrX:0-550:+", "pad_side": 0},
        ...     "inputs": {"sequence": "N" * 25 + "A" * 550 + "N" * 25},
        ...     "outputs": {"total": {
        ...         "indices": torch.tensor([[0, 0], [30, 580]]),
        ...         "values":  torch.tensor([3., 1.]),
        ...         "size":    [9, 600],
        ...     }},
        ... }
        >>> stripped = strip_element_padding(elem)
        >>> len(stripped["inputs"]["sequence"])
        550
        >>> stripped["outputs"]["total"]["indices"][1].tolist()
        [5, 555]
    """
    meta     = elem["meta"]
    pad_side = meta.get("pad_side")
    if pad_side == -1 or pad_side is None:
        return elem  # no parnet-inserted padding (tile fills window, or unknown)

    seq        = elem["inputs"]["sequence"]
    window_len = len(seq)
    name       = meta.get("name", "")
    if isinstance(name, bytes):
        name = name.decode("utf-8")

    # Coordinate-based parnet-inserted padding sizes — exact, unaffected by reference N's.
    left, right = infer_pad_sizes(name, pad_side, window_len)
    if left == 0 and right == 0:
        return elem  # pad_side was set but total==0 (shouldn't happen after bug fix)

    native_seq = seq[left : window_len - right] if right else seq[left:]
    native_len = len(native_seq)

    new_outputs: dict = {}
    for task, sp in elem["outputs"].items():
        idx  = sp["indices"]           # shape (2, nnz) Tensor
        vals = sp["values"]
        pos  = idx[1]
        keep = (pos >= left) & (pos < window_len - right) if right else (pos >= left)
        new_outputs[task] = {
            "indices": torch.stack([idx[0][keep], pos[keep] - left]),
            "values":  vals[keep],
            "size":    [list(sp["size"])[0], native_len],
        }

    return {
        "meta":    dict(meta),
        "inputs":  {"sequence": native_seq},
        "outputs": new_outputs,
    }


def get_padding_mask(name: str, pad_side: int, window_len: int) -> torch.Tensor:
    """Return a 1-D bool mask with ``True`` at edge-padded positions.

    Uses coordinate-based padding sizes (via :func:`infer_pad_sizes`), so only
    the true edge-padding positions are marked — internal ``'N'`` characters
    (assembly gaps, repeat-masked regions) are correctly left as ``False``.

    This is more accurate than the one-hot sum trick
    ``(1 - torch.sum(onehot, dim=0)) * 4`` used in
    ``parnet.data.datasets.HFDSDataset``, which mis-labels all zero rows (including
    internal N's) as padding.

    Args:
        name:       Tile name ``"chrN:start-end:strand"``.
        pad_side:   Parnet convention: 0=both, 1=left, 2=right.
        window_len: Total sequence length after padding.

    Returns:
        Bool tensor of shape ``(window_len,)`` — ``True`` where padding was added.

    Example:
        >>> mask = get_padding_mask("chrX:100-550:+", pad_side=0, window_len=600)
        >>> mask[:3].tolist(), mask[-3:].tolist()
        ([True, True, True], [True, True, True])
    """
    left, right = infer_pad_sizes(name, pad_side, window_len)
    mask = torch.zeros(window_len, dtype=torch.bool)
    if left  > 0: mask[:left]   = True
    if right > 0: mask[-right:] = True
    return mask
