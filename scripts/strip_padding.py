"""Strip edge N-padding from a pre-padded .pt dataset, producing native-length tiles.

Input: a .pt (or .pt.gz) file where sequences are stored at full window length (e.g.
600 nt) with leading/trailing N's for short tiles, and each element's ``meta["pad_side"]``
encodes the padding intent (as set by ``convert_hfds_to_pt.py``).

Output: a .pt file where sequences are stored at their native genomic length.
``meta["pad_side"]`` is preserved so that the wrapper dataset classes
(``GzListDataset``, ``PreloadedListDataset``) can re-pad at load time when passed
``length=<window_len>``.

The sidecar ``.metadata.yaml`` records ``is_pre_padded: False`` and ``seq_len`` set
to the original window length (e.g. 600), which is the length the model expects —
not the variable native tile length.

Usage
-----
python scripts/strip_padding.py \\
    --input  resources/parnet-encore-eclip/600nt_windows/encode.filtered.pt \\
    --output resources/parnet-encore-eclip/600nt_windows/encode.filtered.stripped.pt

# Optionally give the metadata sidecar an explicit base name:
python scripts/strip_padding.py \\
    --input  encode.filtered.pt \\
    --output encode.filtered.stripped.pt \\
    --metadata-basename encode.filtered.stripped
"""

import argparse
import gzip
import io
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from parnet_demo_utils import strip_element_padding


def _load_pt(input_path: Path) -> dict:
    """Load a .pt or .pt.gz file, preferring the mmap-friendly .pt companion."""
    pt_companion = Path(str(input_path).replace(".pt.gz", ".pt"))
    if pt_companion.exists() and pt_companion != input_path:
        print(f"Loading (mmap) {pt_companion.name} ...", flush=True)
        return torch.load(pt_companion, mmap=True, weights_only=False)
    if str(input_path).endswith(".gz"):
        print(f"Loading (gzip) {input_path.name} — may take several minutes ...", flush=True)
        with gzip.open(input_path, "rb") as f:
            return torch.load(io.BytesIO(f.read()), weights_only=False)
    print(f"Loading (mmap) {input_path.name} ...", flush=True)
    return torch.load(input_path, mmap=True, weights_only=False)


def strip(
    input_path: Path,
    output_path: Path,
    metadata_basename: str | None = None,
    total_key_arg: str | None = None,
) -> None:
    data = _load_pt(input_path)
    splits = list(data.keys())
    print(f"Splits: {splits}")

    # Infer window length from the first padded sequence before stripping.
    window_len: int | None = None
    for split in splits:
        for elem in data[split]:
            seq = elem["inputs"]["sequence"]
            if isinstance(seq, str) and elem["meta"].get("pad_side", -1) != -1:
                window_len = len(seq)
                break
        if window_len is not None:
            break
    if window_len is None:
        # All tiles fill the window (pad_side == -1) — nothing to strip.
        print("No padded tiles found (all pad_side == -1). Writing output unchanged.")
        window_len = len(data[splits[0]][0]["inputs"]["sequence"])

    print(f"Window length: {window_len} nt")

    result: dict[str, list] = {}
    split_counts: dict[str, int] = {}

    for split_name in splits:
        samples = data[split_name]
        n = len(samples)
        print(f"  Stripping '{split_name}' ({n:,} samples) ...", flush=True)
        result[split_name] = [
            strip_element_padding(s)
            for s in tqdm(samples, desc=split_name)
        ]
        split_counts[split_name] = n

    print(f"\nSaving → {output_path} ...", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wb") as f:
            torch.save(result, f)
    else:
        torch.save(result, output_path)
    print(f"Done. ({output_path.stat().st_size / 1e9:.2f} GB)")

    # Infer metadata from first sample.
    first = result[splits[0]][0]
    task_names = list(first["outputs"].keys())
    if total_key_arg is not None:
        if total_key_arg not in task_names:
            print(f"ERROR: --total-key '{total_key_arg}' not in task names {task_names}", file=sys.stderr)
            sys.exit(1)
        total_key = total_key_arg
    else:
        total_key = task_names[0]
        if len(task_names) > 1:
            print(f"WARNING: --total-key not set; using first task '{total_key}'. "
                  f"Available: {task_names}", flush=True)
    n_tracks   = list(first["outputs"][total_key]["size"])[0]

    stem = metadata_basename or output_path.stem
    if output_path.suffix == ".gz" and stem.endswith(".pt"):
        stem = stem[: -len(".pt")]
    meta_path = output_path.parent / f"{stem}.metadata.yaml"
    meta = {
        "sequence_format": "string",
        "signal_format":   "sparse_tensor",
        "seq_len":         window_len,   # model window; stored seqs are shorter (native len)
        "n_tracks":        int(n_tracks),
        "task_names":      task_names,
        "total_key":       total_key,
        "splits":          split_counts,
        "source":          str(input_path),
        "is_pre_padded":   False,
    }
    meta_path.write_text(yaml.dump(meta, default_flow_style=False))
    print(f"Metadata → {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input",  type=Path, required=True,
                        help="Source .pt or .pt.gz file with pre-padded sequences and pad_side in meta")
    parser.add_argument("--output", type=Path, required=False,
                        help="Output .pt path. "
                             "(default: next to input with .pt/.pt.gz replaced by .stripped.pt)")
    parser.add_argument("--metadata-basename", type=str, default=None,
                        help="Base name for the metadata sidecar (default: derived from output stem)")
    parser.add_argument("--total-key", type=str, default=None,
                        help="Key to record as total_key in the metadata sidecar. "
                             "Defaults to the first task key (with a warning). "
                             "Example: --total-key eCLIP")
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: input does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        stem = input_path.name
        for suffix in (".pt.gz", ".pt"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        output_path = input_path.parent / f"{stem}.stripped.pt"
    else:
        output_path = args.output.resolve()

    strip(input_path, output_path, metadata_basename=args.metadata_basename,
          total_key_arg=args.total_key)


if __name__ == "__main__":
    main()
