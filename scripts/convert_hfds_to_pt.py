"""Convert an HFDS dataset to a .pt file compatible with GzListDataset.

Sequences are stored as DNA strings (compact, lossless).
Signals are stored as sparse dicts with torch.Tensor values,
matching the format expected by GzListDataset and torch_sparse_to_dense().

Split names are normalised: "validation" → "valid" to match the convention
expected by GzListDataset / HFDSDataset and the training notebooks.

A metadata sidecar (<basename>.metadata.yaml) is written alongside the output
with dataset statistics for quick identity checks.

Usage
-----
# Recommended: save uncompressed, then compress in parallel with pigz
python scripts/convert_hfds_to_pt.py \\
    --input          resources/parnet-encore-eclip/600nt_windows/encode.filtered.hfds \\
    --output         resources/parnet-encore-eclip/600nt_windows/encode.filtered.pt \\
    --total-key      eCLIP \\
    --is-pre-padded  true \\
    --seq-len        600
pigz -p 16 resources/parnet-encore-eclip/600nt_windows/encode.filtered.pt

# Alternative: write compressed directly (slow — single-threaded gzip)
python scripts/convert_hfds_to_pt.py \\
    --input          resources/parnet-encore-eclip/600nt_windows/encode.filtered.hfds \\
    --output         resources/parnet-encore-eclip/600nt_windows/encode.filtered.pt.gz \\
    --total-key      eCLIP \\
    --is-pre-padded  true \\
    --seq-len        600

# Rename "eCLIP" → "total" and pin the total_key in one pass.
# --total-key must name the POST-rename key ("total", not "eCLIP").
python scripts/convert_hfds_to_pt.py \\
    --input          encode.filtered.hfds \\
    --output         encode.filtered.pt \\
    --rename-track-names '{"eCLIP": "total"}' \\
    --total-key      total \\
    --is-pre-padded  true \\
    --seq-len        600

By default, if --output is omitted, the output is written next to the input HFDS
directory with the `.hfds` suffix replaced by `.pt`.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from tqdm import tqdm

from parnet_demo_utils import classify_seq_padding, infer_pad_sizes, parse_tile_name, sparse_onehot_to_seq

# Map HFDS split names to the canonical .pt convention used by GzListDataset / HFDSDataset.
_SPLIT_RENAMES: dict[str, str] = {"validation": "valid"}


def _restore_sparse_tensors(d: dict) -> dict:
    """Arrow list-valued sparse dict → torch.Tensor sparse dict."""
    return {
        "indices": torch.tensor(d["indices"]),
        "values":  torch.tensor(d["values"]),
        "size":    d["size"] if isinstance(d["size"], list) else list(d["size"]),
    }


def _convert_sample(elem: dict, rename: dict[str, str]) -> dict:
    """Convert one HFDS row to pt.gz-compatible dict.

    Sequence: sparse one-hot dict or plain string → DNA string (kept as string).
    Signals:  list-valued sparse dicts → tensor sparse dicts.
    Task keys are renamed according to ``rename`` (may be empty).
    """
    seq_raw = elem["inputs"]["sequence"]
    if isinstance(seq_raw, dict):
        seq = sparse_onehot_to_seq(seq_raw)
    elif isinstance(seq_raw, bytes):
        seq = seq_raw.decode("utf-8")
    else:
        seq = seq_raw  # already a DNA string

    name = elem["meta"]["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")

    outputs = {
        rename.get(k, k): _restore_sparse_tensors(v)
        for k, v in elem["outputs"].items()
    }

    meta_out: dict = {"name": name}
    if "pad_side" not in elem["meta"]:
        # Source has no pad_side (pre-padded HFDS): derive from coordinates + physical Ns.
        _, _start, _end, _strand = parse_tile_name(name)
        _total = len(seq) - (_end - _start)
        if _total < 0:
            raise ValueError(
                f"Tile {name}: coordinate span ({_end - _start}) exceeds sequence length "
                f"({len(seq)}) — malformed HFDS entry."
            )
        if _total == 0:
            # Tile fills the full window — no geometric padding was inserted.
            # Any edge Ns are reference genome Ns, not parnet padding.
            meta_out["pad_side"] = -1
        else:
            _pad = classify_seq_padding(seq, strand=_strand)
            meta_out["pad_side"] = _pad["pad_side"]
            if _pad["pad_side"] >= 0:
                _left_inf, _right_inf = infer_pad_sizes(name, _pad["pad_side"], len(seq))
                assert _left_inf == _pad["left_pad_size"] and _right_inf == _pad["right_pad_size"], (
                    f"Padding mismatch for {name}: "
                    f"physical=({_pad['left_pad_size']},{_pad['right_pad_size']}) "
                    f"vs inferred=({_left_inf},{_right_inf})"
                )
    else:
        meta_out["pad_side"] = elem["meta"]["pad_side"]

    return {
        "meta":    meta_out,
        "inputs":  {"sequence": seq},
        "outputs": outputs,
    }


def convert(
    input_path: Path,
    output_path: Path,
    rename: dict[str, str] | None = None,
    metadata_basename: str | None = None,
    total_key_arg: str = "",
    is_pre_padded_arg: str = "false",
    seq_len_arg: int = 600,
) -> None:
    rename = rename or {}
    print(f"Loading HFDS from {input_path} ...", flush=True)
    hfds = load_from_disk(str(input_path))
    src_splits = list(hfds.keys())
    print(f"Splits: {src_splits}")

    result: dict[str, list] = {}
    split_counts: dict[str, int] = {}

    for src_name in src_splits:
        out_name = _SPLIT_RENAMES.get(src_name, src_name)
        if out_name != src_name:
            print(f"  Renaming split '{src_name}' → '{out_name}'")
        split_ds = hfds[src_name]
        n = len(split_ds)
        print(f"  Converting '{out_name}' ({n:,} samples) ...", flush=True)
        result[out_name] = [
            _convert_sample(split_ds[i], rename)
            for i in tqdm(range(n), desc=out_name)
        ]
        split_counts[out_name] = n

    # Validate metadata args against first sample (after renaming).
    first = result[list(result.keys())[0]][0]
    task_names = list(first["outputs"].keys())
    if total_key_arg not in task_names:
        print(f"ERROR: --total-key '{total_key_arg}' not in task names {task_names}", file=sys.stderr)
        sys.exit(1)
    total_key = total_key_arg
    n_tracks = first["outputs"][total_key]["size"][0]
    seq_len = seq_len_arg
    is_pre_padded = (is_pre_padded_arg == "true")

    print(f"\nSaving → {output_path} ...", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wb") as f:
            torch.save(result, f)
    else:
        torch.save(result, output_path)
    print(f"Done. ({output_path.stat().st_size / 1e9:.2f} GB)")

    # Write sidecar metadata alongside the .pt file
    stem = metadata_basename or output_path.stem
    # strip double suffix for .pt.gz
    if output_path.suffix == ".gz":
        stem = Path(stem).stem if stem.endswith(".pt") else stem
    meta_path = output_path.parent / f"{stem}.metadata.yaml"
    meta = {
        "sequence_format": "string",
        "signal_format":   "sparse_tensor",
        "seq_len":         seq_len,
        "n_tracks":        n_tracks,
        "task_names":      task_names,
        "total_key":       total_key,
        "splits":          split_counts,
        "source":          str(input_path),
        "is_pre_padded":   is_pre_padded,
    }
    meta_path.write_text(yaml.dump(meta, default_flow_style=False))
    print(f"Metadata → {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  type=Path, required=True,  help="Path to source HFDS directory")
    parser.add_argument("--output", type=Path, required=False,
                        help="Output path. Use .pt (default) for uncompressed + pigz, "
                             "or .pt.gz to write gzip directly (slower). "
                             "(default: next to input with .hfds replaced by .pt)")
    parser.add_argument("--rename-track-names", type=str, default=None,
                        metavar="JSON",
                        help="Optional JSON dict to rename output task keys, e.g. "
                             "'{\"eCLIP\": \"total\"}'. Applied before writing. "
                             "Useful to align file keys with total_key= in GzListDataset.")
    parser.add_argument("--metadata-basename", type=str, default=None,
                        help="Base name for the metadata sidecar file written alongside "
                             "the .pt output (default: derived from output file stem). "
                             "Use this to avoid collisions when multiple datasets share "
                             "the same output directory.")
    parser.add_argument("--total-key", type=str, required=True,
                        help="Key to record as total_key in the metadata sidecar. "
                             "Must name a key in the OUTPUT after any --rename-track-names "
                             "is applied (rename runs first). "
                             "Example without rename: --total-key eCLIP. "
                             "Example with rename '{\"eCLIP\":\"total\"}': --total-key total.")
    parser.add_argument("--is-pre-padded", choices=["true", "false"], required=True,
                        help="Whether the SOURCE sequences are stored at fixed window length "
                             "with N-padding ('true') or at native genomic length / stripped "
                             "('false'). Written as is_pre_padded in the metadata sidecar. "
                             "Example: --is-pre-padded false")
    parser.add_argument("--seq-len", type=int, required=True,
                        help="Window length (nt) to record as seq_len in the metadata sidecar. "
                             "For pre-padded sources this equals the stored sequence length. "
                             "For stripped sources the stored sequences are shorter; seq_len "
                             "records the model window, not the stored tile length. "
                             "Example: --seq-len 600")
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    rename: dict[str, str] = {}
    if args.rename_track_names:
        try:
            rename = json.loads(args.rename_track_names)
        except json.JSONDecodeError as e:
            print(f"ERROR: --rename-track-names is not valid JSON: {e}", file=sys.stderr)
            sys.exit(1)

    if args.output is None:
        stem = input_path.name
        if stem.endswith(".hfds"):
            stem = stem[: -len(".hfds")]
        output_path = input_path.parent / f"{stem}.pt"
    else:
        output_path = args.output.resolve()

    convert(input_path, output_path,
            rename=rename,
            metadata_basename=args.metadata_basename,
            total_key_arg=args.total_key,
            is_pre_padded_arg=args.is_pre_padded,
            seq_len_arg=args.seq_len)


if __name__ == "__main__":
    main()
