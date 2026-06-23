"""Convert a .pt / .pt.gz file (GzListDataset format) to a HuggingFace Arrow HFDS.

Sequences are stored as plain DNA strings (new HFDS convention).
Signals are stored as sparse dicts with Python list values
(required for Arrow serialization).

Output directory structure (written directly under --outputdir)::

    <outputdir>/
    ├── dataset_dict.json
    ├── train/
    │   ├── data-00000-of-NNNNN.arrow   # 5-digit zero-padded, fixed by HF library
    │   └── dataset_info.json
    ├── valid/                           # split name preserved from source .pt
    │   └── ...
    └── test/
        └── ...

A metadata sidecar (<basename>.metadata.yaml) is written inside the output directory.

Usage
-----
python scripts/convert_pt_to_hfds.py \\
    --input          resources/parnet-encore-eclip/600nt_windows/encode.filtered.pt \\
    --outputdir      resources/parnet-encore-eclip/600nt_windows/encode.filtered.hfds \\
    --total-key      eCLIP \\
    --is-pre-padded  true \\
    --seq-len        600

# Match the sharding of the source HFDS (train: 76, valid: 18, test: 10)
python scripts/convert_pt_to_hfds.py \\
    --input          encode.filtered.pt \\
    --outputdir      encode.filtered.hfds \\
    --num-shards     train:76,valid:18,test:10 \\
    --total-key      eCLIP \\
    --is-pre-padded  true \\
    --seq-len        600

# Rename "eCLIP" → "total" and pin the total_key in one pass.
# --total-key must name the POST-rename key ("total", not "eCLIP").
python scripts/convert_pt_to_hfds.py \\
    --input          encode.filtered.pt \\
    --outputdir      encode.filtered.hfds \\
    --rename-track-names '{"eCLIP": "total"}' \\
    --total-key      total \\
    --is-pre-padded  true \\
    --seq-len        600

By default, if --outputdir is omitted, the output is written next to the input
with the `.pt` / `.pt.gz` suffix replaced by `.no-one-hot.hfds`.
"""

import argparse
import gzip
import io
import json
import sys
from pathlib import Path

import math

import torch
import yaml
import datasets as hf_datasets
from tqdm import tqdm

from parnet_demo_utils import classify_seq_padding, dense_onehot_to_seq, infer_pad_sizes, parse_tile_name


def _sparse_dict_to_lists(d: dict) -> dict:
    """torch.Tensor sparse dict → Python list sparse dict (Arrow-serializable)."""
    return {
        "indices": d["indices"].tolist() if isinstance(d["indices"], torch.Tensor) else d["indices"],
        "values":  d["values"].tolist()  if isinstance(d["values"],  torch.Tensor) else d["values"],
        "size":    d["size"].tolist()    if isinstance(d["size"],    torch.Tensor) else list(d["size"]),
    }


def _convert_sample(elem: dict, rename: dict[str, str]) -> dict:
    """Convert one pt.gz sample dict to HFDS-compatible Arrow row.

    Sequence: DNA string passthrough or dense tensor → string.
    Signals:  tensor sparse dicts → list sparse dicts.
    Task keys are renamed according to ``rename`` (may be empty).
    """
    seq = elem["inputs"]["sequence"]
    if isinstance(seq, torch.Tensor):
        seq = dense_onehot_to_seq(seq)
    elif isinstance(seq, bytes):
        seq = seq.decode("utf-8")
    # else: already a DNA string

    name = elem["meta"]["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")

    outputs = {
        rename.get(k, k): _sparse_dict_to_lists(v)
        for k, v in elem["outputs"].items()
    }

    meta_out: dict = {"name": name}
    if "pad_side" not in elem["meta"]:
        # Source has no pad_side (pre-padded format): derive from physical N positions + strand.
        _, _, _, _strand = parse_tile_name(name)
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
        "inputs":  {"sequence": seq},
        "outputs": outputs,
        "meta":    meta_out,
    }


def _load_pt(input_path: Path) -> dict:
    """Load a .pt or .pt.gz file, preferring the mmap-friendly .pt companion."""
    pt_companion = Path(str(input_path).replace(".pt.gz", ".pt"))
    if pt_companion.exists():
        print(f"Loading (mmap) {pt_companion.name} ...", flush=True)
        return torch.load(pt_companion, mmap=True, weights_only=False)
    print(f"Loading (gzip) {input_path.name} — may take several minutes ...", flush=True)
    with gzip.open(input_path, "rb") as f:
        return torch.load(io.BytesIO(f.read()), weights_only=False)


def _parse_num_shards(value: str, splits: list[str]) -> dict[str, int]:
    """Parse --num-shards as int or 'split:N,...' into a per-split dict."""
    try:
        n = int(value)
        return {s: n for s in splits}
    except ValueError:
        result: dict[str, int] = {}
        for part in value.split(","):
            split, _, count = part.strip().partition(":")
            if not _ or not count:
                raise ValueError(f"Invalid --num-shards format: {value!r}. "
                                 "Use an integer or 'split:N,split:N,...'")
            result[split.strip()] = int(count.strip())
        return result


def convert(
    input_path: Path,
    output_path: Path,
    num_shards: str = "4",
    rename: dict[str, str] | None = None,
    metadata_basename: str | None = None,
    total_key_arg: str = "",
    is_pre_padded_arg: str = "false",
    seq_len_arg: int = 600,
) -> None:
    rename = rename or {}
    data = _load_pt(input_path)
    splits = list(data.keys())
    print(f"Splits: {splits}")

    shard_map = _parse_num_shards(num_shards, splits)

    # Validate metadata args against first sample (rename applied first).
    _first = _convert_sample(data[splits[0]][0], rename)
    task_names = list(_first["outputs"].keys())
    if total_key_arg not in task_names:
        print(f"ERROR: --total-key '{total_key_arg}' not in task names {task_names}", file=sys.stderr)
        sys.exit(1)
    total_key = total_key_arg
    n_tracks   = _first["outputs"][total_key]["size"][0]
    seq_len    = seq_len_arg
    is_pre_padded = (is_pre_padded_arg == "true")

    split_counts: dict[str, int] = {}

    hf_datasets.enable_progress_bars()
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name in splits:
        samples = data[split_name]
        n = len(samples)
        n_shards = shard_map.get(split_name, 4)
        rows_per_shard = math.ceil(n / n_shards)
        print(f"\n[{split_name}] {n:,} samples → {n_shards} shards ({rows_per_shard:,} rows/shard) ...", flush=True)

        # Convert shard-by-shard with from_list() on small chunks.
        # from_generator's fingerprinting (dill-serialises the generator + its captured
        # data) hangs for hours on large datasets; from_list on ~4k-row chunks stays fast.
        # Each converted shard is ~15 MB in Arrow — all n_shards fit in RAM comfortably.
        shard_datasets = []
        for shard_idx in range(n_shards):
            start = shard_idx * rows_per_shard
            end = min(start + rows_per_shard, n)
            rows = [
                _convert_sample(s, rename)
                for s in tqdm(samples[start:end],
                              desc=f"  shard {shard_idx + 1:>{len(str(n_shards))}}/{n_shards}",
                              unit="sample", leave=False)
            ]
            shard_datasets.append(hf_datasets.Dataset.from_list(rows))
            del rows

        split_dir = output_path / split_name
        print(f"[{split_name}] Saving {n_shards} shards ...", flush=True)
        hf_datasets.concatenate_datasets(shard_datasets).save_to_disk(
            str(split_dir), num_shards=n_shards
        )
        del shard_datasets
        split_counts[split_name] = n
        print(f"[{split_name}] Done.", flush=True)

    # Write dataset_dict.json so load_from_disk() recognises the directory as a DatasetDict.
    (output_path / "dataset_dict.json").write_text(
        json.dumps({"splits": list(split_counts.keys())})
    )
    print(f"\nAll splits saved → {output_path}")

    # Sidecar metadata inside the HFDS directory
    basename = metadata_basename or output_path.stem.split(".")[0]
    # strip common suffixes like "hfds" so the basename is the dataset name
    for suffix in (".hfds", ".no-one-hot"):
        if basename.endswith(suffix):
            basename = basename[: -len(suffix)]
    meta_path = output_path / f"{basename}.metadata.yaml"
    meta = {
        "sequence_format": "string",
        "signal_format":   "sparse_list",
        "seq_len":         int(seq_len),
        "n_tracks":        int(n_tracks),
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
    parser.add_argument("--input",      type=Path, required=True,
                        help="Path to source .pt or .pt.gz file")
    parser.add_argument("--outputdir",  type=Path, required=False,
                        help="Output HFDS directory. dataset_dict.json and split subdirs "
                             "are written directly under this path. "
                             "(default: next to input with .pt/.pt.gz replaced by .no-one-hot.hfds)")
    parser.add_argument("--num-shards", type=str,  default="4",
                        help="Arrow shards per split. Integer → same for all splits. "
                             "'train:76,valid:18,test:10' → per-split control. "
                             "Source HFDS uses 76/18/10 for train/valid/test. (default: 4)")
    parser.add_argument("--rename-track-names", type=str, default=None,
                        metavar="JSON",
                        help="Optional JSON dict to rename output task keys, e.g. "
                             "'{\"eCLIP\": \"total\"}'. Applied before writing. "
                             "Useful to align file keys with total_key= in GzListDataset.")
    parser.add_argument("--metadata-basename", type=str, default=None,
                        help="Base name for the metadata sidecar file written inside "
                             "the HFDS directory (default: derived from output dir name). "
                             "Use this to avoid collisions when multiple datasets share "
                             "the same parent directory.")
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

    if args.outputdir is None:
        stem = input_path.name
        for suffix in (".pt.gz", ".pt"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        output_path = input_path.parent / f"{stem}.no-one-hot.hfds"
    else:
        output_path = args.outputdir.resolve()

    convert(input_path, output_path,
            num_shards=args.num_shards,
            rename=rename,
            metadata_basename=args.metadata_basename,
            total_key_arg=args.total_key,
            is_pre_padded_arg=args.is_pre_padded,
            seq_len_arg=args.seq_len)


if __name__ == "__main__":
    main()
