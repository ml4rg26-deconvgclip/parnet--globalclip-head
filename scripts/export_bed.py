"""Export a BED6 file from an HFDS or .pt dataset.

Each row corresponds to one sample tile.

Columns:
  1-3  chrom / start / end  (0-based half-open, from tile name)
  4    name                 ``tile_id;{json_metadata}``
                            tile_id = the original ``chrN:start-end:strand`` name
                            json_metadata = remaining meta fields + split name,
                            keys alphabetically sorted for stable output
  5    score                ``pad_side`` integer:
                               -1  tile fills window exactly (no parnet padding)
                                0  center-padded
                                1  left-padded
                                2  right-padded
  6    strand

``pad_side`` is read from ``meta["pad_side"]``, which is set by ``convert_hfds_to_pt.py``
(coordinate-validated for HFDS / pre-padded .pt) and preserved by ``strip_element_padding``
for stripped .pt files.

Tile name format in both dataset types: ``chrN:start-end:strand``
(e.g. ``chr3:618239-620239:+``).

Usage
-----
# From HFDS
python scripts/export_bed.py \\
    --input results/spliceosome-hepg2-hfds/datasets/dataset.hfds \\
    --format hfds \\
    --output results/spliceosome-hepg2-hfds/datasets/tiles.bed

# From .pt / .pt.gz
python scripts/export_bed.py \\
    --input results/spliceosome-hepg2/datasets/dataset.pt \\
    --format pt \\
    --output results/spliceosome-hepg2/datasets/tiles.bed

# Specific splits only (space-separated; default: all splits)
python scripts/export_bed.py \\
    --input ... --format hfds --splits train valid --output tiles.bed
"""

import argparse
import gzip
import io
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from parnet_demo_utils import parse_tile_name


def _get_name(elem: dict) -> str:
    name = elem["meta"]["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return name


def _write_rows(
    out,
    samples,
    split_name: str,
    n_total: int,
) -> int:
    n_written = 0
    for elem in tqdm(samples, desc=split_name, total=n_total):
        name = _get_name(elem)
        chrom, start, end, strand = parse_tile_name(name)
        score = elem["meta"].get("pad_side", -1)
        extra_meta = {k: v for k, v in elem["meta"].items() if k != "name"}
        extra_meta["split"] = split_name
        name_col = f"{name};{json.dumps(extra_meta, sort_keys=True, separators=(',', ':'))}"
        out.write(f"{chrom}\t{start}\t{end}\t{name_col}\t{score}\t{strand}\n")
        n_written += 1
    return n_written


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_pt(path: Path) -> dict:
    pt_companion = Path(str(path).replace(".pt.gz", ".pt"))
    if pt_companion.exists() and not path.suffix == ".pt":
        print(f"Loading (mmap) {pt_companion.name} ...", flush=True)
        return torch.load(pt_companion, mmap=True, weights_only=False)
    if path.suffix == ".gz":
        print(f"Loading (gzip) {path.name} ...", flush=True)
        with gzip.open(path, "rb") as f:
            return torch.load(io.BytesIO(f.read()), weights_only=False)
    print(f"Loading {path.name} ...", flush=True)
    return torch.load(path, mmap=True, weights_only=False)


def export_pt(
    input_path: Path,
    output_path: Path,
    splits: list[str] | None,
) -> None:
    data = _load_pt(input_path)
    available = list(data.keys())
    target_splits = splits if splits else available

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    with open(output_path, "w") as f:
        for split_name in target_splits:
            if split_name not in data:
                print(f"WARNING: split '{split_name}' not found in {available}; skipping.", file=sys.stderr)
                continue
            samples = data[split_name]
            total_written += _write_rows(f, samples, split_name, len(samples))

    print(f"\nWrote {total_written:,} rows → {output_path}")


def export_hfds(
    input_path: Path,
    output_path: Path,
    splits: list[str] | None,
) -> None:
    from datasets import load_from_disk
    hfds = load_from_disk(str(input_path))
    available = list(hfds.keys())
    target_splits = splits if splits else available

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    with open(output_path, "w") as f:
        for split_name in target_splits:
            if split_name not in hfds:
                print(f"WARNING: split '{split_name}' not found in {available}; skipping.", file=sys.stderr)
                continue
            split_ds = hfds[split_name]
            total_written += _write_rows(f, split_ds, split_name, len(split_ds))

    print(f"\nWrote {total_written:,} rows → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  type=Path, required=True,
                        help="Path to source dataset (.pt / .pt.gz or HFDS directory)")
    parser.add_argument("--format", choices=["pt", "hfds"], required=True,
                        help="Dataset format: 'pt' for .pt/.pt.gz, 'hfds' for HuggingFace Arrow")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output BED6 file path")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Splits to export (space-separated; default: all splits)")
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.format == "pt":
        export_pt(input_path, args.output.resolve(), args.splits)
    else:
        export_hfds(input_path, args.output.resolve(), args.splits)


if __name__ == "__main__":
    main()
