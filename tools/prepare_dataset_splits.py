#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vista.data import load_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test CSV files from OPH metadata."
    )
    parser.add_argument("--zip_path", default="", help="Optional dataset zip to extract first.")
    parser.add_argument("--extract_dir", default="data/minimum_dataset")
    parser.add_argument("--metadata", default="", help="CSV/XLSX metadata path. Defaults to extracted metadata.xlsx.")
    parser.add_argument("--output_dir", default="data/minimum_dataset/splits")
    parser.add_argument("--image_col", default="OCT_pathway")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add_image_path_alias", action="store_true")
    return parser.parse_args()


def maybe_extract(zip_path: str, extract_dir: str) -> None:
    if not zip_path:
        return
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print(f"Extracted {zip_path} -> {extract_dir}")


def split_rows(rows, split_col: str, train_ratio: float, val_ratio: float, seed: int):
    if rows and split_col in rows[0] and any(str(row.get(split_col, "")).strip() for row in rows):
        out = {"train": [], "val": [], "test": []}
        aliases = {"valid": "val", "validation": "val"}
        for row in rows:
            split = str(row.get(split_col, "")).strip().lower()
            split = aliases.get(split, split)
            if split not in out:
                raise ValueError(f"unsupported split value: {row.get(split_col)!r}")
            out[split].append(row)
        return out

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n < 3:
        return {"train": shuffled, "val": shuffled, "test": shuffled}
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    maybe_extract(args.zip_path, args.extract_dir)

    metadata = args.metadata
    if not metadata:
        metadata = os.path.join(args.extract_dir, "metadata.xlsx")
    rows = load_table(metadata)
    if not rows:
        raise ValueError(f"no rows loaded from {metadata}")
    if args.image_col not in rows[0]:
        raise ValueError(f"image column {args.image_col!r} not found. Columns: {list(rows[0])}")

    if args.add_image_path_alias:
        for row in rows:
            row.setdefault("image_path", row.get(args.image_col, ""))

    splits = split_rows(rows, args.split_col, args.train_ratio, args.val_ratio, args.seed)
    output_dir = Path(args.output_dir)
    for name, split_rows_ in splits.items():
        write_csv(output_dir / f"{name}.csv", split_rows_)
        print(f"{name}: {len(split_rows_)} -> {output_dir / f'{name}.csv'}")


if __name__ == "__main__":
    main()
