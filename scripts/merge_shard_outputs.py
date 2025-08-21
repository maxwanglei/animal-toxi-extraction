"""
Merge per-shard outputs from the enhanced SLURM array into single combined CSVs.

Usage:
  python -m scripts.merge_shard_outputs --base-out ./toxicity_output_meta_enhanced

This will look for shard directories under base-out (e.g., shard_0, shard_1, ...),
and merge their combined_lab_results.csv and combined_organ_results.csv into
base-out/combined_lab_results.csv and base-out/combined_organ_results.csv.
"""

import argparse
import glob
import os
import sys
from typing import List, Optional


def merge_csvs(files: List[str], out_file: str, key_cols: Optional[List[str]] = None) -> int:
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install -r requirements.txt", file=sys.stderr)
        return 2

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}", file=sys.stderr)

    if not frames:
        print(f"No input CSVs found to merge for {out_file}")
        return 1

    df = pd.concat(frames, ignore_index=True)
    if key_cols:
        before = len(df)
        df = df.drop_duplicates(subset=key_cols)
        after = len(df)
        print(f"Deduplicated {before - after} rows on keys {key_cols}")

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Wrote {len(df)} rows to {out_file}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-out", required=True, help="Base output dir, e.g., ./toxicity_output_meta_enhanced")
    ap.add_argument(
        "--dedupe-keys-lab",
        nargs="*",
        default=None,
        help="Optional columns to use for lab CSV deduplication",
    )
    ap.add_argument(
        "--dedupe-keys-organ",
        nargs="*",
        default=None,
        help="Optional columns to use for organ CSV deduplication",
    )
    args = ap.parse_args()

    base = args.base_out.rstrip("/")
    shard_dirs = sorted(glob.glob(os.path.join(base, "shard_*")))
    if not shard_dirs:
        print(f"No shard_* directories found under {base}", file=sys.stderr)
        return 1

    lab_files = [os.path.join(d, "combined_lab_results.csv") for d in shard_dirs]
    organ_files = [os.path.join(d, "combined_organ_results.csv") for d in shard_dirs]
    lab_files = [f for f in lab_files if os.path.exists(f)]
    organ_files = [f for f in organ_files if os.path.exists(f)]

    rc1 = merge_csvs(lab_files, os.path.join(base, "combined_lab_results.csv"), args.dedupe_keys_lab)
    rc2 = merge_csvs(
        organ_files,
        os.path.join(base, "combined_organ_results.csv"),
        args.dedupe_keys_organ,
    )
    return rc1 or rc2


if __name__ == "__main__":
    sys.exit(main())
