#!/usr/bin/env python3
"""
Flatten multi-resident processed data: one row per (event, activity) with activity_flattened and resident_info.

Reads data_processed.csv and optional data_processed_train.csv / data_processed_test.csv from a given folder,
explodes rows with multiple activities into one row per activity, adds activity_flattened and resident_info,
and saves data_processed_flattened.csv (and _train / _test) in the same folder.

Usage:
  python -m src.data.multiresident_flatten data/raw/casas/twor.2009
  python -m src.data.multiresident_flatten --folder data/raw/casas/twor.2009
"""

import argparse
import ast
import re
from pathlib import Path

import pandas as pd


def _parse_activity_list(ser: pd.Series) -> pd.Series:
    """Parse activity_list column: may be list (in-memory) or str (from CSV)."""
    def parse(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return []
        return []

    return ser.map(parse)


def _resident_info(activity_flattened: str) -> str:
    """Infer resident from activity name: R1, R2, R3, ... or 'both' if none."""
    if pd.isna(activity_flattened) or not activity_flattened:
        return "both"
    # Match R followed by one or more digits (R1, R2, R10, ...)
    m = re.search(r"R(\d+)", activity_flattened, re.IGNORECASE)
    if m:
        return f"R{m.group(1)}"
    return "both"


def _strip_resident_prefix(label: str) -> str:
    """Remove leading R1_, R2_, R3_, ... from activity label if present."""
    if pd.isna(label) or not label:
        return label
    return re.sub(r"^R\d+_", "", label, flags=re.IGNORECASE)


def flatten_processed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten multi-resident data: one row per activity per event.

    - activity_flattened: single activity for this row (from first_activity if num_activities==1, else one from activity_list).
    - resident_info: R1/R2/R3/... if activity name contains it, else "both".
    """
    required = ["num_activities", "first_activity", "activity_list"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    # Parse activity_list if it came from CSV (string representation of list)
    activity_list_parsed = _parse_activity_list(df["activity_list"])

    # Rows with single (or zero) activity: keep one row, activity_flattened = first_activity
    single = df["num_activities"] <= 1
    df_single = df.loc[single].copy()
    df_single["activity_flattened"] = df_single["first_activity"].astype(str)

    # Rows with multiple activities: explode to one row per activity
    multi = ~single
    if multi.any():
        df_multi = df.loc[multi].copy()
        df_multi["_act_list"] = activity_list_parsed.loc[multi].values
        exploded = df_multi.explode("_act_list").reset_index(drop=True)
        exploded["activity_flattened"] = exploded["_act_list"].astype(str)
        exploded = exploded.drop(columns=["_act_list"])
        df_multi_flat = exploded
    else:
        df_multi_flat = pd.DataFrame()

    out = pd.concat([df_single, df_multi_flat], ignore_index=True)

    # resident_info from raw activity label (before stripping prefix)
    out["resident_info"] = out["activity_flattened"].map(_resident_info)
    # remove R1_, R2_, R3_, ... from activity_flattened for the new label
    out["activity_flattened"] = out["activity_flattened"].map(_strip_resident_prefix)

    return out


def run_folder(folder: Path) -> None:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    main_file = folder / "data_processed.csv"
    if not main_file.exists():
        raise FileNotFoundError(f"Missing {main_file}")

    def read_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def write_csv(df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False)

    # Full dataset
    df = read_csv(main_file)
    out = flatten_processed(df)
    out_path = folder / "data_processed_flattened.csv"
    write_csv(out, out_path)
    print(f"Wrote {out_path} (shape {out.shape})")

    # Train
    train_file = folder / "data_processed_train.csv"
    if train_file.exists():
        df_train = read_csv(train_file)
        out_train = flatten_processed(df_train)
        out_train_path = folder / "data_processed_flattened_train.csv"
        write_csv(out_train, out_train_path)
        print(f"Wrote {out_train_path} (shape {out_train.shape})")

    # Test
    test_file = folder / "data_processed_test.csv"
    if test_file.exists():
        df_test = read_csv(test_file)
        out_test = flatten_processed(df_test)
        out_test_path = folder / "data_processed_flattened_test.csv"
        write_csv(out_test, out_test_path)
        print(f"Wrote {out_test_path} (shape {out_test.shape})")


def main():
    parser = argparse.ArgumentParser(
        description="Flatten multi-resident processed CSVs: add activity_flattened and resident_info, save _flattened CSVs."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing data_processed.csv (and optionally _train / _test).",
    )
    args = parser.parse_args()
    run_folder(args.folder)


if __name__ == "__main__":
    main()
