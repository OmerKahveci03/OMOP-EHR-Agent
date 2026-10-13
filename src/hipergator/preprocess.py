import os
import glob

import pandas as pd


from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Optional

from . import HPG_BASE_DIR, PID

from .utils.paths import get_path

def raw_csv_to_parquet(raw_file: str, write_dir: str, meta_dir: str, overwrite: bool) -> None:
    """ Load the raw csv file and write to the `write_dir` as a parquet.
    
    Coerces any 'concept_id' column to `Int64` before doing so."""

    file_type: str = os.path.basename(raw_file).removesuffix(".csv")

    # -- Skip Duplicate Work --
    write_fp: str = os.path.join(write_dir, f"{file_type}.parquet")
    part_fp: str = f"{write_fp}.part"

    if not overwrite and os.path.exists(write_fp):
        return

    # Remove incomplete/corrupted file
    if os.path.exists(part_fp):
        os.remove(part_fp)

    print(f" - Reading {file_type}...\n")

    # -- Loading --
    df: pd.DataFrame = pd.read_csv(raw_file, low_memory=False)
    shape: tuple[int, int] = df.shape
    print(f"  - Loaded {file_type} with shape: {shape}\n")

    for col in df.columns:
        if "concept_id" in col or col == PID:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        elif df[col].dtype == "object":
            df[col] = df[col].astype("string")

    # -- Save Metadata --
    meta_fp: str = os.path.join(meta_dir, f"{file_type}_metadata.csv")
    meta_df: pd.DataFrame = df.dtypes.rename_axis("column").to_frame("dtype")
    meta_df["row_count"] = shape[0]
    meta_df["pct_missing"] = df.isna().mean().mul(100)
    if PID in df.columns:
        meta_df[f"{PID}_nunique"] = df[PID].nunique()

    meta_df.to_csv(meta_fp)

    # -- Write to part first, then rename once successful --
    # NOTE: We do it in this order to ensure that the write was successful.
    df.to_parquet(part_fp, index=False)
    os.replace(part_fp, write_fp)
    print(f" - Wrote {file_type} to '{write_fp}'")


def preprocess_data(base_dir: str = HPG_BASE_DIR, max_workers: Optional[int] = None,
                    overwrite: bool = False) -> None:
    """ Convert raw csvs to parquet format. """

    raw_dir: str = get_path(path_type="raw", require_exists=True, base_dir=base_dir)
    cleaned_dir: str = get_path(path_type="cleaned", makedir=True, base_dir=base_dir)
    meta_dir: str = get_path(path_type="metadata", makedir=True, base_dir=base_dir)

    print(" === Preprocessing Raw Data === \n")

    print(f"Reading from: '{raw_dir}'\nWriting to: '{cleaned_dir}'")

    raw_files: list[str] = glob.glob(os.path.join(raw_dir, "*.csv"))

    # --- Process Files in Parallel ---
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures: list[Future] = [
            executor.submit(raw_csv_to_parquet, 
                            raw_file=raw_file, 
                            write_dir=cleaned_dir,
                            meta_dir=meta_dir,
                            overwrite=overwrite)

            for raw_file in raw_files
        ]
        for future in as_completed(futures):
            future.result()

    print("\nDone preprocessing data!\n")

    # --- Concat All Metadata ---
    master_meta_fp: str = os.path.join(meta_dir, "metadata_master.csv")
    if overwrite or not os.path.exists(master_meta_fp):
        print("Concatenating all metadata...")

        meta_files: list[str] = glob.glob(os.path.join(meta_dir, "*metadata.csv"))
        meta_dfs: list[pd.DataFrame] = []
        for meta_file in meta_files:
            file_type: str = os.path.basename(meta_file).removesuffix("_metadata.csv")
            meta_df: pd.DataFrame = pd.read_csv(meta_file)

            meta_df["file_type"] = file_type
            meta_dfs.append(meta_df)

        meta_df = pd.concat(meta_dfs, ignore_index=True)
        meta_df.to_csv(master_meta_fp, index=False)

        print("\nDone concatenating all metadata!")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    overwrite: bool = args.overwrite

    preprocess_data(overwrite=overwrite)
