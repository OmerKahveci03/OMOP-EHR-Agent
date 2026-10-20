import os
import glob
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Optional

from . import HPG_BASE_DIR, PID

from .utils.paths import get_path


def merge_concept_vocabularies(local_file: str, athena_file: str,
                               write_dir: str, meta_dir: str,
                               overwrite: bool = False) -> bool:
    """Stream Athena concepts and append MIMIC concepts to the cleaned table.

    Returns whether the combined parquet was rebuilt. The source CSV files are
    retained so the merge can be reproduced and refreshed later.
    """
    write_fp = os.path.join(write_dir, "concept.parquet")
    part_fp = f"{write_fp}.part"
    marker_fp = os.path.join(meta_dir, "concept_sources.json")
    sources = {
        name: {"size": os.path.getsize(path), "mtime_ns": os.stat(path).st_mtime_ns}
        for name, path in (("mimic", local_file), ("athena", athena_file))
    }
    if not overwrite and os.path.exists(write_fp) and os.path.exists(marker_fp):
        with open(marker_fp, encoding="utf-8") as marker:
            if json.load(marker) == sources:
                return False

    local = pd.read_csv(local_file, dtype="string", low_memory=False)
    athena_columns = pd.read_csv(athena_file, sep="\t", nrows=0).columns.tolist()
    columns = list(dict.fromkeys([*local.columns, *athena_columns]))
    if "concept_id" not in columns or "concept_name" not in columns:
        raise ValueError("Both concept sources must include concept_id and concept_name")
    columns = ["concept_id", *[column for column in columns if column != "concept_id"]]
    schema = pa.schema([
        pa.field(column, pa.int64() if column == "concept_id" else pa.string())
        for column in columns
    ])

    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.reindex(columns=columns)
        frame["concept_id"] = pd.to_numeric(frame["concept_id"], errors="raise").astype("Int64")
        for column in columns[1:]:
            frame[column] = frame[column].astype("string")
        return frame.dropna(subset=["concept_id"])

    local = normalize(local).drop_duplicates(subset=["concept_id"], keep="last")
    local_ids = set(local["concept_id"].astype(int))
    missing_counts = pd.Series(0, index=columns, dtype="int64")
    row_count = 0
    if os.path.exists(part_fp):
        os.remove(part_fp)
    writer = pq.ParquetWriter(part_fp, schema=schema, compression="snappy")
    try:
        for chunk in pd.read_csv(athena_file, sep="\t", dtype="string",
                                 chunksize=100_000):
            chunk = normalize(chunk)
            chunk = chunk[~chunk["concept_id"].isin(local_ids)]
            if chunk.empty:
                continue
            writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
            missing_counts += chunk.isna().sum()
            row_count += len(chunk)
        writer.write_table(pa.Table.from_pandas(local, schema=schema, preserve_index=False))
        missing_counts += local.isna().sum()
        row_count += len(local)
    finally:
        writer.close()

    os.replace(part_fp, write_fp)
    meta_df = pd.DataFrame({
        "dtype": ["Int64" if column == "concept_id" else "string" for column in columns],
        "row_count": row_count,
        "pct_missing": missing_counts.mul(100).div(row_count) if row_count else 0,
    }, index=pd.Index(columns, name="column"))
    meta_df.to_csv(os.path.join(meta_dir, "concept_metadata.csv"))
    with open(f"{marker_fp}.part", "w", encoding="utf-8") as marker:
        json.dump(sources, marker)
    os.replace(f"{marker_fp}.part", marker_fp)
    print(f" - Wrote {row_count} combined concepts to '{write_fp}'")
    return True

def preprocess_concept_ancestor(source_file: str, write_dir: str,
                                meta_dir: str, overwrite: bool = False) -> bool:
    """Stream Athena's concept hierarchy into a typed parquet table."""
    columns = ["ancestor_concept_id", "descendant_concept_id",
               "min_levels_of_separation", "max_levels_of_separation"]
    write_fp = os.path.join(write_dir, "concept_ancestor.parquet")
    marker_fp = os.path.join(meta_dir, "concept_ancestor_source.json")
    source = {"size": os.path.getsize(source_file),
              "mtime_ns": os.stat(source_file).st_mtime_ns}
    if not overwrite and os.path.exists(write_fp) and os.path.exists(marker_fp):
        with open(marker_fp, encoding="utf-8") as marker:
            if json.load(marker) == source:
                return False

    header = pd.read_csv(source_file, sep="\t", nrows=0).columns.tolist()
    if set(header) != set(columns):
        raise ValueError(f"Unexpected CONCEPT_ANCESTOR columns: {header}")

    schema = pa.schema([pa.field(column, pa.int64()) for column in columns])
    part_fp = f"{write_fp}.part"
    writer = pq.ParquetWriter(part_fp, schema=schema, compression="snappy")
    row_count = 0
    try:
        for chunk in pd.read_csv(source_file, sep="\t", dtype="string",
                                 usecols=columns, chunksize=100_000):
            for column in columns:
                chunk[column] = pd.to_numeric(chunk[column], errors="raise").astype("Int64")
            if chunk[columns].isna().any().any():
                raise ValueError("CONCEPT_ANCESTOR contains missing values")
            writer.write_table(pa.Table.from_pandas(
                chunk[columns], schema=schema, preserve_index=False))
            row_count += len(chunk)
    except BaseException:
        writer.close()
        if os.path.exists(part_fp):
            os.remove(part_fp)
        raise
    writer.close()
    os.replace(part_fp, write_fp)
    pd.DataFrame({"dtype": "Int64", "row_count": row_count,
                  "pct_missing": 0.0}, index=pd.Index(columns, name="column")).to_csv(
                      os.path.join(meta_dir, "concept_ancestor_metadata.csv"))
    with open(marker_fp, "w", encoding="utf-8") as marker:
        json.dump(source, marker)
    print(f" - Wrote {row_count} concept ancestor links to '{write_fp}'")
    return True


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
    """Convert MIMIC CSVs and combine local and Athena concepts in parquet."""

    raw_dir: str = get_path(path_type="raw", require_exists=True, base_dir=base_dir)
    cleaned_dir: str = get_path(path_type="cleaned", makedir=True, base_dir=base_dir)
    meta_dir: str = get_path(path_type="metadata", makedir=True, base_dir=base_dir)
    local_concepts = os.path.join(raw_dir, "2b_concept.csv")
    athena_concepts = os.path.join(base_dir, "vocabulary_download_v5", "CONCEPT.csv")
    athena_ancestors = os.path.join(base_dir, "vocabulary_download_v5", "CONCEPT_ANCESTOR.csv")
    for source in (local_concepts, athena_concepts, athena_ancestors):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Missing concept source: {source}")

    print(" === Preprocessing Raw Data === \n")

    print(f"Reading from: '{raw_dir}'\nWriting to: '{cleaned_dir}'")

    raw_files: list[str] = [path for path in glob.glob(os.path.join(raw_dir, "*.csv"))
                            if os.path.normcase(path) != os.path.normcase(local_concepts)]

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

    concepts_updated = merge_concept_vocabularies(
        local_file=local_concepts, athena_file=athena_concepts,
        write_dir=cleaned_dir, meta_dir=meta_dir, overwrite=overwrite,
    )
    ancestors_updated = preprocess_concept_ancestor(
        source_file=athena_ancestors, write_dir=cleaned_dir,
        meta_dir=meta_dir, overwrite=overwrite,
    )

    print("\nDone preprocessing data!\n")

    # --- Concat All Metadata ---
    master_meta_fp: str = os.path.join(meta_dir, "metadata_master.csv")
    if overwrite or concepts_updated or ancestors_updated or not os.path.exists(master_meta_fp):
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
