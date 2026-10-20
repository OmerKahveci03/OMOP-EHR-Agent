"""Write a descriptive summary workbook for each cleaned parquet table."""

from pathlib import Path

import pandas as pd

from .. import HPG_BASE_DIR
from ..concepts import load_concept_names
from ..utils.paths import get_path
from .summarize_df import DatasetSummary, summarize_df, write_df_summary


def analyze_dataset(base_dir: str = HPG_BASE_DIR) -> list[str]:
    """Summarize each parquet file in cleaned_data and return the workbook paths."""

    # --- Locate input tables and summary output ---
    cleaned_dir: Path = Path(get_path(path_type="cleaned", base_dir=base_dir, require_exists=True))
    summary_dir: Path = Path(get_path(path_type="summary", base_dir=base_dir, makedir=True))
    parquet_files: list[Path] = sorted(cleaned_dir.glob("*.parquet"))
    concept_names: dict[int, str] = load_concept_names(data_dir=str(cleaned_dir))

    # --- Summarize each table ---
    workbook_paths: list[str] = []
    for parquet_file in parquet_files:
        if parquet_file.stem == "concept":
            continue  # Vocabulary lookup table, not a patient dataset.
        df: pd.DataFrame = pd.read_parquet(path=parquet_file)
        concept_columns: list[str] = [
            column for column in df.columns if column.endswith("_concept_id")
        ]
        for column in concept_columns:
            concept_ids: pd.Series = pd.to_numeric(df[column], errors="coerce").astype("Int64")
            df[f"{column}_name"] = concept_ids.map(concept_names)

        # A long format OMOP table has one row per recorded concept. Other
        # concept columns (such as units) are labels, not grouping keys.
        table_name: str = parquet_file.stem.split("_", maxsplit=1)[-1]
        primary_concept_column: str = f"{table_name}_concept_id"
        group_by: str | None = (
            f"{primary_concept_column}_name"
            if primary_concept_column in concept_columns else None
        )
        summary: DatasetSummary = summarize_df(
            df=df,
            group_by=group_by,
            group_types=["numerical"] if group_by else "all",
            max_groups=1000 if group_by else None,
        )
        workbook_path: str = write_df_summary(
            summary=summary,
            path=str(summary_dir / f"{parquet_file.stem}_summary.xlsx"),
        )
        workbook_paths.append(workbook_path)

    return workbook_paths


if __name__ == "__main__":
    for path in analyze_dataset():
        print(path)
