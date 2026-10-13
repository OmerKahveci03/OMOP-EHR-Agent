"""Write a descriptive summary workbook for each cleaned parquet table."""

from pathlib import Path

import pandas as pd

from .. import HPG_BASE_DIR
from ..utils.paths import get_path
from .summarize_df import DatasetSummary, summarize_df, write_df_summary


def analyze_dataset(base_dir: str = HPG_BASE_DIR) -> list[str]:
    """Summarize each parquet file in cleaned_data and return the workbook paths."""

    # --- Locate input tables and summary output ---
    cleaned_dir: Path = Path(get_path(path_type="cleaned", base_dir=base_dir, require_exists=True))
    summary_dir: Path = Path(get_path(path_type="summary", base_dir=base_dir, makedir=True))
    parquet_files: list[Path] = sorted(cleaned_dir.glob("*.parquet"))

    # --- Summarize each table ---
    workbook_paths: list[str] = []
    for parquet_file in parquet_files:
        df: pd.DataFrame = pd.read_parquet(path=parquet_file)
        summary: DatasetSummary = summarize_df(df=df)
        workbook_path: str = write_df_summary(
            summary=summary,
            path=str(summary_dir / f"{parquet_file.stem}_summary.xlsx"),
        )
        workbook_paths.append(workbook_path)

    return workbook_paths


if __name__ == "__main__":
    for path in analyze_dataset():
        print(path)
