"""Checks limits on workbook summaries for high-cardinality data."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.hipergator.data_analysis.summarize_column import summarize_categorical
from src.hipergator.data_analysis.summarize_df import summarize_df
from src.hipergator.data_analysis.analyze_dataset import analyze_dataset


class DataAnalysisLimitTests(unittest.TestCase):
    def test_analysis_skips_vocabulary_table(self):
        with tempfile.TemporaryDirectory() as directory:
            cleaned = Path(directory) / "cleaned_data"
            cleaned.mkdir()
            pd.DataFrame({"concept_id": [300], "concept_name": ["Example"]}).to_parquet(
                cleaned / "concept.parquet"
            )
            pd.DataFrame({"measurement_concept_id": [300, 300],
                          "value_as_number": [1.0, 2.0]}).to_parquet(
                cleaned / "measurement.parquet"
            )
            with patch("src.hipergator.data_analysis.analyze_dataset.write_df_summary",
                       side_effect=lambda summary, path: path) as write:
                paths = analyze_dataset(base_dir=directory)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith("measurement_summary.xlsx"))
            self.assertEqual(write.call_args.kwargs["summary"]["dataset"]["total_groups"].iloc[0], 1)

    def test_categorical_keeps_top_values_and_full_cardinality(self):
        values = [f"value-{index}" for index in range(105)]
        summary = summarize_categorical(pd.DataFrame({"category": values}), ["category"])
        self.assertEqual(len(summary), 100)
        self.assertEqual(summary["unique_count"].iloc[0], 105)

    def test_numerical_groups_keep_most_frequent_concepts(self):
        df = pd.DataFrame({
            "measurement_concept_id_name": ["rare", "common", "common", "middle", "middle", "middle"],
            "value_as_number": [1, 2, 4, 3, 6, 9],
            "source_value": ["a", "b", "c", "d", "e", "f"],
        })
        summary = summarize_df(df, group_by="measurement_concept_id_name",
                               group_types=["numerical"], max_groups=2)
        self.assertEqual(summary["dataset"]["group"].tolist(), ["middle", "common"])
        self.assertEqual(summary["dataset"]["total_groups"].tolist(), [3, 3])
        self.assertEqual(summary["dataset"]["included_groups"].tolist(), [2, 2])
        self.assertEqual(set(summary["numerical"]["group"]), {"middle", "common"})
        self.assertNotIn("group", summary["categorical"].columns)


if __name__ == "__main__":
    unittest.main()
