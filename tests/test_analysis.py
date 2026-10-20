"""Checks patient-specific concept discovery and chart-ready calculations."""

import unittest

import pandas as pd

from src.hipergator.analysis import analyze_measurements, measurement_catalog


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame([
            {"measurement_id": 1, "measurement_concept_id": "300",
             "measurement_source_value": "Creatinine", "unit_source_value": "mg/dL",
             "measurement_datetime": "2200-01-01", "value_as_number": 1.0},
            {"measurement_id": 2, "measurement_concept_id": 300,
             "measurement_source_value": "Creatinine", "unit_source_value": "mg/dL",
             "measurement_datetime": "2200-01-02", "value_as_number": 2.0},
            {"measurement_id": 3, "measurement_concept_id": 301,
             "measurement_source_value": "Glucose", "unit_source_value": "mg/dL",
             "measurement_datetime": "2200-01-03", "value_as_number": 99.0},
            {"measurement_id": 4, "measurement_concept_id": 300,
             "measurement_source_value": "Creatinine", "unit_source_value": "mg/dL",
             "measurement_datetime": "2200-01-04", "value_as_number": None},
        ])

    def test_catalog_uses_source_label_when_standard_name_is_missing(self):
        candidates = measurement_catalog(self.rows, {}, "creatinine over time")
        self.assertEqual(candidates[0]["concept_id"], 300)
        self.assertEqual(candidates[0]["name"], "Creatinine")

    def test_time_series_filters_by_concept_and_keeps_source_ids(self):
        result = analyze_measurements(self.rows, {}, [300], "time_series", "history")
        self.assertEqual(result["source_rows"], 3)
        self.assertEqual(len(result["series"]), 1)
        series = result["series"][0]
        self.assertEqual([point["value"] for point in series["points"]], [1.0, 2.0])
        self.assertEqual([point["measurement_id"] for point in series["points"]], [1, 2])
        self.assertEqual(series["excluded_count"], 1)

    def test_distribution_and_stats_use_only_numeric_matches(self):
        distribution = analyze_measurements(self.rows, {}, [300], "distribution", "history")
        self.assertEqual(sum(bin_["count"] for bin_ in distribution["series"][0]["bins"]), 2)
        statistics = analyze_measurements(self.rows, {}, [300], "statistics", "history")
        self.assertEqual(statistics["series"][0]["stats"]["median"], 1.5)
        self.assertEqual(statistics["series"][0]["stats"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
