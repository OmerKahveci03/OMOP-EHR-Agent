"""Checks for encounter scoping and measurement aggregation."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.hipergator.patient_search import (
    run_patient_request, search_patient, select_action_plan, select_retrieval_plan,
)
from src.hipergator.retrieval import latest_encounter, patient_table, summarize_measurements


class Tokenizer:
    def encode(self, text):
        return list(text)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        pd.DataFrame([{"person_id": 7, "year_of_birth": 1980}]).to_parquet(root / "person.parquet")
        pd.DataFrame([
            {"person_id": 7, "visit_occurrence_id": 10,
             "visit_start_datetime": "2200-01-01", "visit_end_datetime": "2200-01-02"},
            {"person_id": 7, "visit_occurrence_id": 11,
             "visit_start_datetime": "2201-08-12", "visit_end_datetime": "2201-08-13"},
        ]).to_parquet(root / "visit_occurrence.parquet")
        pd.DataFrame([
            {"person_id": 7, "measurement_id": 1, "visit_occurrence_id": 10,
             "measurement_datetime": "2200-01-01", "measurement_concept_id": 300,
             "measurement_source_value": "Example", "unit_concept_id": 900,
             "unit_source_value": "mg/dL", "value_as_number": 100.0,
             "value_as_concept_id": None, "value_source_value": None},
            {"person_id": 7, "measurement_id": 2, "visit_occurrence_id": 11,
             "measurement_datetime": "2201-08-12", "measurement_concept_id": 300,
             "measurement_source_value": "Example", "unit_concept_id": 900,
             "unit_source_value": "mg/dL", "value_as_number": 10.0,
             "value_as_concept_id": None, "value_source_value": None},
            {"person_id": 7, "measurement_id": 89318244079633095, "visit_occurrence_id": 11,
             "measurement_datetime": "2201-08-13", "measurement_concept_id": 300,
             "measurement_source_value": "Example", "unit_concept_id": 900,
             "unit_source_value": "mg/dL", "value_as_number": 14.0,
             "value_as_concept_id": None, "value_source_value": None},
            {"person_id": 7, "measurement_id": 4, "visit_occurrence_id": 11,
             "measurement_datetime": "2201-08-13", "measurement_concept_id": 301,
             "measurement_source_value": "Test", "unit_concept_id": None,
             "unit_source_value": None, "value_as_number": None,
             "value_as_concept_id": 500, "value_source_value": "Positive"},
        ]).to_parquet(root / "measurement.parquet")

    def test_latest_visit_and_measurement_summary(self):
        encounter = latest_encounter(self.directory.name, 7)
        self.assertEqual(encounter["visit_occurrence_id"], 11)
        latest = patient_table(self.directory.name, "measurement", 7,
                               "latest_encounter", encounter)
        self.assertEqual(len(latest), 3)
        summary = summarize_measurements(latest)
        self.assertEqual(summary["source_rows"], 3)
        numeric = next(row for row in summary["shown_rows"] if row["result_type"] == "numeric")
        self.assertEqual(numeric["count"], 2)
        self.assertEqual((numeric["min"], numeric["mean"], numeric["max"]), (10, 12, 14))
        self.assertTrue(math.isclose(numeric["std"], math.sqrt(8)))
        self.assertEqual(numeric["latest_value"], 14)
        self.assertEqual(numeric["latest_measurement_id"], "89318244079633095")
        categorical = next(row for row in summary["shown_rows"] if row["result_type"] == "non_numeric")
        self.assertEqual(categorical["value_as_concept_id"], 500)
        self.assertEqual(categorical["count"], 1)

        history = patient_table(self.directory.name, "measurement", 7, "history", None)
        self.assertEqual(len(history), 4)
        self.assertEqual(summarize_measurements(history)["total_rows"], 3)

    def test_plan_and_prompt_use_selected_scope(self):
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=16384))
        with patch("src.hipergator.patient_search.generate", return_value='{"tables":["measurement"],"scope":"history"}'):
            self.assertEqual(select_retrieval_plan(model, Tokenizer(), "trend", ["measurement"]),
                             (["measurement"], "history"))
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"tables":["measurement"],"scope":"latest_encounter"}', "answer",
        ]) as generate:
            answer, evidence = search_patient(7, "give me measurements", self.directory.name,
                                              model, Tokenizer(), {300: "Example", 900: "mg/dL"})
        self.assertEqual(answer, "answer")
        self.assertEqual(evidence[0]["retrieval_scope"], "latest_encounter")
        self.assertEqual(evidence[-1]["source_rows"], 3)
        self.assertIn("measurement_concept_id_name", generate.call_args.kwargs["user"])

        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"tables":["measurement"],"scope":"history"}', "answer",
        ]):
            _, history_evidence = search_patient(7, "trend across visits", self.directory.name,
                                                 model, Tokenizer(), {300: "Example", 900: "mg/dL"})
        self.assertEqual(history_evidence[0]["retrieval_scope"], "history")
        self.assertEqual(history_evidence[-1]["source_rows"], 4)

    def test_action_plan_rejects_concepts_outside_patient_catalog(self):
        model = SimpleNamespace()
        with patch("src.hipergator.patient_search.generate", return_value=(
            '{"action":"time_series","scope":"history","tables":["measurement"],'
            '"concept_ids":[999]}'
        )):
            with self.assertRaisesRegex(ValueError, "Unavailable measurement concepts"):
                select_action_plan(model, Tokenizer(), "trend", ["measurement"],
                                   [{"concept_id": 300, "name": "Example"}])

    def test_action_plan_handles_patient_as_time_scope(self):
        model = SimpleNamespace()
        with patch("src.hipergator.patient_search.generate", return_value=(
            '{"action":"unsupported","scope":"patient"}'
        )):
            plan = select_action_plan(model, Tokenizer(),
                                      "Calculate urine output rate during the latest encounter",
                                      ["measurement"], [])
        self.assertEqual(plan["action"], "unsupported")

        with patch("src.hipergator.patient_search.generate", return_value=(
            '{"action":"records","scope":"patient",'
            '"tables":["measurement"],"concept_ids":[]}'
        )):
            plan = select_action_plan(model, Tokenizer(),
                                      "Show measurements during the latest encounter",
                                      ["measurement"], [])
        self.assertEqual(plan["scope"], "latest_encounter")

    def test_unsupported_rate_request_returns_result(self):
        model = SimpleNamespace()
        with patch("src.hipergator.patient_search.generate", return_value=(
            '{"action":"unsupported","scope":"patient"}'
        )):
            result = run_patient_request(
                7, "Calculate urine output rate during the latest encounter",
                self.directory.name, model, Tokenizer(), {},
            )
        self.assertEqual(result["kind"], "unsupported")
        self.assertIsNone(result["analysis"])

    def test_action_route_returns_chart_ready_patient_data(self):
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=16384))
        with patch("src.hipergator.patient_search.generate", return_value=(
            '{"action":"time_series","scope":"latest_encounter",'
            '"tables":["measurement"],"concept_ids":[300]}'
        )):
            result = run_patient_request(7, "plot Example during latest encounter",
                                         self.directory.name, model, Tokenizer(),
                                         {300: "Example"})
        self.assertEqual(result["kind"], "time_series")
        self.assertEqual(result["analysis"]["source_rows"], 2)
        self.assertEqual(len(result["analysis"]["series"][0]["points"]), 2)

    def test_action_route_returns_filtered_source_records(self):
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=16384))
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"action":"records","scope":"history",'
            '"tables":["measurement"],"concept_ids":[300]}',
            "Here are the matching measurements.",
        ]):
            result = run_patient_request(7, "show Example measurements",
                                         self.directory.name, model, Tokenizer(),
                                         {300: "Example"})
        self.assertEqual(result["kind"], "records")
        measurements = next(item for item in result["evidence"] if item["table"] == "measurement")
        self.assertEqual(measurements["total_rows"], 3)
        self.assertTrue(all(row["measurement_concept_id"] == 300
                            for row in measurements["shown_rows"]))


if __name__ == "__main__":
    unittest.main()
