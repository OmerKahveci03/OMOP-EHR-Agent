"""Checks for encounter scoping and measurement aggregation."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.hipergator.concepts import load_measurement_ancestors
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
        pd.DataFrame([
            {"ancestor_concept_id": 100, "descendant_concept_id": 300},
            {"ancestor_concept_id": 100, "descendant_concept_id": 301},
        ]).to_parquet(root / "measurement_concept_ancestor.parquet")

    def test_missing_measurement_ancestor_index_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Run python -u -m src.hipergator.preprocess"):
                load_measurement_ancestors(directory)

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
            plan = select_action_plan(model, Tokenizer(), "trend", ["measurement"],
                                      [{"concept_id": 300, "name": "Example"}])
        self.assertEqual(plan["action"], "unsupported")

    def test_action_plan_accepts_integer_strings_and_many_concepts(self):
        model = SimpleNamespace()
        candidates = [{"concept_id": concept_id, "name": str(concept_id)}
                      for concept_id in range(300, 310)]
        with patch("src.hipergator.patient_search.generate", return_value=(
            '{"action":"statistics","scope":"history",'
            '"tables":["measurement"],"concept_ids":["300","301"]}'
        )):
            plan = select_action_plan(model, Tokenizer(), "statistics", ["measurement"], candidates)
        self.assertEqual(plan["concept_ids"], [300, 301])
        with patch("src.hipergator.patient_search.generate", return_value=(
            '{"action":"statistics","scope":"history",'
            '"tables":["measurement"],"concept_ids":'
            '["300","301","302","303","304","305","306","307","308","309"]}'
        )):
            plan = select_action_plan(model, Tokenizer(), "statistics", ["measurement"], candidates)
        self.assertEqual(plan["concept_ids"], list(range(300, 310)))

    def test_generic_analysis_plan_asks_model_for_specific_kind(self):
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"action":"analysis","scope":"latest_encounter",'
            '"tables":["measurement"],"concept_ids":[300]}',
            '{"action":"statistics"}',
        ]) as generate:
            plan = select_action_plan(SimpleNamespace(), Tokenizer(),
                                      "Count and median glucose measurements",
                                      ["measurement"], [{"concept_id": 300, "name": "Glucose"}])
        self.assertEqual(plan["action"], "statistics")
        self.assertEqual(plan["concept_ids"], [300])
        self.assertEqual(generate.call_count, 2)

    def test_generic_analysis_route_uses_selected_statistics_kind(self):
        question = ("For the latest encounter, what are the count, minimum, median, "
                    "maximum, and standard deviation of this patient's glucose measurements?")
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"action":"analysis","scope":"latest_encounter"}',
            '{"action":"statistics","scope":"latest_encounter",'
            '"tables":["measurement"],"concept_ids":[300]}',
        ]) as generate:
            result = run_patient_request(7, question, self.directory.name,
                                         SimpleNamespace(), Tokenizer(), {300: "Glucose"})
        self.assertEqual(result["kind"], "statistics")
        self.assertEqual(result["analysis"]["series"][0]["stats"]["count"], 2)
        self.assertEqual(result["concept_selection"]["selected_concepts"][0]["concept_id"], 300)
        self.assertIn("Action: statistics", result["decision_trace"][2]["detail"])
        self.assertEqual(generate.call_count, 2)

    def test_action_plan_expands_only_patient_descendants_of_selected_ancestor(self):
        ancestor_options = [{"ancestor_id": 100, "name": "Example family",
                             "patient_descendant_count": 2,
                             "descendant_ids": [300, 301]}]
        with patch("src.hipergator.patient_search.generate", return_value=(
            '{"action":"statistics","scope":"history",'
            '"tables":["measurement"],"concept_ids":[],"ancestor_ids":["100"]}'
        )):
            plan = select_action_plan(SimpleNamespace(), Tokenizer(), "Example family stats",
                                      ["measurement"], [], ancestor_options)
        self.assertEqual(plan["concept_ids"], [300, 301])
        self.assertEqual(plan["ancestor_ids"], [100])
        self.assertEqual(plan["direct_concept_ids"], [])

    def test_action_route_uses_ancestor_catalog(self):
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"action":"statistics","scope":"latest_encounter",'
            '"tables":["measurement"]}',
            '{"action":"statistics","scope":"latest_encounter",'
            '"tables":["measurement"],"concept_ids":[],"ancestor_ids":[100]}'
        ]) as generate:
            result = run_patient_request(
                7, "Example family statistics", self.directory.name,
                SimpleNamespace(), Tokenizer(), {100: "Example family", 300: "Example"},
            )
        self.assertEqual(result["analysis"]["concept_ids"], [300, 301])
        self.assertEqual(result["analysis"]["ancestor_ids"], [100])
        selection = result["concept_selection"]
        self.assertEqual(selection["selection_mode"], "selected")
        self.assertEqual([item["concept_id"] for item in selection["selected_concepts"]], [300, 301])
        self.assertTrue(all(item["source"] == "ancestor" for item in selection["selected_concepts"]))
        self.assertEqual(selection["selected_ancestors"][0]["ancestor_id"], 100)
        self.assertIn('"ancestor_id": 100', generate.call_args.kwargs["user"])

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

    def test_visit_list_skips_measurement_catalog_and_uses_history(self):
        question = "Show me this patient's visits, with the most recent visit first."
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"action":"records","scope":"latest_encounter",'
            '"tables":["visit_occurrence"]}',
            "Two visits are recorded [visit_occurrence:10] [visit_occurrence:11].",
        ]) as generate, patch(
            "src.hipergator.patient_search.load_measurement_ancestors"
        ) as load_ancestors:
            result = run_patient_request(
                7, question, self.directory.name, SimpleNamespace(), Tokenizer(), {},
            )
        self.assertEqual(generate.call_count, 2)
        load_ancestors.assert_not_called()
        self.assertEqual(result["evidence"][1]["total_rows"], 2)
        self.assertEqual(result["evidence"][1]["retrieval_scope"], "history")
        self.assertNotIn("Candidates", [step["stage"] for step in result["decision_trace"]])
        self.assertIsNone(result["concept_selection"])

    def test_condition_history_supplies_complete_summary_for_generated_answer(self):
        root = Path(self.directory.name)
        pd.DataFrame([
            {"person_id": 7, "condition_occurrence_id": index,
             "visit_occurrence_id": 10 if index == 0 else 11,
             "condition_concept_id": 400 if index < 100 else 401,
             "condition_source_value": "Source condition",
             "condition_start_date": "2200-01-01" if index == 0 else "2201-08-12"}
            for index in range(101)
        ]).to_parquet(root / "condition_occurrence.parquet")
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"action":"records","scope":"history",'
            '"tables":["condition_occurrence"]}',
            "Condition A appears in the record from 2200-01-01 to 2201-08-12 "
            "[condition_summary:1]. Condition B also appears [condition_summary:2].",
        ]) as generate:
            result = run_patient_request(
                7, "Which conditions appear and when?", self.directory.name,
                SimpleNamespace(), Tokenizer(), {400: "Condition A", 401: "Condition B"},
            )
        self.assertEqual(generate.call_count, 2)
        self.assertIn("Condition A appears", result["answer"])
        raw = next(item for item in result["evidence"] if item["table"] == "condition_occurrence")
        summary = next(item for item in result["evidence"] if item["table"] == "condition_summary")
        self.assertEqual(len(raw["shown_rows"]), 100)
        self.assertEqual(summary["source_rows"], 101)
        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(summary["shown_rows"][0]["first_date"], "2200-01-01")
        self.assertEqual(summary["shown_rows"][0]["last_date"], "2201-08-12")
        self.assertIn("condition_summary", generate.call_args.kwargs["user"])

    def test_record_answer_rejects_number_missing_from_evidence(self):
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"action":"records","scope":"history","tables":["visit_occurrence"]}',
            "An invented value of 64.2 is recorded [visit_occurrence:10].",
        ]):
            result = run_patient_request(
                7, "Show visits", self.directory.name,
                SimpleNamespace(), Tokenizer(), {},
            )
        self.assertIn("could not be verified", result["answer"])

    def test_rate_calculation_cannot_fall_through_to_generated_record_answer(self):
        with patch("src.hipergator.patient_search.generate", return_value=(
            '{"action":"records","scope":"latest_encounter",'
            '"tables":["measurement"]}'
        )) as generate:
            result = run_patient_request(
                7, "Calculate this patient's urine output rate during the latest encounter",
                self.directory.name, SimpleNamespace(), Tokenizer(), {300: "Creatinine"},
            )
        self.assertEqual(result["kind"], "unsupported")
        self.assertIn("No rate or other value was estimated", result["answer"])
        self.assertEqual(generate.call_count, 1)
        self.assertEqual([step["stage"] for step in result["decision_trace"]],
                         ["Model route", "Validation"])

    def test_action_route_returns_chart_ready_patient_data(self):
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=16384))
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"action":"time_series","scope":"latest_encounter",'
            '"tables":["measurement"]}',
            '{"action":"time_series","scope":"latest_encounter",'
            '"tables":["measurement"],"concept_ids":[300]}'
        ]):
            result = run_patient_request(7, "show Example over time during latest encounter",
                                         self.directory.name, model, Tokenizer(),
                                         {300: "Example"})
        self.assertEqual(result["kind"], "time_series")
        self.assertEqual(result["analysis"]["source_rows"], 2)
        self.assertEqual(len(result["analysis"]["series"][0]["points"]), 2)
        selection = result["concept_selection"]
        self.assertEqual(selection["selected_concepts"], [
            {"concept_id": 300, "name": "Example", "row_count": 3,
             "scope_row_count": 2, "source": "direct"}
        ])
        self.assertEqual({item["concept_id"] for item in selection["candidates"]}, {300, 301})

    def test_selected_history_only_concept_is_explained_when_latest_scope_has_no_rows(self):
        path = Path(self.directory.name) / "measurement.parquet"
        rows = pd.read_parquet(path)
        history_only = rows.iloc[0].copy()
        history_only["measurement_id"] = 5
        history_only["measurement_concept_id"] = 302
        history_only["measurement_source_value"] = "Invasive blood pressure"
        rows = pd.concat([rows, history_only.to_frame().T], ignore_index=True)
        rows.to_parquet(path)
        pd.DataFrame([{"person_id": 7, "observation_id": 6}]).to_parquet(
            Path(self.directory.name) / "observation.parquet")
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"action":"time_series","scope":"latest_encounter",'
            '"tables":["measurement","observation"]}',
            '{"action":"time_series","scope":"latest_encounter",'
            '"tables":["measurement"],"concept_ids":[300,302]}',
        ]):
            result = run_patient_request(
                7, "Show invasive and noninvasive blood pressure over time in the latest encounter",
                self.directory.name, SimpleNamespace(), Tokenizer(),
                {300: "Noninvasive blood pressure", 302: "Invasive blood pressure"},
            )
        self.assertEqual(result["analysis"]["source_rows"], 2)
        self.assertEqual(len(result["analysis"]["series"]), 1)
        selected = {item["concept_id"]: item for item in result["concept_selection"]["selected_concepts"]}
        self.assertEqual(selected[302]["row_count"], 1)
        self.assertEqual(selected[302]["scope_row_count"], 0)
        self.assertEqual(selected[300]["scope_row_count"], 2)
        self.assertIn("1 selected concept had no measurement records in the latest encounter", result["answer"])
        self.assertTrue(any("observation was not analyzed" in step["detail"]
                            for step in result["decision_trace"]))

    def test_action_route_returns_filtered_source_records(self):
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=16384))
        with patch("src.hipergator.patient_search.generate", side_effect=[
            '{"action":"records","scope":"history","tables":["measurement"]}',
            '{"action":"records","scope":"history",'
            '"tables":["measurement"],"concept_ids":[300]}',
            "Example measurements are recorded [measurement:1].",
        ]) as generate:
            result = run_patient_request(7, "show Example measurements",
                                         self.directory.name, model, Tokenizer(),
                                         {300: "Example"})
        self.assertEqual(result["kind"], "records")
        self.assertEqual(generate.call_count, 3)
        self.assertIn("Example measurements", result["answer"])
        self.assertEqual(result["decision_trace"][-1]["stage"], "Answer")
        measurements = next(item for item in result["evidence"] if item["table"] == "measurement")
        self.assertEqual(measurements["total_rows"], 3)
        self.assertTrue(all(row["measurement_concept_id"] == 300
                            for row in measurements["shown_rows"]))


if __name__ == "__main__":
    unittest.main()
