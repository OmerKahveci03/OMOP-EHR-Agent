"""Checks streaming merge of Athena and MIMIC concepts."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.hipergator.concepts import load_concept_names
from src.hipergator.preprocess import merge_concept_vocabularies, preprocess_data


class ConceptPreprocessTests(unittest.TestCase):
    def test_preprocess_writes_combined_concept_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "physionet.org" / "files" / "mimic-iv-demo-omop" / "0.9" / "1_omop_data_csv"
            raw.mkdir(parents=True)
            athena = root / "vocabulary_download_v5"
            athena.mkdir()
            (raw / "2b_concept.csv").write_text(
                "concept_id,concept_name\n2000000001,Local name\n", encoding="utf-8"
            )
            (raw / "person.csv").write_text(
                "person_id,year_of_birth\n7,1980\n", encoding="utf-8"
            )
            (athena / "CONCEPT.csv").write_text(
                "concept_id\tconcept_name\n456\tCreatinine\n", encoding="utf-8"
            )
            preprocess_data(base_dir=str(root), max_workers=1)
            names = pd.read_parquet(root / "cleaned_data" / "concept.parquet")
            self.assertEqual(set(names["concept_id"]), {456, 2000000001})
            self.assertTrue((root / "cleaned_data" / "person.parquet").exists())
            self.assertTrue((root / "cleaned_data" / "metadata" / "metadata_master.csv").exists())

    def test_merge_preserves_local_names_and_skips_unchanged_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cleaned = root / "cleaned_data"
            metadata = cleaned / "metadata"
            cleaned.mkdir()
            metadata.mkdir()
            local = root / "2b_concept.csv"
            athena = root / "CONCEPT.csv"
            local.write_text(
                "concept_id,concept_name,vocabulary_id\n"
                "2000000001,Local name,MIMIC\n"
                "123,Local override,MIMIC\n", encoding="utf-8"
            )
            athena.write_text(
                "concept_id\tconcept_name\tvocabulary_id\tdomain_id\n"
                "123\tAthena name\tLOINC\tMeasurement\n"
                "456\tCreatinine\tLOINC\tMeasurement\n", encoding="utf-8"
            )
            arguments = (str(local), str(athena), str(cleaned), str(metadata))
            self.assertTrue(merge_concept_vocabularies(*arguments))
            concepts = pd.read_parquet(cleaned / "concept.parquet")
            self.assertEqual(len(concepts), 3)
            names = dict(zip(concepts["concept_id"], concepts["concept_name"]))
            self.assertEqual(names[123], "Local override")
            self.assertEqual(names[456], "Creatinine")
            self.assertEqual(names[2000000001], "Local name")
            self.assertEqual(load_concept_names(str(cleaned))[123], "Local override")
            self.assertFalse(merge_concept_vocabularies(*arguments))
            self.assertTrue(merge_concept_vocabularies(*arguments, overwrite=True))


if __name__ == "__main__":
    unittest.main()
