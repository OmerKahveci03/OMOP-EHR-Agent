"""Resolve OMOP concept IDs in evidence for the model and user interface."""

import os
from functools import lru_cache

import pandas as pd


def load_concept_names(data_dir: str) -> dict[int, str]:
    """Load only the ID and name columns from the cleaned concept table."""
    path: str = os.path.join(data_dir, "concept.parquet")
    concepts: pd.DataFrame = pd.read_parquet(path, columns=["concept_id", "concept_name"])
    return {
        int(concept_id): str(name)
        for concept_id, name in concepts.itertuples(index=False, name=None)
        if pd.notna(concept_id) and pd.notna(name)
    }


@lru_cache(maxsize=1)
def load_measurement_ancestors(data_dir: str) -> dict[int, set[int]]:
    """Load the compact hierarchy index built from recorded measurements."""
    path = os.path.join(data_dir, "measurement_concept_ancestor.parquet")
    links = pd.read_parquet(path, columns=["ancestor_concept_id", "descendant_concept_id"])
    return {
        int(ancestor_id): set(group["descendant_concept_id"].astype(int))
        for ancestor_id, group in links.groupby("ancestor_concept_id")
    }


def add_concept_names(evidence: list[dict], names: dict[int, str]) -> list[dict]:
    """Add a name beside every concept ID without changing the original evidence."""
    enriched: list[dict] = []
    for item in evidence:
        rows: list[dict] = []
        for row in item["shown_rows"]:
            result: dict = {}
            for field, value in row.items():
                result[field] = value
                if field.endswith("_concept_id"):
                    result[f"{field}_name"] = names.get(value) if value is not None else None
            rows.append(result)
        enriched.append({**item, "shown_rows": rows})
    return enriched
