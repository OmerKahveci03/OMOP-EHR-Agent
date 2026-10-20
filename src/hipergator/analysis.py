"""Validated, patient-scoped measurement analyses for the search agent."""

from typing import Literal

import numpy as np
import pandas as pd

from .retrieval import json_scalar


AnalysisKind = Literal["time_series", "distribution", "statistics"]
ANALYSIS_KINDS: tuple[AnalysisKind, ...] = ("time_series", "distribution", "statistics")


def measurement_catalog(rows: pd.DataFrame, names: dict[int, str], request: str,
                        limit: int = 40) -> list[dict]:
    """Offer the model patient-specific measurement concepts, ranked by request text."""
    if rows.empty or "measurement_concept_id" not in rows:
        return []
    ids = pd.to_numeric(rows["measurement_concept_id"], errors="coerce").astype("Int64")
    catalog_rows = rows.assign(_concept_id=ids).dropna(subset=["_concept_id"])
    entries: list[dict] = []
    for concept_id, group in catalog_rows.groupby("_concept_id", sort=False):
        source_values = (
            group["measurement_source_value"].dropna().astype(str).drop_duplicates().head(3).tolist()
            if "measurement_source_value" in group else []
        )
        label = names.get(int(concept_id)) or (source_values[0] if source_values else str(concept_id))
        entries.append({"concept_id": int(concept_id), "name": label,
                        "source_values": source_values, "row_count": len(group)})

    query_words = {word for word in request.lower().replace("-", " ").split()
                   if len(word) > 2 and word not in {"the", "this", "patient", "their", "during", "latest", "encounter", "show", "over", "time", "measurements", "results", "across", "all"}}

    def rank(entry: dict) -> tuple[int, int]:
        searchable = " ".join([entry["name"], *entry["source_values"]]).lower()
        score = sum(3 if word in searchable.split() else 1 if word in searchable else 0
                    for word in query_words)
        return score, entry["row_count"]

    return sorted(entries, key=rank, reverse=True)[:limit]


def analyze_measurements(rows: pd.DataFrame, names: dict[int, str],
                         concept_ids: list[int], kind: AnalysisKind,
                         scope: str) -> dict:
    """Compute a chart-ready result without asking the model to do arithmetic."""
    if kind not in ANALYSIS_KINDS:
        raise ValueError(f"Unsupported analysis: {kind}")
    if rows.empty:
        return {"kind": kind, "table": "measurement", "scope": scope,
                "concept_ids": concept_ids, "source_rows": 0, "series": []}

    working = rows.copy()
    working["_concept_id"] = pd.to_numeric(
        working["measurement_concept_id"], errors="coerce"
    ).astype("Int64")
    if concept_ids:
        working = working[working["_concept_id"].isin(concept_ids)]
    working["_value"] = pd.to_numeric(working["value_as_number"], errors="coerce")
    working["_unit_id"] = pd.to_numeric(
        working.get("unit_concept_id", pd.Series(index=working.index, dtype="object")),
        errors="coerce"
    ).astype("Int64")
    working["_unit"] = working.get("unit_source_value", pd.Series(index=working.index, dtype="object"))
    working["_unit"] = working["_unit"].fillna("Unspecified").astype(str)
    date_field = "measurement_datetime" if "measurement_datetime" in working else "measurement_date"
    if kind == "time_series":
        working["_date"] = pd.to_datetime(working[date_field], errors="coerce")

    series: list[dict] = []
    for (concept_id, unit_id, unit), group in working.dropna(subset=["_concept_id"]).groupby(
        ["_concept_id", "_unit_id", "_unit"], sort=False, dropna=False
    ):
        numeric = group[np.isfinite(group["_value"].to_numpy(dtype=float, na_value=np.nan))]
        source_values = group["measurement_source_value"].dropna() if "measurement_source_value" in group else pd.Series(dtype="object")
        label = names.get(int(concept_id)) or (str(source_values.iloc[0]) if not source_values.empty else str(concept_id))
        item: dict = {"concept_id": int(concept_id), "label": label, "unit": unit,
                      "unit_concept_id": int(unit_id) if pd.notna(unit_id) else None,
                      "source_rows": len(group), "numeric_count": len(numeric),
                      "excluded_count": len(group) - len(numeric)}
        if kind == "time_series":
            dated = numeric.dropna(subset=["_date"]).sort_values("_date")
            item["excluded_count"] += len(numeric) - len(dated)
            item["points"] = [
                {"date": row["_date"].isoformat(), "value": float(row["_value"]),
                 "measurement_id": json_scalar("measurement_id", row["measurement_id"])}
                for _, row in dated.iterrows()
            ]
        elif kind == "distribution":
            values = numeric["_value"].to_numpy(dtype=float)
            if len(values):
                counts, edges = np.histogram(values, bins=min(10, max(1, int(np.sqrt(len(values))))))
                item["bins"] = [
                    {"start": float(edges[index]), "end": float(edges[index + 1]),
                     "count": int(count)}
                    for index, count in enumerate(counts)
                ]
            else:
                item["bins"] = []
        else:
            values = numeric["_value"]
            item["stats"] = {
                "count": len(values),
                "min": float(values.min()) if len(values) else None,
                "median": float(values.median()) if len(values) else None,
                "mean": float(values.mean()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
                "std": float(values.std()) if len(values) > 1 else None,
            }
        series.append(item)
    return {"kind": kind, "table": "measurement", "scope": scope,
            "concept_ids": concept_ids, "source_rows": len(working), "series": series}
