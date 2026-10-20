"""Retrieve a patient's latest encounter and compact measurement evidence."""

import json
import os
from numbers import Integral, Real

from typing import Any, Literal, Optional
import pandas as pd

from .utils.types import ScopeType


MAX_SAFE_JSON_INTEGER = 2**53 - 1
# Tables with an encounter id
ENCOUNTER_LINKED_TABLES = {
    "visit_occurrence", "visit_detail", "condition_occurrence", "drug_exposure",
    "procedure_occurrence", "device_exposure", "measurement", "observation",
}


def json_scalar(field: str, value: Any) -> str | int | float | None:
    """Convert Pandas scalars and preserve IDs too large for JavaScript numbers."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Integral):
        number = int(value)
        return str(number) if field.endswith("_id") and abs(number) > MAX_SAFE_JSON_INTEGER else number
    if isinstance(value, Real):
        return float(value)
    return str(value)


def json_records(rows: pd.DataFrame) -> list[dict]:
    records: list[dict] = json.loads(rows.to_json(orient="records", date_format="iso"))
    for row in records:
        for field, value in row.items():
            if field.endswith("_id") and isinstance(value, int) and abs(value) > MAX_SAFE_JSON_INTEGER:
                row[field] = str(value)
    return records


def latest_encounter(data_dir: str, person_id: int) -> dict[str, Any] | None:
    """Return the visit with the latest recorded start time."""

    path: str = os.path.join(data_dir, "visit_occurrence.parquet")
    visits: pd.DataFrame = pd.read_parquet(path, filters=[("person_id", "==", person_id)])
    if visits.empty:
        return None

    start_col: str = ("visit_start_datetime" 
                      if "visit_start_datetime" in visits.columns
                      else "visit_start_date")

    visits = visits.sort_values(start_col, ascending=False, na_position="last").head(1)

    return json_records(rows=visits)[0]


def patient_table(data_dir: str, table: str, person_id: int,
                  scope: ScopeType, encounter: Optional[dict]) -> pd.DataFrame:
    """Read patient rows, limiting the default scope by the exact visit ID."""

    path: str = os.path.join(data_dir, f"{table}.parquet")
    filters: list[tuple[str, Literal["=="], int]] = [("person_id", "==", person_id)]

    if scope == "latest_encounter" and encounter is not None and table in ENCOUNTER_LINKED_TABLES:
        filters.append(("visit_occurrence_id", "==", int(encounter["visit_occurrence_id"])))

    rows: pd.DataFrame = pd.read_parquet(path, filters=filters).drop(columns=["person_id"])

    if scope == "history":
        return rows

    if encounter is None:
        return rows.iloc[0:0]

    if table in ENCOUNTER_LINKED_TABLES:
        return rows

    # Without an encounter link, a date overlap alone cannot establish that a
    # record belongs to this visit. Such tables require explicit history scope.
    return rows.iloc[0:0]


def summarize_measurements(rows: pd.DataFrame) -> dict:
    """Summarize numeric results by visit, measurement type, and unit."""
    source_rows: int = len(rows)
    if rows.empty:
        return {"table": "measurement_summary", "total_rows": 0,
                "source_rows": 0, "shown_rows": [], "aggregation": "by_visit_and_type"}

    date_field = "measurement_datetime" if "measurement_datetime" in rows else "measurement_date"
    rows = rows.sort_values(date_field, ascending=False, na_position="last")
    keys = [field for field in (
        "visit_occurrence_id", "measurement_concept_id", "measurement_source_value",
        "unit_concept_id", "unit_source_value",
    ) if field in rows.columns]
    shown_rows = []
    for _, group in rows.groupby(keys, dropna=False, sort=False):
        latest = group.iloc[0]
        base = {field: json_scalar(field, latest[field]) for field in keys}
        dates = group[date_field].dropna()
        base.update({
            "first_date": json_scalar("first_date", dates.min()) if not dates.empty else None,
            "last_date": json_scalar("last_date", dates.max()) if not dates.empty else None,
        })
        numbers = pd.to_numeric(group["value_as_number"], errors="coerce")
        numeric = group.loc[numbers.notna()]
        if not numeric.empty:
            values = numbers.loc[numbers.notna()]
            latest_numeric = numeric.iloc[0]
            shown_rows.append({
                **base, "result_type": "numeric", "count": len(values),
                "min": float(values.min()), "mean": float(values.mean()),
                "max": float(values.max()),
                "std": float(values.std()) if len(values) > 1 else None,
                "latest_value": float(latest_numeric["value_as_number"]),
                "latest_measurement_id": json_scalar("latest_measurement_id", latest_numeric["measurement_id"]),
            })
        categorical = group.loc[numbers.isna()]
        if not categorical.empty:
            category_keys = [field for field in ("value_as_concept_id", "value_source_value")
                             if field in categorical.columns]
            for _, category in categorical.groupby(category_keys, dropna=False, sort=False):
                latest_category = category.iloc[0]
                category_dates = category[date_field].dropna()
                shown_rows.append({
                    **base, "result_type": "non_numeric", "count": len(category),
                    "first_date": json_scalar("first_date", category_dates.min()) if not category_dates.empty else None,
                    "last_date": json_scalar("last_date", category_dates.max()) if not category_dates.empty else None,
                    **{field: json_scalar(field, latest_category[field]) for field in category_keys},
                    "latest_measurement_id": json_scalar("latest_measurement_id", latest_category["measurement_id"]),
                })

    for index, row in enumerate(shown_rows, start=1):
        row["summary_id"] = f"group-{index}"
    return {"table": "measurement_summary", "total_rows": len(shown_rows),
            "source_rows": source_rows, "shown_rows": shown_rows,
            "aggregation": "by_visit_and_type"}
