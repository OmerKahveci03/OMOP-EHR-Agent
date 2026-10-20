import argparse
import json
import re
import os

import torch

import pandas as pd

from typing import Any, Optional, cast

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from . import LLAMA_PATH
from .analysis import (ANALYSIS_KINDS, analyze_measurements,
                       measurement_ancestor_catalog, measurement_catalog)
from .concepts import add_concept_names, load_concept_names, load_measurement_ancestors
from .retrieval import json_records, latest_encounter, patient_table, summarize_measurements
from .utils.paths import get_path
from .utils.types import RowsKeys, ScopeType, SCOPE_VALUES

CLINICAL_TABLES: tuple[str] = (
    "person", "visit_occurrence", "visit_detail", "condition_occurrence",
    "drug_exposure", "procedure_occurrence", "device_exposure", "measurement",
    "observation", "death", "specimen", "observation_period",
)


def generate(model: PreTrainedModel, 
             tokenizer: PreTrainedTokenizerBase, 
             system: str, 
             user: str, 
             max_new_tokens: int) -> str:
    """ Generate a result for the given prompt using the model. """

    # role & content
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system}, 
        {"role": "user", "content": user}
    ]

    inputs: BatchEncoding = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        output: torch.Tensor = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], 
                            skip_special_tokens=True).strip()


def select_retrieval_plan(model: PreTrainedModel,
                          tokenizer: PreTrainedTokenizerBase,
                          request: str,
                          available: list[str]) -> tuple[list[str], ScopeType]:
    """Select clinical tables and whether earlier encounters are needed."""

    # --- Prompt the Model ---
    response: str = generate(
        model=model,
        tokenizer=tokenizer,
        system=(
            "Select OMOP tables relevant to the question and its time scope. "
            "Return only JSON such as "
            "{\"tables\":[\"measurement\"],\"scope\":\"latest_encounter\"}. "
            "Use at most five table names from the supplied list. "
            "Set scope to 'latest_encounter' by default. Set it to 'history' only "
            "when the question asks about earlier visits, trends, ever/any-time "
            "events, or records that may not belong to the latest encounter. "
            "Never generate SQL or Python."
        ),
        user=f"Available tables: {json.dumps(available)}\nQuestion: {request}",
        max_new_tokens=160,
    )

    # --- Get Model Response ---
    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        raise ValueError(f"Model did not return a retrieval plan: {response}")

    plan = json.loads(match.group())

    names = plan.get("tables") if isinstance(plan, dict) else None
    scope = plan.get("scope") if isinstance(plan, dict) else None
    if not isinstance(names, list) or any(not isinstance(name, str) or name not in available for name in names):
        raise ValueError(f"Model selected an unavailable table: {names}")
    if scope not in SCOPE_VALUES:
        raise ValueError(f"Model selected an invalid time scope: {scope}")
    return list(dict.fromkeys(names))[:5], cast(ScopeType, scope)


def select_route_plan(model: PreTrainedModel,
                      tokenizer: PreTrainedTokenizerBase,
                      request: str, available: list[str]) -> dict:
    """Choose the operation and tables before loading measurement candidates."""
    response = generate(
        model=model, tokenizer=tokenizer,
        system=(
            "You navigate one patient's OMOP data. Return only JSON with keys "
            "action, scope, tables. Actions: records, time_series, distribution, "
            "statistics, unsupported. Use records to show source rows; use an "
            "analysis action for numeric measurement summaries. Choose unsupported "
            "for calculations requiring derived rates or unit conversions. Scope "
            "must be latest_encounter or history. A request for visits or encounters "
            "in plural needs history, even if sorted most recent first. Choose at "
            "most five tables from the available list. Analysis actions require "
            "measurement. Do not select measurement for unrelated questions."
        ),
        user=f"Available tables: {json.dumps(available)}\nQuestion: {request}",
        max_new_tokens=128,
    )
    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        raise ValueError(f"Model did not return a route plan: {response}")
    plan = json.loads(match.group())
    action = plan.get("action")
    if action not in ("records", *ANALYSIS_KINDS, "unsupported"):
        raise ValueError(f"Unsupported action: {action}")
    if action == "unsupported":
        return {"action": action, "scope": "latest_encounter", "tables": []}
    tables = plan.get("tables")
    if not isinstance(tables, list) or len(tables) > 5 or any(
        not isinstance(table, str) or table not in available for table in tables
    ):
        raise ValueError(f"Unavailable tables in route plan: {tables}")
    scope = plan.get("scope")
    if re.search(r"\b(visits|encounters|history|earlier|previous)\b", request, re.I):
        scope = "history"
    elif scope not in SCOPE_VALUES:
        scope = "latest_encounter"
    if action in ANALYSIS_KINDS and "measurement" not in tables:
        raise ValueError("Analysis action requires the measurement table")
    return {"action": action, "scope": scope,
            "tables": list(dict.fromkeys(tables))}


def select_action_plan(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizerBase,
                       request: str, available: list[str],
                       candidates: list[dict],
                       ancestor_candidates: list[dict] | None = None) -> dict:
    """Choose a validated patient-scoped retrieval or measurement operation."""
    ancestor_choices = [
        {"ancestor_id": item["ancestor_id"], "name": item["name"],
         "patient_descendant_count": item["patient_descendant_count"]}
        for item in (ancestor_candidates or [])
    ]
    response = generate(
        model=model, tokenizer=tokenizer,
        system=(
            "You navigate one patient's OMOP data. Return only JSON with keys "
            "action, scope, tables, concept_ids, ancestor_ids. Actions: records, time_series, "
            "distribution, statistics, unsupported. Use records to show source rows; "
            "use an analysis action only for numeric measurement requests. "
            "The analysis actions calculate values directly from recorded measurements; "
            "they cannot derive rates or perform unit conversions. Choose unsupported "
            "when the requested calculation needs either. Choose concept_ids only from "
            "the supplied patient-specific candidates. Leave both concept_ids and "
            "ancestor_ids empty only when the user explicitly asks about all "
            "measurements; for a named measurement with no matching candidate "
            "or group choose unsupported. You may also select "
            "ancestor_ids from the supplied patient-specific ancestor groups to "
            "include their recorded descendants. Use a group only when all its "
            "descendants fit the question. Select all and only "
            "the candidates relevant to the question, regardless of how many "
            "there are. Write concept IDs as JSON integers, not strings. "
            "Scope means time only: use exactly latest_encounter or history, never "
            "patient or person. Choose latest_encounter unless the question asks for history, earlier "
            "encounters, or a comparison across encounters. Use history for a list "
            "of visits. Choose tables only from the supplied list, at most five. "
            "Analysis actions use the measurement table. Never generate SQL or Python."
        ),
        user=f"Available tables: {json.dumps(available)}\n"
             f"Measurement candidates: {json.dumps(candidates)}\n"
             f"Ancestor groups: {json.dumps(ancestor_choices)}\nQuestion: {request}",
        max_new_tokens=256,
    )
    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        raise ValueError(f"Model did not return an action plan: {response}")
    plan = json.loads(match.group())
    if not isinstance(plan, dict):
        raise ValueError("Action plan must be a JSON object")
    action = plan.get("action")
    scope = plan.get("scope")
    tables = plan.get("tables")
    concept_ids = plan.get("concept_ids")
    ancestor_ids = plan.get("ancestor_ids", [])
    if action not in ("records", *ANALYSIS_KINDS, "unsupported"):
        raise ValueError(f"Unsupported action: {action}")
    if action == "unsupported":
        return {"action": "unsupported", "scope": "latest_encounter",
                "tables": [], "concept_ids": []}
    if scope not in SCOPE_VALUES:
        scope = (
            "history" if re.search(r"\b(history|earlier|previous|ever|all (?:encounters|visits)|across (?:all )?(?:encounters|visits))\b",
                                   request, flags=re.IGNORECASE)
            else "latest_encounter"
        )
    if not isinstance(tables, list) or len(tables) > 5 or any(
        not isinstance(table, str) or table not in available for table in tables
    ):
        raise ValueError(f"Unavailable tables in action plan: {tables}")
    candidate_ids = {item["concept_id"] for item in candidates}
    if not isinstance(concept_ids, list):
        concept_ids = []
    parsed_ids: list[int] = []
    for concept_id in concept_ids:
        if isinstance(concept_id, bool) or not (
            isinstance(concept_id, int) or
            isinstance(concept_id, str) and re.fullmatch(r"-?\d+", concept_id.strip())
        ):
            return {"action": "unsupported", "reason": "Could not identify the requested measurement concepts."}
        parsed_id = int(concept_id)
        if parsed_id not in candidate_ids:
            return {"action": "unsupported", "reason": "Could not identify the requested measurement concepts."}
        parsed_ids.append(parsed_id)
    parsed_ids = list(dict.fromkeys(parsed_ids))
    direct_concept_ids = parsed_ids.copy()
    if not isinstance(ancestor_ids, list):
        ancestor_ids = []
    ancestor_options = {item["ancestor_id"]: item["descendant_ids"]
                        for item in (ancestor_candidates or [])}
    selected_ancestors: list[int] = []
    for ancestor_id in ancestor_ids:
        if isinstance(ancestor_id, bool) or not (
            isinstance(ancestor_id, int) or
            isinstance(ancestor_id, str) and re.fullmatch(r"-?\d+", ancestor_id.strip())
        ) or int(ancestor_id) not in ancestor_options:
            return {"action": "unsupported", "reason":
                    "Could not identify the requested measurement ancestor group."}
        selected_ancestors.append(int(ancestor_id))
        parsed_ids.extend(ancestor_options[int(ancestor_id)])
    parsed_ids = list(dict.fromkeys(parsed_ids))
    if action in ANALYSIS_KINDS and "measurement" not in available:
        raise ValueError("Measurement table is unavailable")
    return {"action": action, "scope": scope,
            "tables": list(dict.fromkeys(tables)),
            "concept_ids": parsed_ids,
            "direct_concept_ids": direct_concept_ids,
            "ancestor_ids": list(dict.fromkeys(selected_ancestors))}


def patient_rows(data_dir: str, table: str, 
                 person_id: int, limit: int = 100,
                 scope: ScopeType = "history", 
                 encounter: dict | None = None) -> dict[RowsKeys, str | int | list[dict]]:
    """ Load only rows belonging to the patient from the requested table. 
    
    Returns result as dict with the keys: `table`, `total_rows`, `shown_rows`."""

    df: pd.DataFrame = patient_table(data_dir=data_dir, table=table, 
                                     person_id=person_id, scope=scope, 
                                     encounter=encounter)

    # --- Sort by date if possible ---
    date_columns: list[str] = [
        col for col in df.columns 
        if col.endswith("_date") or col.endswith("_datetime")
    ]

    if date_columns:
        df = df.sort_values(date_columns[0], ascending=False, na_position="last")

    rows: list[dict] = json_records(rows=df.head(limit))
    return {"table": table, "total_rows": len(df), "shown_rows": rows}


def fit_evidence(evidence: list[dict], 
                 tokenizer: PreTrainedTokenizerBase, 
                 max_tokens: int) -> list[dict]:
    """Keep the model input within its context window, across all tables."""

    fitted: list[dict] = [{**item, "shown_rows": []} for item in evidence]

    for row_index in range(max(len(item["shown_rows"]) for item in evidence)):
        for target, source in zip(fitted, evidence):
            if row_index >= len(source["shown_rows"]):
                continue

            target["shown_rows"].append(source["shown_rows"][row_index])

            if len(tokenizer.encode(json.dumps(fitted))) > max_tokens:
                target["shown_rows"].pop()

    return fitted


def load_model(model_path: str = LLAMA_PATH) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the local model and tokenizer. """
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path, local_files_only=True
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        device_map="auto", torch_dtype="auto", local_files_only=True
    )
    model.eval()
    return model, tokenizer


def search_patient(person_id: int, request: str,
                   data_dir: str,
                   model: PreTrainedModel,
                   tokenizer: PreTrainedTokenizerBase,
                   concept_names: dict[int, str],
                   retrieval_plan: tuple[list[str], ScopeType] | None = None,
                   measurement_concept_ids: list[int] | None = None,
                   include_measurement_rows: bool = False,
                   generate_answer: bool = True) -> tuple[str, list[dict]]:
    """ Given a patient and natural language request, return EHR. """

    # --- Sanity Check ---
    required_tables: tuple[str] = ("person", "visit_occurrence")
    for required_table in required_tables:
        path: str = os.path.join(data_dir, f"{required_table}.parquet")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required OMOP table: {path}; run preprocessing first")

    # --- Pull Patient Metadata ---
    # Keys are `table`, `total_rows`, `shown_rows` 
    person: dict[RowsKeys, Any] = patient_rows(data_dir=data_dir, 
                                               table="person", 
                                               person_id=person_id, 
                                               limit=1)

    if not person["shown_rows"]:
        raise ValueError(f"person_id {person_id} was not found in PERSON")

    # --- Select Tables Based on What is Available ---
    available: list[str] = [table for table in CLINICAL_TABLES
                            if table not in required_tables and
                            os.path.isfile(os.path.join(data_dir, f"{table}.parquet"))]

    # NOTE: By default we focus on the latest encounter's EHR
    selected, scope = retrieval_plan or select_retrieval_plan(
        model=model, tokenizer=tokenizer, request=request, available=available
    )

    # --- Optionally Use Latest Encounter ---
    encounter: Optional[dict] = (
        latest_encounter(data_dir=data_dir, person_id=person_id)
        if scope == "latest_encounter" 
        else None
    )

    if scope == "latest_encounter" and encounter is None:
        raise ValueError(f"person_id {person_id} has no VISIT_OCCURRENCE record")

    # --- Pull the Encounters Involved in the Scope ---
    visits: dict[RowsKeys, Any] = patient_rows(data_dir=data_dir, 
                                               table="visit_occurrence", 
                                               person_id=person_id,
                                               scope=scope, 
                                               encounter=encounter)

    evidence: list[dict] = [person, visits]
    for name in selected:
        if name == "measurement":
            measurements: pd.DataFrame = patient_table(data_dir=data_dir, 
                                                       table=name, 
                                                       person_id=person_id, 
                                                       scope=scope, 
                                                       encounter=encounter)
            if measurement_concept_ids:
                ids = pd.to_numeric(measurements["measurement_concept_id"], errors="coerce")
                measurements = measurements[ids.isin(measurement_concept_ids)]
            if include_measurement_rows:
                evidence.append({"table": "measurement", "total_rows": len(measurements),
                                 "shown_rows": json_records(measurements.head(100))})
            else:
                evidence.append(summarize_measurements(rows=measurements))
        else:
            evidence.append(patient_rows(data_dir=data_dir, table=name, 
                                         person_id=person_id, scope=scope, 
                                         encounter=encounter))

    for item in evidence:
        item["retrieval_scope"] = scope

    # The model and the frontend should see the same concept names. Enrich
    # before fitting so the names count toward the model's context limit.
    evidence: list[dict] = add_concept_names(evidence=evidence, names=concept_names)

    if not generate_answer:
        counts = ", ".join(f"{item['table']}: {item['total_rows']}"
                           for item in evidence)
        return f"Retrieved patient records ({counts}). Review the source rows below.", evidence

    context_limit: int = min(5000, model.config.max_position_embeddings - 1024)
    evidence: list[dict] = fit_evidence(evidence=evidence, 
                                        tokenizer=tokenizer, 
                                        max_tokens=context_limit)

    answer: str = generate(
        model=model,
        tokenizer=tokenizer,
        system="Answer the user's question using only the supplied OMOP rows. "
        "Cite supporting records as [table:record_id], using the actual table and "
        "record ID field shown in the data. Cite aggregate measurement statistics "
        "as [measurement_summary:summary_id]; a summary ID identifies a computed "
        "group rather than an individual source record. "
        "If the rows do not support an answer, "
        "say so. Do not infer a diagnosis or treatment from incomplete records. "
        "Dates in this de-identified dataset are shifted, sometimes by more than "
        "a century. Keep dates as recorded, but do not describe them as future events.",
        user=f"Question: {request}\nPerson ID: {person_id}\n"
        f"Retrieval scope: {scope}. "
        "Measurement summaries, if present, group numeric values by visit, type, and unit. "
        "Do not claim to have seen rows beyond the supplied evidence.\n"
        f"Evidence (some tables may be truncated): {json.dumps(evidence)}",
        max_new_tokens=512,
    )
    return answer, evidence


def requires_derived_calculation(request: str) -> bool:
    """Block requests for calculations the measurement tools cannot perform."""
    return bool(
        re.search(r"\b(?:calculat\w*|comput\w*|deriv\w*|estimat\w*)\b", request, re.I)
        and re.search(r"\b(?:rate|ratio|clearance|conversion|percent(?:age)? change|difference)\b",
                      request, re.I)
    )


def summarize_condition_history(rows: pd.DataFrame,
                                concept_names: dict[int, str], scope: str) -> dict:
    """Provide complete condition counts and date ranges as evidence."""
    groups: dict[tuple[int | None, str], dict] = {}
    for _, row in rows.iterrows():
        raw_id = row.get("condition_concept_id")
        concept_id = int(raw_id) if pd.notna(raw_id) else None
        source = row.get("condition_source_value")
        label = (concept_names.get(concept_id) if concept_id else None) or (
            str(source) if pd.notna(source) and str(source).strip() else
            f"Concept {concept_id}" if concept_id else "Unnamed condition"
        )
        key = (concept_id, label)
        item = groups.setdefault(key, {"condition_concept_id": concept_id,
                                       "condition_name": label, "count": 0,
                                       "first_date": None, "last_date": None})
        item["count"] += 1
        date = row.get("condition_start_datetime")
        if pd.isna(date):
            date = row.get("condition_start_date")
        if pd.notna(date):
            recorded = str(date)[:10]
            item["first_date"] = min(item["first_date"] or recorded, recorded)
            item["last_date"] = max(item["last_date"] or recorded, recorded)
    summaries = sorted(groups.values(), key=lambda item: (-item["count"], item["condition_name"]))
    for index, item in enumerate(summaries, start=1):
        item["summary_id"] = index
    return {"table": "condition_summary", "source_table": "condition_occurrence",
                "aggregation": "conditions",
                "source_rows": len(rows), "total_rows": len(summaries),
                "shown_rows": summaries, "retrieval_scope": scope}


def generate_record_answer(model: PreTrainedModel,
                           tokenizer: PreTrainedTokenizerBase,
                           request: str, person_id: int, scope: str,
                           evidence: list[dict]) -> str:
    """Verbalize retrieved facts, accepting only references to returned evidence."""
    summary_tables = {item["source_table"] for item in evidence
                      if item.get("aggregation") and item.get("source_table")}
    model_evidence = [
        {**item, "shown_rows": item["shown_rows"][:5]}
        if item["table"] in summary_tables else item
        for item in evidence
    ]
    model_evidence.sort(key=lambda item: not bool(item.get("aggregation")))
    context_limit = min(5000, getattr(getattr(model, "config", None),
                                      "max_position_embeddings", 8192) - 1024)
    fitted = fit_evidence(model_evidence, tokenizer, context_limit)
    answer = generate(
        model=model, tokenizer=tokenizer,
        system=(
            "You are a patient data navigation assistant. Answer in natural language "
            "using only the supplied record facts. Describe recorded names, values, "
            "dates, counts, and retrieval limits. Do not infer diagnoses, physiology, "
            "or missing values. Do not calculate or estimate new clinical quantities. "
            "For each factual claim cite a supplied row as [table:record_id] or a "
            "supplied summary as [table:summary_id]. Use only IDs actually shown. "
            "If the evidence does not answer the question, say so. Dates are "
            "de-identified; do not call them future events."
        ),
        user=f"Question: {request}\nPerson ID: {person_id}\nScope: {scope}\n"
             f"Evidence: {json.dumps(fitted)}",
        max_new_tokens=512,
    )
    allowed = {
        (item["table"], str(row[id_field]))
        for item in fitted
        for row in item["shown_rows"]
        for id_field in ("summary_id" if item.get("aggregation") else
                         f"{item['table']}_id",)
        if row.get(id_field) is not None
    }
    citations = re.findall(r"\[([a-z_]+):([^\]]+)\]", answer)
    evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", json.dumps(fitted)))
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", answer))
    if (not citations or any(citation not in allowed for citation in citations)
            or not answer_numbers.issubset(evidence_numbers)):
        counts = ", ".join(f"{item['table']}: {item['total_rows']}"
                           for item in evidence if not item.get("aggregation"))
        return (f"Retrieved patient records ({counts}). The generated summary could "
                "not be verified against the returned records; review the evidence below.")
    return answer


def run_patient_request(person_id: int, request: str, data_dir: str,
                        model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                        concept_names: dict[int, str]) -> dict:
    """Route a request to source records or a validated measurement analysis."""
    person = patient_table(data_dir, "person", person_id, "history", None)
    if person.empty:
        raise ValueError(f"person_id {person_id} was not found in PERSON")
    available = [table for table in CLINICAL_TABLES
                 if os.path.isfile(os.path.join(data_dir, f"{table}.parquet"))]
    route = select_route_plan(model, tokenizer, request, available)
    trace = [{"stage": "Model route", "detail":
              f"Action: {route['action']}; scope: {route['scope']}; "
              f"tables: {', '.join(route['tables']) or 'none'}."}]
    if requires_derived_calculation(request):
        reason = ("This request needs a derived calculation that the available "
                  "measurement tools do not support. No rate or other value was estimated.")
        trace.append({"stage": "Validation", "detail": reason})
        return {"kind": "unsupported", "answer": reason,
                "evidence": [], "analysis": None, "decision_trace": trace}
    if route["action"] == "unsupported":
        reason = "The requested operation is not supported by the available analysis tools."
        trace.append({"stage": "Validation", "detail": reason})
        return {"kind": "unsupported", "answer": reason,
                "evidence": [], "analysis": None, "decision_trace": trace}
    plan = {**route, "concept_ids": [], "ancestor_ids": []}
    concept_selection = None
    if "measurement" in route["tables"]:
        measurement_rows = patient_table(data_dir, "measurement", person_id, "history", None)
        candidates = measurement_catalog(measurement_rows, concept_names, request)
        ancestor_candidates = measurement_ancestor_catalog(
            measurement_rows, concept_names, load_measurement_ancestors(data_dir), request)
        concept_selection = {
            "candidates": candidates,
            "ancestor_candidates": [
                {"ancestor_id": item["ancestor_id"], "name": item["name"],
                 "patient_descendant_count": item["patient_descendant_count"]}
                for item in ancestor_candidates
            ],
            "selected_concepts": [], "selected_ancestors": [],
            "selection_mode": "none",
        }
        trace.append({"stage": "Candidates", "detail":
                      f"Found {len(candidates)} patient measurement concepts and "
                      f"{len(ancestor_candidates)} matching ancestor groups."})
        selection = select_action_plan(model, tokenizer, request, route["tables"],
                                       candidates, ancestor_candidates)
        if selection["action"] == "unsupported":
            reason = selection.get("reason", "Could not select the requested measurements.")
            trace.append({"stage": "Validation", "detail": reason})
            return {"kind": "unsupported", "answer": reason,
                    "evidence": [], "analysis": None, "decision_trace": trace,
                    "concept_selection": concept_selection}
        plan["concept_ids"] = selection["concept_ids"]
        plan["ancestor_ids"] = selection["ancestor_ids"]
        counts = pd.to_numeric(measurement_rows["measurement_concept_id"], errors="coerce").value_counts()
        candidate_by_id = {item["concept_id"]: item for item in candidates}
        direct_ids = set(selection["direct_concept_ids"])
        concept_selection["selected_concepts"] = [
            {"concept_id": concept_id,
             "name": candidate_by_id[concept_id]["name"] if concept_id in candidate_by_id
                     else concept_names.get(concept_id, str(concept_id)),
             "row_count": int(counts.get(concept_id, 0)),
             "source": "direct" if concept_id in direct_ids else "ancestor"}
            for concept_id in plan["concept_ids"]
        ]
        selected_ancestor_ids = set(plan["ancestor_ids"])
        concept_selection["selected_ancestors"] = [
            item for item in concept_selection["ancestor_candidates"]
            if item["ancestor_id"] in selected_ancestor_ids
        ]
        concept_selection["selection_mode"] = (
            "selected" if plan["concept_ids"] else "all_measurements"
        )
        trace.append({"stage": "Model concept selection", "detail":
                      f"Concepts: {', '.join(f'{concept_id} ({concept_names.get(concept_id, "unknown")})' for concept_id in plan['concept_ids']) or 'none'}; "
                      f"ancestors: {', '.join(f'{ancestor_id} ({concept_names.get(ancestor_id, "unknown")})' for ancestor_id in plan['ancestor_ids']) or 'none'}."})
    action = plan["action"]
    trace.append({"stage": "Validation", "detail": "Selected patient data and operation passed validation."})
    if action == "records":
        selected = [table for table in plan["tables"] if table not in ("person", "visit_occurrence")]
        answer, evidence = search_patient(
            person_id=person_id, request=request, data_dir=data_dir,
            model=model, tokenizer=tokenizer, concept_names=concept_names,
            retrieval_plan=(selected, plan["scope"]),
            measurement_concept_ids=plan["concept_ids"],
            include_measurement_rows=True,
            generate_answer=False,
        )
        if "condition_occurrence" in selected:
            encounter = (latest_encounter(data_dir, person_id)
                         if plan["scope"] == "latest_encounter" else None)
            condition_rows = patient_table(data_dir, "condition_occurrence", person_id,
                                           plan["scope"], encounter)
            condition_summary = summarize_condition_history(
                condition_rows, concept_names, plan["scope"])
            evidence.append(condition_summary)
        trace.append({"stage": "Retrieval", "detail": "; ".join(
            f"{item['table']}: {item['total_rows']} rows ({len(item['shown_rows'])} shown)"
            for item in evidence)})
        answer = generate_record_answer(model, tokenizer, request, person_id,
                                        plan["scope"], evidence)
        trace.append({"stage": "Answer", "detail":
                      "Generated a record summary and checked its citations against returned evidence."})
        return {"kind": "records", "answer": answer, "evidence": evidence,
                "analysis": None, "decision_trace": trace,
                "concept_selection": concept_selection}

    encounter = latest_encounter(data_dir, person_id) if plan["scope"] == "latest_encounter" else None
    if plan["scope"] == "latest_encounter" and encounter is None:
        raise ValueError(f"person_id {person_id} has no VISIT_OCCURRENCE record")
    rows = patient_table(data_dir, "measurement", person_id, plan["scope"], encounter)
    result = analyze_measurements(rows, concept_names, plan["concept_ids"], action,
                                  plan["scope"])
    result["ancestor_ids"] = plan["ancestor_ids"]
    matched = rows
    if plan["concept_ids"]:
        ids = pd.to_numeric(rows["measurement_concept_id"], errors="coerce")
        matched = rows[ids.isin(plan["concept_ids"])]
    evidence = add_concept_names([{
        "table": "measurement", "total_rows": len(matched),
        "shown_rows": json_records(matched.head(100)),
        "retrieval_scope": plan["scope"],
    }], concept_names)
    if result["source_rows"]:
        answer = f"Found {result['source_rows']} matching measurement records across {len(result['series'])} concept and unit groups."
    else:
        answer = "No matching measurement records were found for this patient and time scope."
    trace.append({"stage": "Analysis", "detail":
                  f"Computed {action} from {result['source_rows']} source rows; "
                  f"{len(result['series'])} concept and unit groups."})
    return {"kind": action, "answer": answer, "evidence": evidence,
            "analysis": result, "decision_trace": trace,
            "concept_selection": concept_selection}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person_id", type=int, required=True)
    parser.add_argument("--request", required=True)

    args = parser.parse_args()

    person_id: int = args.person_id
    request: str = args.request

    cleaned_dir: str = get_path(path_type="cleaned", require_exists=True)
    model, tokenizer = load_model()
    concept_names: dict[int, str] = load_concept_names(data_dir=cleaned_dir)

    results: tuple[str, list[dict]] = search_patient(person_id=person_id, 
                                                     request=request, 
                                                     data_dir=cleaned_dir,
                                                     model=model,
                                                     tokenizer=tokenizer,
                                                     concept_names=concept_names)
    answer, evidence = results

    print("\nAnswer:\n" + answer)
    print("\nRetrieved OMOP rows:\n" + json.dumps(evidence, indent=2))
