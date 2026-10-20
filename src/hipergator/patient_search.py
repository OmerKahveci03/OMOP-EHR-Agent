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
from .analysis import ANALYSIS_KINDS, analyze_measurements, measurement_catalog
from .concepts import add_concept_names, load_concept_names
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


def select_action_plan(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizerBase,
                       request: str, available: list[str],
                       candidates: list[dict]) -> dict:
    """Choose a validated patient-scoped retrieval or measurement operation."""
    response = generate(
        model=model, tokenizer=tokenizer,
        system=(
            "You navigate one patient's OMOP data. Return only JSON with keys "
            "action, scope, tables, concept_ids. Actions: records, time_series, "
            "distribution, statistics, unsupported. Use records to show source rows; "
            "use an analysis action only for numeric measurement requests. "
            "The analysis actions calculate values directly from recorded measurements; "
            "they cannot derive rates or perform unit conversions. Choose unsupported "
            "when the requested calculation needs either. Choose concept_ids only from "
            "the supplied patient-specific candidates. Use [] only when the user "
            "explicitly asks about all measurements; for a named measurement "
            "with no matching candidate choose unsupported. "
            "Scope means time only: use exactly latest_encounter or history, never "
            "patient or person. Choose latest_encounter unless the question asks for history, earlier "
            "encounters, or a comparison across encounters. Use history for a list "
            "of visits. Choose tables only from the supplied list, at most five. "
            "Analysis actions use the measurement table. Never generate SQL or Python."
        ),
        user=f"Available tables: {json.dumps(available)}\n"
             f"Measurement candidates: {json.dumps(candidates)}\nQuestion: {request}",
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
    if not isinstance(concept_ids, list) or any(
        isinstance(concept_id, bool) or not isinstance(concept_id, int)
        or concept_id not in candidate_ids for concept_id in concept_ids
    ):
        raise ValueError(f"Unavailable measurement concepts: {concept_ids}")
    if action in ANALYSIS_KINDS and "measurement" not in available:
        raise ValueError("Measurement table is unavailable")
    return {"action": action, "scope": scope,
            "tables": list(dict.fromkeys(tables)),
            "concept_ids": list(dict.fromkeys(concept_ids))}


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
                   include_measurement_rows: bool = False) -> tuple[str, list[dict]]:
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


def run_patient_request(person_id: int, request: str, data_dir: str,
                        model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                        concept_names: dict[int, str]) -> dict:
    """Route a request to source records or a validated measurement analysis."""
    person = patient_table(data_dir, "person", person_id, "history", None)
    if person.empty:
        raise ValueError(f"person_id {person_id} was not found in PERSON")
    available = [table for table in CLINICAL_TABLES
                 if os.path.isfile(os.path.join(data_dir, f"{table}.parquet"))]
    measurement_rows = (
        patient_table(data_dir, "measurement", person_id, "history", None)
        if "measurement" in available else pd.DataFrame()
    )
    candidates = measurement_catalog(measurement_rows, concept_names, request)
    plan = select_action_plan(model, tokenizer, request, available, candidates)
    action = plan["action"]
    if action == "unsupported":
        return {"kind": "unsupported", "answer":
                "This request needs a calculation that the available analysis tools do not support.",
                "evidence": [], "analysis": None}
    if action == "records":
        selected = [table for table in plan["tables"] if table not in ("person", "visit_occurrence")]
        answer, evidence = search_patient(
            person_id=person_id, request=request, data_dir=data_dir,
            model=model, tokenizer=tokenizer, concept_names=concept_names,
            retrieval_plan=(selected, plan["scope"]),
            measurement_concept_ids=plan["concept_ids"],
            include_measurement_rows=True,
        )
        return {"kind": "records", "answer": answer, "evidence": evidence,
                "analysis": None}

    encounter = latest_encounter(data_dir, person_id) if plan["scope"] == "latest_encounter" else None
    if plan["scope"] == "latest_encounter" and encounter is None:
        raise ValueError(f"person_id {person_id} has no VISIT_OCCURRENCE record")
    rows = patient_table(data_dir, "measurement", person_id, plan["scope"], encounter)
    result = analyze_measurements(rows, concept_names, plan["concept_ids"], action,
                                  plan["scope"])
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
    return {"kind": action, "answer": answer, "evidence": evidence, "analysis": result}


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
