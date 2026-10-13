import argparse
import json
import re
import os

import torch

import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from . import LLAMA_PATH
from .concepts import add_concept_names, load_concept_names
from .utils.paths import get_path

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


def select_tables(model: PreTrainedModel, 
                  tokenizer: PreTrainedTokenizerBase, 
                  request: str, 
                  available: list[str]) -> list[str]:
    """ Uses a pretrained model to select OMOP tables relevant to the request. """

    # --- Prompt the Model ---
    response: str = generate(
        model=model,
        tokenizer=tokenizer,
        system=(
            "Select OMOP tables relevant to the question. "
            "Return only a JSON array of table names. "
            "Use only names from the supplied list. "
            "Select at most five tables. Never generate SQL or Python."
        ),
        user=f"Available tables: {json.dumps(available)}\nQuestion: {request}",
        max_new_tokens=128,
    )

    # --- Get Model Response ---
    match = re.search(r"\[[\s\S]*?\]", response)
    if not match:
        raise ValueError(f"Model did not return a table list: {response}")

    names = json.loads(match.group())
    if not isinstance(names, list) or any(name not in available for name in names):
        raise ValueError(f"Model selected an unavailable table: {names}")

    return list(dict.fromkeys(names))[:5]


def patient_rows(data_dir: str, table: str, 
                 person_id: int, limit: int = 100) -> dict[str, str | int | list[dict]]:
    """ Load only rows belonging to the patient from the requested table. 
    
    Returns result as dict with the keys: `table`, `total_rows`, `shown_rows`."""

    fp: str = os.path.join(data_dir, f"{table}.parquet")
    df: pd.DataFrame = pd.read_parquet(fp, filters=[("person_id", "==", person_id)])

    # --- Sort by date if possible ---
    date_columns: list[str] = [
        col for col in df.columns 
        if col.endswith("_date") or col.endswith("_datetime")
    ]

    if date_columns:
        df = df.sort_values(date_columns[0], ascending=False, na_position="last")

    rows = json.loads(df.head(limit).to_json(orient="records", date_format="iso"))
    return {"table": table, "total_rows": len(df), "shown_rows": rows}


def fit_evidence(evidence: list[dict], 
                 tokenizer: PreTrainedTokenizerBase, 
                 max_tokens: int) -> list[dict]:
    """Keep the model input within its context window, across all tables."""

    fitted: list[dict] = [
        {
            "table": item["table"], 
            "total_rows": item["total_rows"],
            "shown_rows": []
        } 
        for item in evidence
    ]

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
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto", local_files_only=True
    )
    model.eval()
    return model, tokenizer


def search_patient(person_id: int, request: str,
                   data_dir: str,
                   model: PreTrainedModel,
                   tokenizer: PreTrainedTokenizerBase,
                   concept_names: dict[int, str]) -> tuple[str, list[dict]]:
    """ Given a patient and natural language request, return EHR. """

    person_fp: str = os.path.join(data_dir, "person.parquet")
    if not os.path.exists(person_fp):
        raise FileNotFoundError(f"Missing {person_fp}; run preprocessing first")

    # Keys are `table`, `total_rows`, `shown_rows` 
    person: dict = patient_rows(data_dir, "person", person_id, 1)

    if not person["shown_rows"]:
        raise ValueError(f"person_id {person_id} was not found in PERSON")

    available: list[str] = [name for name in CLINICAL_TABLES
                            if name != "person" and 
                            os.path.exists(os.path.join(data_dir, f"{name}.parquet"))]

    if not available:
        raise ValueError(f"No clinical OMOP Parquet tables found in {data_dir}")

    # --- Use the Model to Select Tables ---
    selected: list[str] = select_tables(model, tokenizer, request, available)

    evidence: list[dict] = [person,
                            *[patient_rows(data_dir=data_dir, 
                                           table=name, 
                                           person_id=person_id)

                              for name in selected]]

    # The model and the frontend should see the same concept names. Enrich
    # before fitting so the names count toward the model's context limit.
    evidence = add_concept_names(evidence, concept_names)

    context_limit: int = min(5000, model.config.max_position_embeddings - 1024)
    evidence: list[dict] = fit_evidence(evidence=evidence, 
                                        tokenizer=tokenizer, 
                                        max_tokens=context_limit)

    answer: str = generate(
        model, tokenizer,
        "Answer the user's question using only the supplied OMOP rows. "
        "Cite supporting records as [table:record_id], using the actual table and "
        "record ID field shown in the data. If the rows do not support an answer, "
        "say so. Do not infer a diagnosis or treatment from incomplete records. "
        "Dates in this de-identified dataset are shifted, sometimes by more than "
        "a century. Keep dates as recorded, but do not describe them as future events.",
        f"Question: {request}\nPerson ID: {person_id}\n"
        f"Evidence (some tables may be truncated): {json.dumps(evidence)}",
        max_new_tokens=512,
    )
    return answer, evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person_id", type=int, required=True)
    parser.add_argument("--request", required=True)

    args = parser.parse_args()

    person_id: int = args.person_id
    request: str = args.request

    cleaned_dir: str = get_path(path_type="cleaned", require_exists=True)
    model, tokenizer = load_model()
    concept_names = load_concept_names(cleaned_dir)

    results: tuple[str, list[dict]] = search_patient(person_id=person_id, 
                                                     request=request, 
                                                     data_dir=cleaned_dir,
                                                     model=model,
                                                     tokenizer=tokenizer,
                                                     concept_names=concept_names)
    answer, evidence = results

    print("\nAnswer:\n" + answer)
    print("\nRetrieved OMOP rows:\n" + json.dumps(evidence, indent=2))
