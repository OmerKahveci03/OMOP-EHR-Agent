"""HTTP API for repeated patient searches with one loaded model."""

from functools import lru_cache
import os
import secrets
import threading

import pandas as pd

from flask import Flask, Response, jsonify, request as flask_request

from . import LLAMA_PATH, SERVER_PORT
from .patient_search import load_model, run_patient_request
from .concepts import load_concept_names
from .utils.paths import get_path


@lru_cache(maxsize=1)
def load_person_ids(data_dir: str) -> tuple[int, ...]:
    """Read the person IDs once, without loading the other PERSON columns."""

    person_file: str = os.path.join(data_dir, "person.parquet")
    ids: pd.Series = pd.read_parquet(person_file, columns=["person_id"])["person_id"]

    return tuple(int(value) for value in ids.dropna().unique())


def create_app(model_path: str = LLAMA_PATH) -> Flask:
    """Load weights at startup and return the patient search app."""

    data_dir: str = get_path(path_type="cleaned", require_exists=True)

    print("\n === Loading model weights ===\n", flush=True)
    model, tokenizer = load_model(model_path=model_path)

    print("\n === Loading concept names ===\n", flush=True)
    concept_names: dict[int, str] = load_concept_names(data_dir=data_dir)

    print("\n === Model loaded; starting patient search server ===\n", flush=True)
    generation_lock = threading.Lock()

    app = Flask(__name__)

    @app.get("/random-person")
    def random_person() -> Response | tuple[Response, int]:
        """ Select a random person ID and return it. """
        try:
            person_ids: tuple[int, ...] = load_person_ids(data_dir=data_dir)

        except (FileNotFoundError, KeyError, OSError) as exc:
            app.logger.error("Could not load person IDs: %s", exc)
            return jsonify(error="PERSON data is unavailable"), 503

        if not person_ids:
            return jsonify(error="PERSON data contains no patient IDs"), 503

        return jsonify({"person_id" : str(secrets.choice(person_ids))})

    @app.post("/patient-search")
    def patient_search() -> Response | tuple[Response, int]:
        """ Trigger the model to run patient search. """

        # --- Get Payload; Expecting a Dict ---
        payload = flask_request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a JSON object"), 400

        # --- Get Person ID; Expecting an Int ---
        person_id = payload.get("person_id")
        if isinstance(person_id, bool) or not isinstance(person_id, int):
            return jsonify(error="person_id must be an integer"), 400

        # --- Get Request; Expecting a String ---
        request = payload.get("request")
        if not isinstance(request, str) or not request.strip():
            return jsonify(error="request must be a nonempty string"), 400

        # --- Run Patient Search ---
        try:
            # Only one at a time
            with generation_lock:
                result = run_patient_request(person_id=person_id,
                                             request=request,
                                             data_dir=data_dir,
                                             model=model,
                                             tokenizer=tokenizer,
                                             concept_names=concept_names)

        except ValueError as exc:
            return jsonify(error=str(exc)), 404 if "was not found" in str(exc) else 400
        except FileNotFoundError as exc:
            app.logger.error("Search data is missing: %s", exc)
            return jsonify(error=str(exc)), 503

        return jsonify(result)

    return app


if __name__ == "__main__":
    app: Flask = create_app()
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, use_reloader=False)
