"""HTTP API for repeated patient searches with one loaded model."""

import threading

from flask import Flask, jsonify, request as flask_request

from . import LLAMA_PATH, SERVER_PORT
from .patient_search import load_model, search_patient
from .utils.paths import get_path


def create_app(model_path: str = LLAMA_PATH) -> Flask:
    """Load weights at startup and return the patient search app."""

    data_dir: str = get_path("cleaned", require_exists=True)

    model, tokenizer = load_model(model_path)
    generation_lock = threading.Lock()
    app = Flask(__name__)

    @app.post("/patient-search")
    def patient_search():
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
                answer, evidence = search_patient(person_id=person_id, 
                                                  request=request, 
                                                  data_dir=data_dir, 
                                                  model=model, 
                                                  tokenizer=tokenizer)

        except ValueError as exc:
            return jsonify(error=str(exc)), 404 if "was not found" in str(exc) else 400

        return jsonify(answer=answer, evidence=evidence)

    return app


if __name__ == "__main__":
    app: Flask = create_app()
    app.run(host="127.0.0.1", port=SERVER_PORT, debug=False, use_reloader=False)
