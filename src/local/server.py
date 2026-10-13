"""Local Flask proxy to a patient search server on a HiPerGator compute node."""

import json
import os
import posixpath

from pathlib import Path

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request as flask_request
from flask_sock import Sock
from simple_websocket import ConnectionClosed

from .utils.tunnel import TunnelManager


def create_app(tunnel: TunnelManager) -> Flask:
    app = Flask(__name__)
    app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}
    sock = Sock(app)

    @sock.route("/tunnel/status")
    def tunnel_status(ws):
        version = -1
        try:
            while True:
                version, status = tunnel.wait_for_update(version)
                ws.send(json.dumps(status))
        except (ConnectionClosed, OSError):
            pass

    @app.get("/tunnel/status")
    def current_tunnel_status():
        return jsonify(tunnel.snapshot())

    @app.post("/tunnel/reconnect")
    def reconnect_tunnel():
        tunnel.start()
        return jsonify(tunnel.snapshot())

    def forward_upstream(path: str, data: bytes | None = None, timeout: int = 30):
        tunnel_port = tunnel.port()
        if tunnel_port is None:
            return jsonify(error="HiPerGator tunnel is not ready", status=tunnel.snapshot()), 503
        upstream = Request(
            f"http://127.0.0.1:{tunnel_port}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET",
        )
        try:
            with urlopen(upstream, timeout=timeout) as response:
                return Response(
                    response.read(), status=response.status,
                    content_type=response.headers.get("Content-Type", "application/json"),
                )
        except HTTPError as exc:
            return Response(
                exc.read(), status=exc.code,
                content_type=exc.headers.get("Content-Type", "application/json"),
            )
        except (URLError, OSError) as exc:
            app.logger.warning("HiPerGator connection failed: %s", exc)
            tunnel.mark_disconnected("The HiPerGator server connection was lost.")
            return jsonify(error="HiPerGator server is unavailable"), 502

    @app.get("/random-person")
    def random_person():
        return forward_upstream("/random-person")

    @app.post("/patient-search")
    def patient_search():
        payload = flask_request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a JSON object"), 400
        person_id = payload.get("person_id")
        request = payload.get("request")
        if isinstance(person_id, bool) or not isinstance(person_id, int):
            return jsonify(error="person_id must be an integer"), 400
        if not isinstance(request, str) or not request.strip():
            return jsonify(error="request must be a nonempty string"), 400

        return forward_upstream("/patient-search", json.dumps(payload).encode("utf-8"), 900)

    return app


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    # --- Load env Variables ---
    host = os.environ.get("HPG_HOST")
    username = os.environ.get("HPG_USER")
    repo_dir = os.environ.get("HPG_REPO_DIR", "").rstrip("/")

    # --- Sanity Check ---
    if not host or not username or not repo_dir:
        raise SystemExit("Set HPG_HOST, HPG_USER, and HPG_REPO_DIR in .env")
    if not posixpath.isabs(repo_dir):
        raise SystemExit("HPG_REPO_DIR must be an absolute path on HiPerGator")

    tunnel = TunnelManager(host, username, repo_dir)
    tunnel.start()
    try:
        create_app(tunnel).run(
            host="127.0.0.1",
            port=int(os.getenv("LOCAL_SERVER_PORT", "8765")),
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    finally:
        tunnel.stop()
