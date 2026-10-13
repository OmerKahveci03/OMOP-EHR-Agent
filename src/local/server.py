"""Local Flask proxy to a patient search server on a HiPerGator compute node."""

import json
import os
import posixpath
import threading
import paramiko

from typing import Optional
from pathlib import Path

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request as flask_request

from src.hipergator import SERVER_PORT
from .utils.ssh import TunnelServer, connect_ssh, ensure_server_job, wait_for_model_server


def create_app(tunnel_port: int) -> Flask:
    app = Flask(__name__)

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

        upstream = Request(
            f"http://127.0.0.1:{tunnel_port}/patient-search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(upstream, timeout=900) as response:
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
            return jsonify(error="HiPerGator server is unavailable"), 502

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

    # --- Connect SSH Local -> Hipergator ---
    gateway: paramiko.Transport = connect_ssh(host, 22, username)
    try:
        # -- Launch The Server Job If Not Accessible --
        job_id, server_host, log_path = ensure_server_job(gateway, username, repo_dir)
        compute_connection: Optional[paramiko.Channel] = gateway.open_channel(
            "direct-tcpip", 
            (server_host, 22), 
            ("127.0.0.1", 0)
        )

        if compute_connection is None:
            raise ConnectionError(f"Could not reach compute node {server_host} over SSH")

        # -- Connect SSH Hipergator -> Compute Node --
        compute: paramiko.Transport = connect_ssh(server_host, 22, username, compute_connection)
        
        try:
            wait_for_model_server(gateway, compute, username, job_id, log_path)
            tunnel: TunnelServer = TunnelServer(compute, ("127.0.0.1", SERVER_PORT))
            try:
                tunnel_thread: threading.Thread = threading.Thread(
                    target=tunnel.serve_forever, 
                    daemon=True
                )

                tunnel_thread.start()
                print(f"SSH tunnel ready for {server_host}:{tunnel.destination[1]}", flush=True)

                # -- We Are Good To Run the App Once the Tunnel is Established --
                create_app(tunnel.server_address[1]).run(
                    host="127.0.0.1",
                    port=int(os.getenv("LOCAL_SERVER_PORT", "8765")),
                    debug=False,
                    use_reloader=False,
                )

            finally:
                tunnel.shutdown()
                tunnel.server_close()

        finally:
            compute.close()

    finally:
        gateway.close()
