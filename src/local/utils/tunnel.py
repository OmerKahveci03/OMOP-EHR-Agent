"""Background HiPerGator tunnel lifecycle and status updates."""

import threading

import paramiko

from src.hipergator import SERVER_PORT
from .ssh import (TunnelServer, connect_ssh, ensure_server_job, remote_command,
                  server_jobs, wait_for_model_server)


class TunnelManager:
    def __init__(self, host: str, username: str, repo_dir: str):
        self.host = host
        self.username = username
        self.repo_dir = repo_dir
        self._condition = threading.Condition()
        self._status = {"state": "not_running", "detail": "Local server is starting."}
        self._version = 0
        self._port: int | None = None
        self._gateway: paramiko.Transport | None = None
        self._job_id: str | None = None
        self._cancel_job_on_stop = False
        self._job_cancelled = False
        self._connection_finished = False
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    def _publish(self, state: str, detail: str) -> None:
        with self._condition:
            status = {"state": state, "detail": detail}
            if status != self._status:
                self._status = status
                self._version += 1
                self._condition.notify_all()

    def snapshot(self) -> dict[str, str]:
        with self._condition:
            return self._status.copy()

    def wait_for_update(self, version: int, timeout: float = 25) -> tuple[int, dict[str, str]]:
        with self._condition:
            self._condition.wait_for(lambda: self._version != version, timeout=timeout)
            return self._version, self._status.copy()

    def port(self) -> int | None:
        with self._condition:
            return self._port if self._status["state"] == "ready" else None

    def start(self) -> bool:
        with self._condition:
            if self._worker is not None and self._worker.is_alive():
                return False
            self._stop.clear()
            self._port = None
            self._job_id = None
            self._cancel_job_on_stop = False
            self._job_cancelled = False
            self._connection_finished = False
            self._publish("connecting", "Enter your SSH password and Duo response in the local terminal.")
            self._worker = threading.Thread(target=self._connect, daemon=True)
            self._worker.start()
            return True

    def mark_disconnected(self, detail: str) -> None:
        self._stop.set()
        with self._condition:
            self._port = None
            gateway = self._gateway
        self._publish("disconnected", detail)
        if gateway is not None:
            gateway.close()

    def _remember_job(self, job_id: str) -> None:
        with self._condition:
            self._job_id = job_id
            self._condition.notify_all()

    def _cancel_job(self, gateway: paramiko.Transport | None) -> None:
        """Called with the condition held, before the SSH transport closes."""
        job_id = self._job_id
        if not job_id or self._job_cancelled:
            return
        if gateway is None or not gateway.is_active():
            print(f"Could not cancel HiPerGator server job {job_id}: SSH is disconnected", flush=True)
            return
        try:
            remote_command(gateway, f"scancel {job_id}")
            self._job_cancelled = True
            print(f"Cancelled HiPerGator server job {job_id}", flush=True)
        except Exception as exc:
            print(f"Could not cancel HiPerGator server job {job_id}: {exc}", flush=True)

    def stop(self, cancel_job: bool = False) -> None:
        """Close the tunnel, optionally cancelling its tracked Slurm job first."""
        with self._condition:
            self._stop.set()
            self._cancel_job_on_stop = cancel_job
            if cancel_job:
                self._condition.wait_for(
                    lambda: self._job_id is not None or self._connection_finished
                    or self._worker is None,
                    timeout=20,
                )
            gateway = self._gateway
            if cancel_job:
                self._cancel_job(gateway)
        if gateway is not None:
            gateway.close()

    def _connect(self) -> None:
        gateway = None
        tunnel = None
        tunnel_thread = None
        try:
            gateway = connect_ssh(self.host, 22, self.username)
            if self._stop.is_set():
                return
            with self._condition:
                self._gateway = gateway
            self._publish("checking", "Checking for a HiPerGator server job.")
            job_id, server_host, log_path = ensure_server_job(
                gateway, self.username, self.repo_dir, on_status=self._publish,
                on_job_id=self._remember_job, is_stopped=self._stop.is_set,
            )
            if self._stop.is_set():
                return
            wait_for_model_server(
                gateway, self.username, job_id, server_host, log_path,
                on_status=self._publish,
            )
            if self._stop.is_set():
                return

            tunnel = TunnelServer(gateway, (server_host, SERVER_PORT))
            tunnel_thread = threading.Thread(target=tunnel.serve_forever, daemon=True)
            tunnel_thread.start()
            with self._condition:
                self._port = tunnel.server_address[1]
            self._publish("ready", f"Connected to HiPerGator job {job_id} on {server_host}.")
            print(f"SSH tunnel ready for {server_host}:{SERVER_PORT}", flush=True)

            while not self._stop.wait(10):
                if not gateway.is_active() or not tunnel_thread.is_alive():
                    self._publish("disconnected", "The SSH tunnel disconnected.")
                    break
                if not any(job[0] == job_id for job in server_jobs(gateway, self.username)):
                    self._publish("disconnected", f"HiPerGator job {job_id} is no longer running.")
                    break
        except Exception as exc:
            if not self._stop.is_set():
                print(f"HiPerGator tunnel failed: {exc}", flush=True)
                self._publish("error", str(exc))
        finally:
            with self._condition:
                if self._cancel_job_on_stop and not self._job_cancelled:
                    self._cancel_job(gateway)
                self._port = None
                self._gateway = None
                if self._status["state"] == "ready":
                    self._publish("disconnected", "The SSH tunnel stopped.")
                self._connection_finished = True
                self._condition.notify_all()
                if gateway is not None:
                    gateway.close()
            if tunnel is not None:
                if tunnel_thread is not None and tunnel_thread.is_alive():
                    tunnel.shutdown()
                tunnel.server_close()
