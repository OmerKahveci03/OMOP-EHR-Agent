"""SSH tunneling and remote Slurm startup helpers for the local server."""

import getpass
from pathlib import Path
import posixpath
import select
import shlex
import socket
import socketserver
import time
from typing import Callable

import paramiko

from src.hipergator import SERVER_PORT


class TunnelHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        tunnel = self.server  # type: ignore[assignment]
        channel = tunnel.transport.open_channel(
            "direct-tcpip", tunnel.destination, self.client_address
        )
        if channel is None:
            return
        try:
            while True:
                readable, _, _ = select.select([self.request, channel], [], [])
                if self.request in readable:
                    data = self.request.recv(65536)
                    if not data:
                        break
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(65536)
                    if not data:
                        break
                    self.request.sendall(data)
        finally:
            channel.close()


class TunnelServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, transport: paramiko.Transport, destination: tuple[str, int]):
        self.transport = transport
        self.destination = destination
        super().__init__(("127.0.0.1", 0), TunnelHandler)


def connect_ssh(
    host: str, port: int, username: str, connection: socket.socket | paramiko.Channel | None = None
) -> paramiko.Transport:
    """Verify the SSH host and authenticate using terminal prompts."""
    if connection is None:
        connection = socket.create_connection((host, port), timeout=20)
    transport = paramiko.Transport(connection)
    try:
        transport.start_client(timeout=20)
        host_keys = paramiko.HostKeys()
        known_hosts = Path.home() / ".ssh" / "known_hosts"
        if known_hosts.exists():
            host_keys.load(str(known_hosts))
        host_key_name = host if port == 22 else f"[{host}]:{port}"
        if not host_keys.check(host_key_name, transport.get_remote_server_key()):
            raise ValueError(
                f"SSH host key for {host_key_name} is missing or different in {known_hosts}; "
                "verify it and connect once with ssh first"
            )

        def answer_prompts(title: str, instructions: str, prompts: list[tuple[str, bool]]) -> list[str]:
            for message in (title, instructions):
                if message.strip():
                    print(message, flush=True)
            return [input(prompt) if echo else getpass.getpass(prompt) for prompt, echo in prompts]

        allowed = ["keyboard-interactive"]
        failed_attempts = 0
        for _ in range(10):
            if transport.is_authenticated():
                break
            if "keyboard-interactive" in allowed:
                try:
                    allowed = transport.auth_interactive(username, answer_prompts)
                    continue
                except paramiko.BadAuthenticationType as exc:
                    allowed = exc.allowed_types
                except paramiko.AuthenticationException:
                    failed_attempts += 1
                    if failed_attempts >= 3:
                        break
                    print("SSH authentication failed. Please try again.", flush=True)
                    continue
            if "password" in allowed:
                try:
                    allowed = transport.auth_password(
                        username, getpass.getpass("SSH password: "), fallback=False
                    )
                    continue
                except paramiko.BadAuthenticationType as exc:
                    allowed = exc.allowed_types
                except paramiko.AuthenticationException:
                    failed_attempts += 1
                    if failed_attempts >= 3:
                        break
                    print("SSH authentication failed. Please try again.", flush=True)
                    continue
            raise paramiko.AuthenticationException(
                f"SSH requires an unsupported authentication method: {allowed}"
            )
        if not transport.is_authenticated():
            raise paramiko.AuthenticationException(
                "SSH authentication did not complete after three failed attempts"
                if failed_attempts >= 3 else "SSH authentication did not complete"
            )
        transport.set_keepalive(30)
        return transport
    except Exception:
        transport.close()
        raise


def remote_command(gateway: paramiko.Transport, command: str) -> str:
    """Run a short command on the login node and return its stdout."""
    channel = gateway.open_session()
    try:
        channel.exec_command(command)
        output = channel.makefile("r").read()
        errors = channel.makefile_stderr("r").read()
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        if isinstance(errors, bytes):
            errors = errors.decode("utf-8", errors="replace")
        if channel.recv_exit_status() != 0:
            raise RuntimeError(f"Remote command failed: {errors.strip()}")
    finally:
        channel.close()
    return output


def server_jobs(gateway: paramiko.Transport, username: str) -> list[tuple[str, str, str, str]]:
    output = remote_command(
        gateway,
        f"squeue -h -u {shlex.quote(username)} -n patient_search_server "
        "-o '%i|%T|%N|%R'",
    )
    jobs = []
    for line in output.splitlines():
        fields = line.split("|", 3)
        if len(fields) == 4:
            jobs.append(tuple(fields))
    return jobs


def show_server_log(sftp: paramiko.SFTPClient, path: str, offset: int) -> tuple[int, str]:
    """Print new Slurm output and return it for startup checks."""
    text = ""
    try:
        size = sftp.stat(path).st_size
        if size < offset:
            offset = 0
        if size > offset:
            with sftp.open(path, "rb") as log:
                log.seek(offset)
                data = log.read(min(size - offset, 65536))
            if data:
                text = data.decode("utf-8", errors="replace")
                print(text, end="", flush=True)
                offset += len(data)
    except OSError:
        pass  # Slurm may not have created the output file yet.
    return offset, text


def job_stdout_path(gateway: paramiko.Transport, job_id: str, repo_dir: str) -> str:
    """Get Slurm's stdout path, including jobs submitted outside this process."""
    details = remote_command(gateway, f"scontrol show job -o {job_id}")
    for field in details.split():
        if field.startswith("StdOut="):
            return field.partition("=")[2]
    return posixpath.join(repo_dir, f"server_{job_id}.out")


def ensure_server_job(
    gateway: paramiko.Transport, username: str, repo_dir: str,
    on_status: Callable[[str, str], None] | None = None,
) -> tuple[str, str, str]:
    """Reuse or submit the Slurm job, then wait for its compute node."""
    jobs = server_jobs(gateway, username)
    if len(jobs) > 1:
        raise RuntimeError("More than one patient_search_server job is active")
    if jobs:
        job_id = jobs[0][0]
        print(f"Using existing server job {job_id}", flush=True)
        if on_status:
            on_status("booting", f"Using server job {job_id}.")
    else:
        if on_status:
            on_status("not_running", "No server job is running on HiPerGator.")
        script = posixpath.join(repo_dir, "slurm", "run_server.slurm")
        print(f"Submitting {script}", flush=True)
        if on_status:
            on_status("booting", "Submitting the HiPerGator server job.")
        output = remote_command(
            gateway,
            f"sbatch --parsable --chdir={shlex.quote(repo_dir)} {shlex.quote(script)}",
        )
        job_id = output.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {output.strip()}")
        print(f"Submitted server job {job_id}", flush=True)

    log_path = job_stdout_path(gateway, job_id, repo_dir)
    sftp = paramiko.SFTPClient.from_transport(gateway)
    try:
        offset = 0
        previous = None
        deadline = time.monotonic() + 1200
        submitted_at = time.monotonic()
        last_report = 0.0
        while time.monotonic() < deadline:
            matches = [job for job in server_jobs(gateway, username) if job[0] == job_id]
            offset, _ = show_server_log(sftp, log_path, offset)
            if not matches:
                if not jobs and time.monotonic() - submitted_at < 30:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"Server job {job_id} left the queue; check its Slurm logs")
            _, state, node, reason = matches[0]
            status = (state, node, reason)
            if status != previous or time.monotonic() - last_report >= 30:
                print(f"Job {job_id}: {state} ({reason})", flush=True)
                if on_status:
                    on_status("booting", f"Slurm job {job_id}: {state.lower()} ({reason}).")
                previous = status
                last_report = time.monotonic()
            if state == "RUNNING" and node not in ("(null)", "None", ""):
                return job_id, node, log_path
            time.sleep(5)
        raise TimeoutError(f"Timed out waiting for server job {job_id} to start")
    finally:
        sftp.close()


def wait_for_model_server(
    gateway: paramiko.Transport, username: str, job_id: str,
    server_host: str, log_path: str,
    on_status: Callable[[str, str], None] | None = None,
) -> None:
    """Wait until the model has loaded and the HTTP port accepts connections."""
    sftp = paramiko.SFTPClient.from_transport(gateway)
    try:
        offset = 0
        error_offset = 0
        error_path = log_path.removesuffix(".out") + ".err"
        model_loaded = False
        recent_output = ""
        deadline = time.monotonic() + 1200
        last_report = time.monotonic()
        print("Waiting for model server to accept connections...", flush=True)
        if on_status:
            on_status("booting", f"Job {job_id} is running; waiting for model weights to load.")
        while time.monotonic() < deadline:
            offset, output = show_server_log(sftp, log_path, offset)
            error_offset, _ = show_server_log(sftp, error_path, error_offset)
            recent_output = (recent_output + output)[-200:]
            if not model_loaded and "Model loaded; starting patient search server." in recent_output:
                model_loaded = True
                if on_status:
                    on_status("booting", "Model loaded; waiting for the server port.")
            if model_loaded:
                try:
                    probe = gateway.open_channel(
                        "direct-tcpip", (server_host, SERVER_PORT), ("127.0.0.1", 0), timeout=5
                    )
                    if probe is not None:
                        probe.close()
                        print("Model server is ready.", flush=True)
                        return
                except (OSError, paramiko.SSHException):
                    pass
            if not any(job[0] == job_id for job in server_jobs(gateway, username)):
                raise RuntimeError(f"Server job {job_id} stopped before the model server was ready")
            if time.monotonic() - last_report >= 30:
                print(f"Job {job_id} is running; still waiting for the model server...", flush=True)
                last_report = time.monotonic()
            time.sleep(5)
        raise TimeoutError(f"Timed out waiting for model server in job {job_id}")
    finally:
        sftp.close()


