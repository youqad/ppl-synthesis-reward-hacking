#!/usr/bin/env python3
"""Launch training services on a RunPod GPU pod.

The launcher manages a complete pod lifecycle: create the pod, poll until SSH
is reachable, rsync the repo (excluding .git, artifacts, .env, and virtualenvs),
start training inside a tmux session named "train", optionally run a smoke
check, then attach you to the session. When you detach (Ctrl+B, D), it asks
whether to terminate the pod. Ctrl+C or any unhandled error auto-terminates.
Billing continues until the pod is actually terminated; don't just close the
terminal.

Usage::

    # TRL GRPO training (args after -- forwarded to trl_reward_hacking.py)
    python scripts/runpod/launch.py --mode trl --name psrh-run7 -- --n-steps 1000

    # TRL GRPO training with the staged Stan integration (cmdsafestan required)
    python scripts/runpod/launch.py --mode trl_stan --name psrh-run7-stan -- --n-steps 1000

    # Tinker API server on the pod
    python scripts/runpod/launch.py --mode tinker_api --name psrh-tinker \\
        --base-model Qwen/Qwen3-0.6B --backend fsdp

    # Validate RunPod setup/catalog access without creating a pod
    python scripts/runpod/launch.py --preflight-only

    # Pull artifacts from a running pod
    scp -P <port> root@<ip>:/workspace/ppl-synthesis-reward-hacking/artifacts/ ./artifacts/

Environment variables (from .env or shell):

    RUNPOD_API_KEY      Required. RunPod API key for pod management.
    WANDB_API_KEY       Forwarded to pod env if set (for W&B logging).
    WANDB_ENTITY        Forwarded to pod env if set.
    WANDB_PROJECT       Forwarded to pod env if set.
    HF_TOKEN            Forwarded to pod env if set (for gated model downloads).
    HUGGING_FACE_HUB_TOKEN  Forwarded to pod env if set (legacy HF auth).
    TINKER_API_KEY      Required for --mode tinker_api. Must not be "tml-dummy".
    CMDSAFESTAN_PIP_SPEC Optional pip spec forwarded for --mode trl_stan
                         (example: "git+https://github.com/jkarwowski/cmdsafestan.git")

SSH: the launcher uses BatchMode=yes (no interactive password prompts). If your
SSH key isn't in the default location, pass --identity-file (-i). RunPod maps
container port 22 to a random public port; the launcher extracts this from pod
metadata and passes -p <port> to ssh/rsync/scp.

Rsync exclusions: .git, .env, .env.*, .scratch/, .notes/, .archive/, artifacts/,
background-documents/, __pycache__, .pixi, .venv, .dgx-venv, wandb-dir/.

The tmux session on the pod is named "train". Reattach manually with:
    ssh -p <port> root@<ip> -t 'tmux attach -t train'
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ppl_synthesis_reward_hacking.utils.runpod import (  # noqa: E402
    DEFAULT_DISK_GB,
    DEFAULT_GPU_TYPE,
    DEFAULT_IMAGE,
    collect_preflight_snapshot,
    create_training_pod,
    terminate_pod,
    validate_runpod_setup,
    wait_for_ssh,
)

DEFAULT_MODE = "trl"
DEFAULT_API_PORT = 8000
DEFAULT_BACKEND = "fsdp"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_SKYRL_REF = "skyrl_train-v0.4.0"
DEFAULT_ENV_MODE = "minimal"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REMOTE_DIR = "/workspace/ppl-synthesis-reward-hacking"
SSH_OPTIONS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=30",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
    "-o",
    "BatchMode=yes",
]
RSYNC_EXCLUDE = [
    ".git",
    ".env",
    ".env.*",
    ".scratch/",
    ".notes/",
    ".archive/",
    "artifacts/",
    "background-documents/",
    "__pycache__",
    ".pixi",
    ".venv",
    ".dgx-venv",
    "wandb-dir/",
]
RSYNC_MAX_RETRIES = 3
RSYNC_RETRY_DELAY = 5


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch training workflows on RunPod")
    parser.add_argument(
        "--mode",
        choices=["trl", "trl_stan", "tinker_api"],
        default=DEFAULT_MODE,
        help="Launcher mode: TRL PyMC reward, TRL Stan reward, or SkyRL Tinker API server",
    )
    parser.add_argument("--name", default="psrh-grpo", help="Pod name (default: psrh-grpo)")
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE, help="RunPod GPU type ID")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image")
    parser.add_argument(
        "--disk-gb", type=int, default=DEFAULT_DISK_GB, help="Container disk size in GB"
    )
    parser.add_argument("--timeout", type=int, default=300, help="SSH readiness timeout (seconds)")
    parser.add_argument(
        "--smoke-timeout",
        type=int,
        default=300,
        help="Tinker API smoke-check timeout (seconds, default: 300)",
    )
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help="API port")
    parser.add_argument(
        "--identity-file",
        "-i",
        help="SSH identity file (private key) for pod access",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["fsdp", "megatron"],
        help="SkyRL backend for tinker_api mode",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model for SkyRL tinker_api mode",
    )
    parser.add_argument(
        "--skyrl-ref",
        default=DEFAULT_SKYRL_REF,
        help="SkyRL git tag/branch/commit for tinker_api mode",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip tinker_api HTTP smoke checks",
    )
    parser.add_argument(
        "--no-attach",
        action="store_true",
        help="Don't attach to tmux after launch",
    )
    parser.add_argument(
        "--template-id",
        default=None,
        help="Optional RunPod template ID (recommended for org-standardized images)",
    )
    parser.add_argument(
        "--network-volume-id",
        default=None,
        help="Optional RunPod network volume ID to mount on the pod",
    )
    parser.add_argument(
        "--env-mode",
        choices=["minimal", "extended"],
        default=DEFAULT_ENV_MODE,
        help=(
            "Environment forwarding policy. "
            "minimal=whitelisted keys only (default), "
            "extended=also forward OPENAI_/ANTHROPIC_/CMDSAFESTAN_ prefixes"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate API key and print account/GPU/template visibility, then exit",
    )
    return parser.parse_known_args()


def ssh_cmd(ip: str, port: int, identity_file: str | None = None) -> list[str]:
    cmd = ["ssh", "-p", str(port), *SSH_OPTIONS]
    if identity_file:
        cmd.extend(["-i", identity_file])
    cmd.append(f"root@{ip}")
    return cmd


def pod_ports_for_mode(mode: str, api_port: int) -> str | None:
    if mode == "tinker_api":
        return f"22/tcp,{api_port}/http"
    return None


def build_pod_env(mode: str, *, env_mode: str = DEFAULT_ENV_MODE) -> dict[str, str]:
    pod_env: dict[str, str] = {}
    for key in (
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "CMDSAFESTAN_PIP_SPEC",
    ):
        val = os.environ.get(key)
        if val:
            pod_env[key] = val

    if env_mode == "extended":
        for key, val in os.environ.items():
            if not val:
                continue
            if key.startswith(("OPENAI_", "ANTHROPIC_", "CMDSAFESTAN_")):
                pod_env[key] = val

    if mode == "tinker_api":
        tinker_key = os.environ.get("TINKER_API_KEY", "").strip()
        if not tinker_key:
            raise RuntimeError("TINKER_API_KEY not set in environment")
        if tinker_key == "tml-dummy":
            raise RuntimeError("tinker_api mode requires a real TINKER_API_KEY (not tml-dummy)")
        pod_env["TINKER_API_KEY"] = tinker_key

    return pod_env


def build_remote_start_command(
    mode: str,
    *,
    extra_args: list[str],
    backend: str,
    base_model: str,
    api_port: int,
    skyrl_ref: str,
) -> str:
    if mode == "trl":
        run_parts = ["bash", "scripts/runpod/run_grpo_pymc_reward.sh", *extra_args]
    elif mode == "trl_stan":
        run_parts = ["bash", "scripts/runpod/run_grpo_stan_reward.sh", *extra_args]
    else:
        run_parts = [
            "bash",
            "scripts/runpod/run_tinker_api_server.sh",
            "--backend",
            backend,
            "--base-model",
            base_model,
            "--port",
            str(api_port),
            "--skyrl-ref",
            skyrl_ref,
            *extra_args,
        ]
    return f"cd {shlex.quote(REMOTE_DIR)} && {shlex.join(run_parts)}"


def rsync_to_pod(ip: str, port: int, identity_file: str | None = None) -> None:
    """Rsync the repo to the pod with retry for SSH startup race."""
    if not shutil.which("rsync"):
        raise RuntimeError("rsync not found on PATH. Install rsync to continue.")

    exclude_args = []
    for pat in RSYNC_EXCLUDE:
        exclude_args.extend(["--exclude", pat])

    ssh_parts = ["ssh", "-p", str(port), *SSH_OPTIONS]
    if identity_file:
        ssh_parts.extend(["-i", identity_file])
    ssh_transport = shlex.join(ssh_parts)

    cmd = [
        "rsync",
        "-avz",
        "--delete",
        "-e",
        ssh_transport,
        *exclude_args,
        f"{REPO_ROOT}/",
        f"root@{ip}:{REMOTE_DIR}/",
    ]
    print(f"Syncing repo to pod... ({REPO_ROOT} -> {REMOTE_DIR})")

    for attempt in range(1, RSYNC_MAX_RETRIES + 1):
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("Sync complete.")
            return
        if attempt < RSYNC_MAX_RETRIES:
            print(f"Rsync attempt {attempt} failed, retrying in {RSYNC_RETRY_DELAY}s...")
            time.sleep(RSYNC_RETRY_DELAY)

    raise RuntimeError(f"Rsync failed after {RSYNC_MAX_RETRIES} attempts")


def start_remote_tmux(
    ip: str, port: int, inner_command: str, identity_file: str | None = None
) -> None:
    remote_cmd = f"tmux new-session -d -s train {shlex.quote(inner_command)}"
    subprocess.run([*ssh_cmd(ip, port, identity_file), remote_cmd], check=True)
    print("Started tmux session 'train'.")


def run_tinker_smoke(ip: str, api_port: int, timeout: int) -> None:
    script_path = Path(__file__).with_name("smoke_tinker_api.sh")
    subprocess.run(
        ["bash", str(script_path), ip, str(api_port), str(timeout)],
        check=True,
    )


def attach_tmux(ip: str, port: int, identity_file: str | None = None) -> None:
    print("Attaching to tmux session (Ctrl+B, D to detach)...")
    result = subprocess.run([*ssh_cmd(ip, port, identity_file), "-t", "tmux attach -t train"])
    if result.returncode != 0:
        print(f"WARNING: tmux attach exited with code {result.returncode}")


def _try_terminate(pod_id: str) -> None:
    if terminate_pod(pod_id):
        print("Pod terminated.")
    else:
        print(f"WARNING: Failed to terminate pod {pod_id}. Check RunPod dashboard.")
        _print_manual_pod_commands(pod_id)


def _print_manual_pod_commands(pod_id: str) -> None:
    print("Terminate later with:")
    print(
        "  python -c "
        f"\"import os, runpod; runpod.api_key=os.environ['RUNPOD_API_KEY']; "
        f"runpod.terminate_pod('{pod_id}')\""
    )
    print("Check pod status with:")
    print(
        "  python -c "
        f"\"import os, json, runpod; runpod.api_key=os.environ['RUNPOD_API_KEY']; "
        f"print(json.dumps(runpod.get_pod('{pod_id}'), indent=2))\""
    )


def _print_preflight(args: argparse.Namespace) -> None:
    snapshot = collect_preflight_snapshot()
    print("RunPod preflight")
    print(f"- api_key_prefix: {snapshot.get('api_key_prefix')}")

    user = snapshot.get("user")
    if user:
        teams = user.get("teams") or []
        team_str = ", ".join(
            f"{t.get('name') or 'unnamed'}({t.get('id')})" for t in teams if isinstance(t, dict)
        )
        print(f"- account: id={user.get('id')} name={user.get('name')} email={user.get('email')}")
        if team_str:
            print(f"- teams: {team_str}")
    elif snapshot.get("user_error"):
        print(f"- account lookup: unavailable ({snapshot['user_error']})")

    gpus = snapshot.get("gpus") or []
    if gpus:
        gpu_names = [g.get("name") or g.get("id") for g in gpus if isinstance(g, dict)]
        print(f"- visible GPU types: {len(gpus)}")
        print(f"- sample GPUs: {', '.join(str(x) for x in gpu_names[:8])}")
        requested_match = any(
            args.gpu_type in {g.get("id"), g.get("name")} for g in gpus if isinstance(g, dict)
        )
        print(
            f"- requested gpu_type ({args.gpu_type}) visible: "
            f"{'yes' if requested_match else 'no/unknown'}"
        )
    elif snapshot.get("gpus_error"):
        print(f"- gpu catalog: unavailable ({snapshot['gpus_error']})")

    templates = snapshot.get("templates") or []
    if templates:
        print(f"- visible templates: {len(templates)}")
    elif snapshot.get("templates_error"):
        print(f"- template catalog: unavailable ({snapshot['templates_error']})")

    if args.template_id:
        template_match = any(
            args.template_id == t.get("id") for t in templates if isinstance(t, dict)
        )
        print(
            f"- requested template_id ({args.template_id}) visible: "
            f"{'yes' if template_match else 'no/unknown'}"
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    args, extra_args = parse_args()

    validate_runpod_setup()
    if args.preflight_only:
        _print_preflight(args)
        print("Preflight completed. No pod created.")
        return

    pod_env = build_pod_env(args.mode, env_mode=args.env_mode)
    pod_ports = pod_ports_for_mode(args.mode, args.api_port)

    print(f"Creating RunPod pod: {args.name} ({args.gpu_type})")
    if pod_ports:
        print(f"Exposing ports: {pod_ports}")

    pod = create_training_pod(
        name=args.name,
        gpu_type_id=args.gpu_type,
        image_name=args.image,
        container_disk_in_gb=args.disk_gb,
        env=pod_env,
        ports=pod_ports,
        template_id=args.template_id,
        network_volume_id=args.network_volume_id,
    )
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    try:
        print(f"Waiting for SSH (timeout={args.timeout}s)...")
        ip, ssh_port, port_map = wait_for_ssh(pod_id, timeout=args.timeout)
        print(f"SSH ready: root@{ip} -p {ssh_port}")

        rsync_to_pod(ip, ssh_port, identity_file=args.identity_file)
        inner_cmd = build_remote_start_command(
            args.mode,
            extra_args=extra_args,
            backend=args.backend,
            base_model=args.base_model,
            api_port=args.api_port,
            skyrl_ref=args.skyrl_ref,
        )
        start_remote_tmux(ip, ssh_port, inner_cmd, identity_file=args.identity_file)

        if args.mode == "tinker_api" and not args.skip_smoke:
            public_api_port = port_map.get(args.api_port)
            if public_api_port is None:
                raise RuntimeError(
                    f"No public mapping for port {args.api_port}. "
                    f"Available: {port_map}. Use --skip-smoke to bypass."
                )
            run_tinker_smoke(ip, public_api_port, args.smoke_timeout)
            print(f"Smoke passed: http://{ip}:{public_api_port}")

        if not args.no_attach:
            attach_tmux(ip, ssh_port, identity_file=args.identity_file)

        print(f"\nPod {pod_id} is still running (billing continues).")
        answer = input("Terminate pod? [Y/n] ").strip().lower()
        if answer != "n":
            _try_terminate(pod_id)
        else:
            print("Pod kept alive.")
            _print_manual_pod_commands(pod_id)

    except KeyboardInterrupt:
        print(f"\nInterrupted. Auto-terminating pod {pod_id}...")
        _try_terminate(pod_id)
        sys.exit(1)
    except Exception:
        print(f"\nError occurred. Auto-terminating pod {pod_id}...")
        _try_terminate(pod_id)
        raise


if __name__ == "__main__":
    main()
