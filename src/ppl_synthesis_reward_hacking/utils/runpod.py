"""RunPod GPU cloud availability and pod management."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

try:
    import runpod

    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False
    runpod = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

DEFAULT_GPU_TYPE = "NVIDIA A100 80GB PCIe"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_DISK_GB = 50
SSH_POLL_INTERVAL = 10
TERMINAL_STATES = frozenset({"EXITED", "TERMINATED", "FAILED"})


def validate_runpod_setup() -> None:
    """Check that the RunPod SDK is installed and API key is set."""
    if not RUNPOD_AVAILABLE:
        raise RuntimeError("runpod SDK not installed. Install with: pip install 'runpod>=1.7.0'")
    if not os.environ.get("RUNPOD_API_KEY"):
        raise RuntimeError("RUNPOD_API_KEY not set in environment")
    runpod.api_key = os.environ["RUNPOD_API_KEY"]


def create_training_pod(
    name: str,
    gpu_type_id: str = DEFAULT_GPU_TYPE,
    image_name: str = DEFAULT_IMAGE,
    container_disk_in_gb: int = DEFAULT_DISK_GB,
    env: dict[str, str] | None = None,
    ports: str | None = None,
    start_ssh: bool = True,
    template_id: str | None = None,
    network_volume_id: str | None = None,
) -> dict:
    """Create a RunPod pod configured for GRPO training.

    Returns the pod dict from the RunPod API. Raises RuntimeError if the
    response doesn't contain a pod ID.
    """
    validate_runpod_setup()
    create_kwargs = {
        "name": name,
        "cloud_type": "ALL",
        "support_public_ip": True,
        "start_ssh": start_ssh,
        "gpu_count": 1,
        "env": env or {},
    }
    if template_id:
        # RunPod template-backed pod creation can be rejected when mixed with
        # explicit image/GPU/disk fields; omit those fields when template_id is used.
        if (
            image_name != DEFAULT_IMAGE
            or gpu_type_id != DEFAULT_GPU_TYPE
            or container_disk_in_gb != DEFAULT_DISK_GB
        ):
            raise ValueError(
                "template_id is incompatible with explicit image/gpu/disk overrides; "
                "configure these in the template instead"
            )
        create_kwargs["template_id"] = template_id
    else:
        create_kwargs.update(
            {
                "image_name": image_name,
                "gpu_type_id": gpu_type_id,
                "container_disk_in_gb": container_disk_in_gb,
                "volume_in_gb": 0,
            }
        )
    if ports:
        create_kwargs["ports"] = ports
    if network_volume_id:
        create_kwargs["network_volume_id"] = network_volume_id

    pod = runpod.create_pod(
        **create_kwargs,
    )
    if not isinstance(pod, dict) or "id" not in pod:
        raise RuntimeError(f"RunPod create_pod returned unexpected response: {pod}")
    log.info("Created pod %s (id=%s)", name, pod["id"])
    return pod


def wait_for_ssh(pod_id: str, timeout: float = 300) -> tuple[str, int, dict[int, int]]:
    """Poll until pod is RUNNING and return (public_ip, ssh_port, port_map).

    port_map is a dict mapping private_port -> public_port for all public
    port mappings reported by RunPod (e.g. {22: 43217, 8000: 51234}).

    Raises TimeoutError if the pod doesn't reach RUNNING within timeout seconds.
    Raises RuntimeError if the pod enters a terminal state (EXITED/TERMINATED/FAILED).
    """
    validate_runpod_setup()
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "UNKNOWN")

        if status in TERMINAL_STATES:
            raise RuntimeError(f"Pod {pod_id} entered terminal state: {status}")

        runtime = pod.get("runtime")
        if status == "RUNNING" and runtime:
            ports = runtime.get("ports") or []
            port_map: dict[int, int] = {}
            ssh_ip = None
            ssh_public_port = None

            for port_info in ports:
                if not port_info.get("isIpPublic"):
                    continue
                priv = port_info.get("privatePort")
                pub = port_info.get("publicPort")
                if priv is not None and pub is not None:
                    port_map[priv] = pub
                    if priv == 22:
                        ssh_ip = port_info.get("ip")
                        ssh_public_port = pub

            if ssh_ip and ssh_public_port:
                log.info("Pod %s SSH ready: %s:%d", pod_id, ssh_ip, ssh_public_port)
                return (ssh_ip, ssh_public_port, port_map)

        log.debug("Pod %s status=%s, waiting...", pod_id, status)
        time.sleep(SSH_POLL_INTERVAL)

    raise TimeoutError(f"Pod {pod_id} did not become SSH-ready within {timeout}s")


def terminate_pod(pod_id: str) -> bool:
    """Terminate a pod. Returns True on success, False on failure.

    Logs errors at CRITICAL level since failed termination means ongoing billing.
    """
    validate_runpod_setup()
    try:
        runpod.terminate_pod(pod_id)
        log.info("Terminated pod %s", pod_id)
        return True
    except Exception:
        log.critical("FAILED to terminate pod %s — CHECK RUNPOD BILLING", pod_id, exc_info=True)
        return False


def get_pod_status(pod_id: str) -> str:
    """Fetch desiredStatus from RunPod API, defaulting to 'UNKNOWN'."""
    validate_runpod_setup()
    pod = runpod.get_pod(pod_id)
    return pod.get("desiredStatus", "UNKNOWN")


def collect_preflight_snapshot() -> dict[str, Any]:
    """Return a best-effort RunPod capability snapshot for launch preflight checks.

    This is intentionally resilient across RunPod SDK versions: optional endpoints
    (user/gpu/template catalog) are queried only when available and errors are
    captured in the returned payload instead of raising.
    """
    validate_runpod_setup()
    snapshot: dict[str, Any] = {
        "api_key_set": True,
        "api_key_prefix": os.environ["RUNPOD_API_KEY"][:6],
        "user": None,
        "user_error": None,
        "gpus": [],
        "gpus_error": None,
        "templates": [],
        "templates_error": None,
    }

    user_raw, user_err = _optional_sdk_call("get_user")
    snapshot["user_error"] = user_err
    snapshot["user"] = _normalize_user_payload(user_raw)

    gpus_raw, gpus_err = _optional_sdk_call("get_gpus")
    snapshot["gpus_error"] = gpus_err
    snapshot["gpus"] = _normalize_gpu_payload(gpus_raw)

    templates_raw, templates_err = _optional_sdk_call("get_templates")
    snapshot["templates_error"] = templates_err
    snapshot["templates"] = _normalize_template_payload(templates_raw)

    return snapshot


def _optional_sdk_call(method_name: str) -> tuple[Any | None, str | None]:
    fn = getattr(runpod, method_name, None)
    if fn is None:
        return None, f"{method_name} not available in installed runpod SDK"
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _normalize_user_payload(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    user = raw.get("user", raw)
    if not isinstance(user, dict):
        return None
    teams_value = user.get("teams")
    teams_raw = teams_value if isinstance(teams_value, list) else []
    teams: list[dict[str, Any]] = []
    for team in teams_raw:
        if isinstance(team, dict):
            teams.append(
                {
                    "id": team.get("id"),
                    "name": team.get("name"),
                }
            )
    return {
        "id": user.get("id"),
        "email": user.get("email"),
        "name": user.get("name"),
        "teams": teams,
    }


def _normalize_gpu_payload(raw: Any) -> list[dict[str, Any]]:
    items = _coerce_list_payload(raw, keys=("gpus", "gpuTypes", "gpu_types"))
    normalized: list[dict[str, Any]] = []
    for gpu in items:
        if not isinstance(gpu, dict):
            continue
        normalized.append(
            {
                "id": gpu.get("id") or gpu.get("gpuTypeId"),
                "name": gpu.get("displayName") or gpu.get("name") or gpu.get("gpuType"),
            }
        )
    return normalized


def _normalize_template_payload(raw: Any) -> list[dict[str, Any]]:
    items = _coerce_list_payload(raw, keys=("templates",))
    normalized: list[dict[str, Any]] = []
    for tpl in items:
        if not isinstance(tpl, dict):
            continue
        normalized.append(
            {
                "id": tpl.get("id"),
                "name": tpl.get("name"),
            }
        )
    return normalized


def _coerce_list_payload(raw: Any, *, keys: tuple[str, ...]) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in keys:
            value = raw.get(key)
            if isinstance(value, list):
                return value
    return []
