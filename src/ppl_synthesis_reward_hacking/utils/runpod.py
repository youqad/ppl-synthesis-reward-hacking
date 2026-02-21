"""RunPod GPU cloud availability and pod management."""

from __future__ import annotations

import logging
import os
import time

try:
    import runpod

    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False
    runpod = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

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
) -> dict:
    """Create a RunPod pod configured for GRPO training.

    Returns the pod dict from the RunPod API. Raises RuntimeError if the
    response doesn't contain a pod ID.
    """
    validate_runpod_setup()
    create_kwargs = {
        "name": name,
        "image_name": image_name,
        "gpu_type_id": gpu_type_id,
        "cloud_type": "ALL",
        "support_public_ip": True,
        "start_ssh": start_ssh,
        "gpu_count": 1,
        "container_disk_in_gb": container_disk_in_gb,
        "volume_in_gb": 0,
        "env": env or {},
    }
    if ports:
        create_kwargs["ports"] = ports

    pod = runpod.create_pod(
        **create_kwargs,
    )
    if not isinstance(pod, dict) or "id" not in pod:
        raise RuntimeError(f"RunPod create_pod returned unexpected response: {pod}")
    logger.info("Created pod %s (id=%s)", name, pod["id"])
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
                logger.info("Pod %s SSH ready: %s:%d", pod_id, ssh_ip, ssh_public_port)
                return (ssh_ip, ssh_public_port, port_map)

        logger.debug("Pod %s status=%s, waiting...", pod_id, status)
        time.sleep(SSH_POLL_INTERVAL)

    raise TimeoutError(f"Pod {pod_id} did not become SSH-ready within {timeout}s")


def terminate_pod(pod_id: str) -> bool:
    """Terminate a pod. Returns True on success, False on failure.

    Logs errors at CRITICAL level since failed termination means ongoing billing.
    """
    validate_runpod_setup()
    try:
        runpod.terminate_pod(pod_id)
        logger.info("Terminated pod %s", pod_id)
        return True
    except Exception:
        logger.critical("FAILED to terminate pod %s — CHECK RUNPOD BILLING", pod_id, exc_info=True)
        return False


def get_pod_status(pod_id: str) -> str:
    """Fetch desiredStatus from RunPod API, defaulting to 'UNKNOWN'."""
    validate_runpod_setup()
    pod = runpod.get_pod(pod_id)
    return pod.get("desiredStatus", "UNKNOWN")
