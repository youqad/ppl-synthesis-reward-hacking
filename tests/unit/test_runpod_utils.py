from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ppl_synthesis_reward_hacking.utils.runpod import (
    collect_preflight_snapshot,
    create_training_pod,
    get_pod_status,
    terminate_pod,
    validate_runpod_setup,
    wait_for_ssh,
)

MOD = "ppl_synthesis_reward_hacking.utils.runpod"


@pytest.fixture()
def rpod():
    """Patch RunPod SDK, env key, and availability flag for all standard tests."""
    m = MagicMock()
    with (
        patch(f"{MOD}.RUNPOD_AVAILABLE", True),
        patch(f"{MOD}.runpod", m),
        patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"}),
    ):
        yield m


def test_validate_missing_sdk():
    with patch(f"{MOD}.RUNPOD_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="runpod SDK not installed"):
            validate_runpod_setup()


def test_validate_missing_api_key():
    with (
        patch(f"{MOD}.RUNPOD_AVAILABLE", True),
        patch.dict("os.environ", {}, clear=True),
    ):
        with pytest.raises(RuntimeError, match="RUNPOD_API_KEY not set"):
            validate_runpod_setup()


def test_validate_success(rpod):
    validate_runpod_setup()
    assert rpod.api_key == "test-key"


def test_create_pod_calls_sdk(rpod):
    rpod.create_pod.return_value = {"id": "pod-abc123"}

    result = create_training_pod(
        name="test-pod",
        gpu_type_id="NVIDIA A100 80GB PCIe",
        env={"WANDB_API_KEY": "wk-xxx"},
    )

    assert result["id"] == "pod-abc123"
    rpod.create_pod.assert_called_once()
    kw = rpod.create_pod.call_args[1]
    assert kw["name"] == "test-pod"
    assert kw["support_public_ip"] is True
    assert kw["start_ssh"] is True
    assert kw["env"] == {"WANDB_API_KEY": "wk-xxx"}


def test_create_pod_rejects_bad_response(rpod):
    rpod.create_pod.return_value = {"error": "quota exceeded"}

    with pytest.raises(RuntimeError, match="unexpected response"):
        create_training_pod(name="bad-pod")


def test_create_pod_forwards_ports(rpod):
    rpod.create_pod.return_value = {"id": "p1"}

    result = create_training_pod(
        name="tinker-pod",
        ports="22/tcp,8000/http",
        env={"TINKER_API_KEY": "real-key"},
    )

    assert result["id"] == "p1"
    kw = rpod.create_pod.call_args[1]
    assert kw["ports"] == "22/tcp,8000/http"


def test_create_pod_forwards_template_and_network_volume(rpod):
    rpod.create_pod.return_value = {"id": "p2"}

    result = create_training_pod(
        name="template-pod",
        template_id="tpl_123",
        network_volume_id="vol_456",
    )

    assert result["id"] == "p2"
    kw = rpod.create_pod.call_args[1]
    assert kw["template_id"] == "tpl_123"
    assert kw["network_volume_id"] == "vol_456"
    assert "image_name" not in kw
    assert "gpu_type_id" not in kw
    assert "container_disk_in_gb" not in kw


def test_create_pod_rejects_template_with_explicit_runtime_overrides(rpod):
    with pytest.raises(ValueError, match="template_id is incompatible with explicit"):
        create_training_pod(
            name="bad-template-pod",
            template_id="tpl_123",
            image_name="custom/image",
        )


def test_ssh_extracts_port(rpod):
    rpod.get_pod.return_value = {
        "desiredStatus": "RUNNING",
        "runtime": {
            "ports": [
                {
                    "ip": "194.26.196.42",
                    "isIpPublic": True,
                    "privatePort": 22,
                    "publicPort": 43217,
                    "type": "tcp",
                },
            ]
        },
    }

    with patch(f"{MOD}.SSH_POLL_INTERVAL", 0.01):
        ip, ssh_port, port_map = wait_for_ssh("pod-abc123", timeout=30)
        assert ip == "194.26.196.42"
        assert ssh_port == 43217
        assert port_map == {22: 43217}


def test_ssh_returns_full_port_map(rpod):
    rpod.get_pod.return_value = {
        "desiredStatus": "RUNNING",
        "runtime": {
            "ports": [
                {
                    "ip": "194.26.196.42",
                    "isIpPublic": True,
                    "privatePort": 22,
                    "publicPort": 43217,
                    "type": "tcp",
                },
                {
                    "ip": "194.26.196.42",
                    "isIpPublic": True,
                    "privatePort": 8000,
                    "publicPort": 51234,
                    "type": "http",
                },
                {
                    "ip": "10.0.0.5",
                    "isIpPublic": False,
                    "privatePort": 3000,
                    "publicPort": 3000,
                    "type": "tcp",
                },
            ]
        },
    }

    with patch(f"{MOD}.SSH_POLL_INTERVAL", 0.01):
        ip, ssh_port, port_map = wait_for_ssh("pod-multi", timeout=30)
        assert ip == "194.26.196.42"
        assert ssh_port == 43217
        assert port_map == {22: 43217, 8000: 51234}
        assert 3000 not in port_map  # non-public ports excluded


def test_ssh_timeout(rpod):
    rpod.get_pod.return_value = {"desiredStatus": "CREATED", "runtime": None}

    with patch(f"{MOD}.SSH_POLL_INTERVAL", 0.01):
        with pytest.raises(TimeoutError, match="did not become SSH-ready"):
            wait_for_ssh("p2", timeout=0.05)


def test_ssh_handles_null_ports(rpod):
    """runtime.ports can be None from API."""
    rpod.get_pod.return_value = {
        "desiredStatus": "RUNNING",
        "runtime": {"ports": None},
    }

    with patch(f"{MOD}.SSH_POLL_INTERVAL", 0.01):
        with pytest.raises(TimeoutError, match="did not become SSH-ready"):
            wait_for_ssh("p3", timeout=0.05)


def test_ssh_fails_fast_on_terminal_state(rpod):
    rpod.get_pod.return_value = {"desiredStatus": "EXITED", "runtime": None}

    with pytest.raises(RuntimeError, match="terminal state: EXITED"):
        wait_for_ssh("p4", timeout=30)


def test_terminate_returns_true(rpod):
    assert terminate_pod("pod-123") is True
    rpod.terminate_pod.assert_called_once_with("pod-123")


def test_terminate_returns_false_on_failure(rpod):
    rpod.terminate_pod.side_effect = Exception("network error")
    assert terminate_pod("p5") is False


def test_status_returns_desired(rpod):
    rpod.get_pod.return_value = {"desiredStatus": "RUNNING"}
    assert get_pod_status("pod-123") == "RUNNING"


def test_status_returns_unknown_when_missing(rpod):
    rpod.get_pod.return_value = {}
    assert get_pod_status("p6") == "UNKNOWN"


def test_collect_preflight_snapshot_with_catalog_data(rpod):
    rpod.get_user.return_value = {
        "user": {
            "id": "u_1",
            "name": "Uni Account",
            "email": "user@example.edu",
            "teams": [{"id": "team_1", "name": "Oxford Lab"}],
        }
    }
    rpod.get_gpus.return_value = {
        "gpus": [{"id": "NVIDIA A100 80GB PCIe", "displayName": "NVIDIA A100 80GB PCIe"}]
    }
    rpod.get_templates.return_value = {"templates": [{"id": "tpl_1", "name": "Stan Template"}]}

    snapshot = collect_preflight_snapshot()

    assert snapshot["api_key_set"] is True
    assert snapshot["api_key_prefix"] == "test-k"
    assert snapshot["user"]["id"] == "u_1"
    assert snapshot["user"]["teams"][0]["id"] == "team_1"
    assert snapshot["gpus"][0]["id"] == "NVIDIA A100 80GB PCIe"
    assert snapshot["templates"][0]["id"] == "tpl_1"
    assert snapshot["user_error"] is None
    assert snapshot["gpus_error"] is None
    assert snapshot["templates_error"] is None


def test_collect_preflight_snapshot_handles_missing_optional_methods(rpod):
    for name in ("get_user", "get_gpus", "get_templates"):
        if hasattr(rpod, name):
            delattr(rpod, name)

    snapshot = collect_preflight_snapshot()
    assert snapshot["user"] is None
    assert "not available" in snapshot["user_error"]
    assert snapshot["gpus"] == []
    assert "not available" in snapshot["gpus_error"]
    assert snapshot["templates"] == []
    assert "not available" in snapshot["templates_error"]
