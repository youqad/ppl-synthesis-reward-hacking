"""Tests for tinker_keepalive module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ppl_synthesis_reward_hacking.experiments.tinker_keepalive import (
    _post_heartbeat,
    _resolve_session_id,
    check_keepalive_status,
)


class TestResolveSessionId:
    def test_extracts_from_holder(self):
        sc = MagicMock()
        sc.holder._session_id = "sess-abc-123"
        assert _resolve_session_id(sc) == "sess-abc-123"

    def test_raises_if_no_holder(self):
        sc = MagicMock(spec=[])  # no holder attr
        with pytest.raises(RuntimeError, match="no holder"):
            _resolve_session_id(sc)

    def test_raises_if_no_session_id(self):
        sc = MagicMock()
        sc.holder._session_id = ""
        with pytest.raises(RuntimeError, match="not set"):
            _resolve_session_id(sc)


class TestCheckKeepaliveStatus:
    def test_returns_none_for_missing_file(self, tmp_path: Path):
        assert check_keepalive_status(tmp_path / "nope.jsonl") is None

    def test_returns_none_for_empty_file(self, tmp_path: Path):
        f = tmp_path / "status.jsonl"
        f.write_text("")
        assert check_keepalive_status(f) is None

    def test_reads_last_status(self, tmp_path: Path):
        f = tmp_path / "status.jsonl"
        f.write_text(
            json.dumps({"status": "warn", "ts": 1.0})
            + "\n"
            + json.dumps({"status": "dead", "ts": 2.0})
            + "\n"
        )
        assert check_keepalive_status(f) == "dead"

    def test_returns_warn(self, tmp_path: Path):
        f = tmp_path / "status.jsonl"
        f.write_text(json.dumps({"status": "warn", "ts": 1.0}) + "\n")
        assert check_keepalive_status(f) == "warn"


class TestPostHeartbeat:
    @patch("ppl_synthesis_reward_hacking.experiments.tinker_keepalive.urllib.request.urlopen")
    def test_sends_correct_request(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        status = _post_heartbeat(
            base_url="https://example.com/services/tinker-prod",
            api_key="test-key",
            session_id="sess-123",
            timeout_s=10.0,
        )
        assert status == 200

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url == "https://example.com/services/tinker-prod/api/v1/session_heartbeat"
        assert req.get_header("X-api-key") == "test-key"
        body = json.loads(req.data)
        assert body == {"session_id": "sess-123", "type": "session_heartbeat"}
