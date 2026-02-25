"""Out-of-band Tinker session keepalive using a separate OS process.

The Tinker SDK runs its heartbeat as an asyncio task in a background thread.
Under CPU contention (8+ concurrent scoring runs on Contabo), that thread
gets starved, heartbeat HTTP requests timeout, and the Tinker server
terminates the session after ~160s of silence.

This module spawns a dedicated multiprocessing.Process that sends heartbeats
via raw HTTP, bypassing the GIL and asyncio contention entirely.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
import urllib.error
import urllib.request
from multiprocessing import Process
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod"

# log a success entry every N heartbeats to avoid log bloat
_SUCCESS_LOG_EVERY = 60  # at 5s interval = every 5 min


def _post_heartbeat(*, base_url: str, api_key: str, session_id: str, timeout_s: float) -> int:
    """Send a single heartbeat ping. Returns HTTP status code."""
    url = f"{base_url.rstrip('/')}/api/v1/session_heartbeat"
    body = json.dumps({"session_id": session_id, "type": "session_heartbeat"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": api_key,
            "User-Agent": "tinker-python/keepalive",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.status


def _keepalive_loop(
    *,
    base_url: str,
    api_key: str,
    session_id: str,
    interval_s: float,
    http_timeout_s: float,
    warn_after_s: float,
    dead_after_s: float,
    status_path: str,
) -> None:
    """Runs in a child process. Pings heartbeat, writes status to JSONL file."""
    last_ok = time.monotonic()
    consecutive_failures = 0
    total_beats = 0
    total_ok = 0

    def _write(entry: dict) -> None:
        with open(status_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    while True:
        ok = False
        err_detail = ""
        try:
            status = _post_heartbeat(
                base_url=base_url,
                api_key=api_key,
                session_id=session_id,
                timeout_s=http_timeout_s,
            )
            ok = status == 200
            if not ok:
                err_detail = f"http_{status}"
        except urllib.error.HTTPError as exc:
            err_detail = f"http_{exc.code}"
        except urllib.error.URLError as exc:
            err_detail = f"url_error:{exc.reason}"
        except TimeoutError:
            err_detail = "timeout"
        except Exception as exc:
            err_detail = f"{type(exc).__name__}:{exc}"

        total_beats += 1
        now = time.monotonic()

        if ok:
            last_ok = now
            consecutive_failures = 0
            total_ok += 1
            if total_beats % _SUCCESS_LOG_EVERY == 0:
                _write(
                    {
                        "ts": time.time(),
                        "status": "ok",
                        "total_beats": total_beats,
                        "total_ok": total_ok,
                    }
                )
        else:
            consecutive_failures += 1
            dt = now - last_ok

            # log every failure so we can diagnose
            _write(
                {
                    "ts": time.time(),
                    "status": "warn" if dt > warn_after_s else "fail",
                    "dt_since_ok_s": round(dt, 1),
                    "consecutive_failures": consecutive_failures,
                    "error": err_detail,
                    "total_beats": total_beats,
                }
            )

            if dt > dead_after_s:
                _write(
                    {
                        "ts": time.time(),
                        "status": "dead",
                        "dt_since_ok_s": round(dt, 1),
                        "consecutive_failures": consecutive_failures,
                        "error": err_detail,
                    }
                )
                try:
                    os.kill(os.getppid(), signal.SIGUSR1)
                except (ProcessLookupError, OSError):
                    pass
                return

        time.sleep(interval_s)


def _resolve_session_id(service_client) -> str:
    """Extract session_id from the SDK's InternalClientHolder."""
    # service_client.holder is the InternalClientHolder that owns the session
    holder = getattr(service_client, "holder", None)
    if holder is None:
        raise RuntimeError("ServiceClient has no holder attribute")
    sid = getattr(holder, "_session_id", None)
    if not sid:
        raise RuntimeError("InternalClientHolder._session_id not set")
    return sid


def check_keepalive_status(status_path: str | Path) -> str | None:
    """Read the last keepalive status line. Returns 'dead', 'warn', or None."""
    path = Path(status_path)
    if not path.exists():
        return None
    try:
        last_line = path.read_text().strip().rsplit("\n", 1)[-1]
        if not last_line:
            return None
        return json.loads(last_line).get("status")
    except (json.JSONDecodeError, IndexError):
        return None


def start_keepalive(
    service_client,
    output_dir: str | Path,
    *,
    interval_s: float = 5.0,
    http_timeout_s: float = 10.0,
    warn_after_s: float = 30.0,
    dead_after_s: float = 60.0,
) -> Process:
    """Spawn a keepalive process for the Tinker session. Caller must .terminate() on exit.

    dead_after_s MUST be less than the training-update timeout (120s) so the
    keepalive can signal SIGUSR1 before the training path raises TinkerSessionDead.
    """
    api_key = os.environ.get("TINKER_API_KEY", "")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY not set; cannot start keepalive")

    base_url = os.environ.get("TINKER_BASE_URL", _DEFAULT_BASE_URL)
    session_id = _resolve_session_id(service_client)
    status_path = str(Path(output_dir) / "tinker_keepalive.jsonl")

    # verify heartbeat works before spawning long-lived process
    try:
        probe_status = _post_heartbeat(
            base_url=base_url,
            api_key=api_key,
            session_id=session_id,
            timeout_s=http_timeout_s,
        )
        log.info("Keepalive probe: session %s, HTTP %d", session_id, probe_status)
    except Exception as exc:
        log.warning("Keepalive probe failed (will start anyway): %s", exc)

    log.info(
        "Starting out-of-band keepalive for session %s (interval=%.0fs, warn=%.0fs, dead=%.0fs)",
        session_id,
        interval_s,
        warn_after_s,
        dead_after_s,
    )

    p = Process(
        target=_keepalive_loop,
        kwargs={
            "base_url": base_url,
            "api_key": api_key,
            "session_id": session_id,
            "interval_s": interval_s,
            "http_timeout_s": http_timeout_s,
            "warn_after_s": warn_after_s,
            "dead_after_s": dead_after_s,
            "status_path": status_path,
        },
        daemon=True,
        name="tinker-keepalive",
    )
    p.start()
    log.info("Keepalive process started (pid=%d)", p.pid)
    return p
