"""Tests for the RunPod Tinker API smoke script."""

from __future__ import annotations

import http.server
import json
import socketserver
import subprocess
import threading
from pathlib import Path


class _OpenAPIHandler(http.server.BaseHTTPRequestHandler):
    """Serve a minimal OpenAPI document for smoke checks."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/openapi.json":
            self.send_response(404)
            self.end_headers()
            return

        payload = {"openapi": "3.1.0", "paths": {}}
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def test_smoke_tinker_api_script_accepts_valid_openapi():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "runpod" / "smoke_tinker_api.sh"

    with socketserver.TCPServer(("127.0.0.1", 0), _OpenAPIHandler) as server:
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            proc = subprocess.run(
                ["bash", str(script), "127.0.0.1", str(port), "5"],
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Smoke OK" in proc.stdout
