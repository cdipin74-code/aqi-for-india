#!/usr/bin/env python3
"""Local development server for India AQI & Early Warning System.

Serves the modern frontend from public/ and routes /api calls to api/index.py.
Run with: python server.py
Access at: http://localhost:8000
"""
from __future__ import annotations

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
API_DIR = ROOT / "api"

sys.path.insert(0, str(API_DIR))
from index import get_report  # noqa: E402


class LocalAQIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api" or parsed.path.startswith("/api/"):
            params = parse_qs(parsed.query)
            city = params.get("city", ["Patna"])[0]
            date_q = params.get("date", ["2024-11-15"])[0]
            state = params.get("state", [""])[0]

            report = get_report(city=city, date_query=date_q, state_hint=state)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(report, indent=2).encode("utf-8"))
            return

        super().do_GET()

    def log_message(self, format, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def run(port: int = 8000):
    server_address = ("127.0.0.1", port)
    try:
        httpd = HTTPServer(server_address, LocalAQIHandler)
    except OSError:
        port = 8080
        server_address = ("127.0.0.1", port)
        httpd = HTTPServer(server_address, LocalAQIHandler)

    print("\n" + "=" * 55)
    print(f"  [*] India AQI Web Dashboard running locally at:")
    print(f"  --> http://localhost:{port}")
    print("=" * 55 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run(port)
