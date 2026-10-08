"""Entry point for the Vector Lab bridge when run as the desktop app's child.

The web workflow keeps launching the bridge with `python -m uvicorn
backend.server:app`; this module exists for the Electron desktop app, where the
bridge is a spawned child process — source-run in development, a frozen
PyInstaller executable when packaged. All configuration arrives through the
VECTOR_LAB_* environment variables documented in server.py's module docstring,
The child binds its own loopback socket, announces readiness, and watches the
parent's pipe so it can shut down even when Electron crashes.
"""
from __future__ import annotations

import os
import json
import socket
import sys
import threading

import uvicorn


def main() -> None:
    # Running `python backend/main.py` puts backend/ (not the project root) on
    # sys.path; restore the root so the absolute `backend.server` import works
    # in source runs. Frozen builds resolve it from the bundle instead.
    if not getattr(sys, "frozen", False):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
    # Imported after the VECTOR_LAB_* environment is guaranteed to be set.
    from backend import server as bridge

    port = int(os.environ.get("VECTOR_LAB_PORT", "8788"))
    # Bind once and retain the socket. Port zero is selected atomically by the
    # OS; Electron never connects to or terminates an unrelated listener.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", port))

    class DesktopServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            if self.started:
                print(json.dumps({"type": "vectorlab-ready", "port": sock.getsockname()[1],
                                  "instance": os.environ.get("VECTOR_LAB_INSTANCE_ID", "")}), flush=True)

    service = DesktopServer(uvicorn.Config(bridge.app, host="127.0.0.1", port=port,
        log_level="warning", access_log=False, loop="asyncio", http="h11", ws="none",
        timeout_graceful_shutdown=5))

    def watch_parent():
        # The main process keeps stdin open for the lifetime of its child.
        # EOF means either an orderly quit or a crashed parent, on every OS.
        sys.stdin.buffer.read()
        bridge.stop()
        service.should_exit = True

    if os.environ.get("VECTOR_LAB_PARENT_PIPE") == "1":
        threading.Thread(target=watch_parent, daemon=True).start()
    try:
        service.run(sockets=[sock])
    finally:
        sock.close()


if __name__ == "__main__":
    main()
