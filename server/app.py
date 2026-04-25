"""
FastAPI application for the Flight Rebooking Environment.

Exposes the FlightRebookingEnvironment over HTTP and WebSocket endpoints
compatible with the OpenEnv EnvClient.

Endpoints (all created automatically by create_app):
  POST /reset   — start a new episode
  POST /step    — execute one tool call
  GET  /state   — current episode metadata
  GET  /schema  — action/observation schemas
  WS   /ws      — WebSocket for persistent sessions
  GET  /health  — liveness check
  GET  /docs    — auto-generated API docs

Usage:
  uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys

# Ensure the repo root is on sys.path so "models" and "server" resolve
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server import create_fastapi_app as create_app  # type: ignore

try:
    from ..models import FlightRebookingAction, FlightRebookingObservation
    from .environment import FlightRebookingEnvironment
except ImportError:
    from models import FlightRebookingAction, FlightRebookingObservation
    from server.environment import FlightRebookingEnvironment


app = create_app(
    FlightRebookingEnvironment,
    FlightRebookingAction,
    FlightRebookingObservation,
    env_name="flight_rebooking",
    max_concurrent_envs=int(os.environ.get("MAX_ENVS", "4")),
)


def main():
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
