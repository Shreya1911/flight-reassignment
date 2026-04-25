"""Flight Rebooking Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import FlightRebookingAction, FlightRebookingObservation, FlightRebookingState
except ImportError:
    from models import FlightRebookingAction, FlightRebookingObservation, FlightRebookingState


class FlightRebookingEnv(EnvClient[FlightRebookingAction, FlightRebookingObservation, FlightRebookingState]):
    """
    WebSocket client for the Flight Rebooking Environment.

    Example:
        >>> with FlightRebookingEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     obs = result.observation
        ...     print(obs.passengers_remaining)
        ...
        ...     action = FlightRebookingAction(
        ...         tool_name="list_passengers",
        ...         args={},
        ...     )
        ...     result = env.step(action)
        ...     print(result.observation.tool_result)
    """

    def _step_payload(self, action: FlightRebookingAction) -> Dict:
        return {"tool_name": action.tool_name, "args": action.args}

    def _parse_result(self, payload: Dict) -> StepResult[FlightRebookingObservation]:
        obs_data = payload.get("observation", payload)
        obs = FlightRebookingObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> FlightRebookingState:
        return FlightRebookingState(**payload)
