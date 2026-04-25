"""
GRPO-compatible environment wrapper for TRL's GRPOTrainer.

Public methods (except reset) are exposed as tools to the LLM.
Each method must have type hints and Google-style docstrings
for TRL's automatic tool schema extraction.

Usage:
    # In train_grpo.py:
    from training.grpo_env import FlightRebookingGRPOEnv
    trainer = GRPOTrainer(
        ...,
        environment_factory=FlightRebookingGRPOEnv,
    )
"""

from __future__ import annotations

import json
import os
import random
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from server.environment import FlightRebookingEnvironment
from models import FlightRebookingAction


class FlightRebookingGRPOEnv:
    """
    GRPO-compatible environment wrapper.

    Public methods (except reset) are exposed as tools to the LLM.
    Each method must have type hints and Google-style docstrings.
    """

    def __init__(self):
        self._env = FlightRebookingEnvironment()
        self._obs = None
        self._done = False
        self.reward = 0.0
        self.grader_score = 0.0

    def reset(self, *, difficulty: float = 0.5, seed: int = 0, **kwargs) -> str:
        """Reset the environment. Called by GRPOTrainer before each generation."""
        if seed == 0:
            seed = random.randint(1, 100000)

        self._obs = self._env.reset(seed=seed, task_id=f"seed_{seed}")
        self._done = False
        self.reward = 0.0
        self.grader_score = 0.0

        return self._format_state(self._obs)

    def list_passengers(self) -> str:
        """
        List all passengers needing rebooking with summary info.

        Returns:
            JSON string with passenger summaries including ID, priority,
            group, SSR, deadline, loyalty.
        """
        return self._step("list_passengers", {})

    def get_passenger_details(self, passenger_id: str) -> str:
        """
        Get full details for one passenger.

        Args:
            passenger_id: The passenger ID (e.g., "PAX-001").

        Returns:
            JSON string with full passenger record.
        """
        return self._step("get_passenger_details", {"passenger_id": passenger_id})

    def list_alternative_flights(self) -> str:
        """
        List all available alternative flights with cabin availability
        and SSR support.

        Returns:
            JSON string with flight details.
        """
        return self._step("list_alternative_flights", {})

    def get_flight_details(self, flight_id: str) -> str:
        """
        Get details for one specific flight.

        Args:
            flight_id: The flight ID (e.g., "FL-201").

        Returns:
            JSON string with flight details and current availability.
        """
        return self._step("get_flight_details", {"flight_id": flight_id})

    def book_passenger(self, passenger_id: str, flight_id: str, cabin: str) -> str:
        """
        Book one passenger onto a flight in a specific cabin.

        Args:
            passenger_id: The passenger ID to book.
            flight_id: The target flight ID.
            cabin: The cabin class (economy, premium_economy, business).

        Returns:
            JSON string with booking result, cost, and cabin match info.
        """
        return self._step("book_passenger", {
            "passenger_id": passenger_id,
            "flight_id": flight_id,
            "cabin": cabin,
        })

    def book_group(self, group_id: str, flight_id: str, cabin_assignments: str) -> str:
        """
        Book an entire group onto one flight atomically.

        Args:
            group_id: The group ID (e.g., "GRP-001").
            flight_id: The target flight ID.
            cabin_assignments: JSON string mapping passenger_id to cabin
                for each group member.

        Returns:
            JSON string with group booking result and total cost.
        """
        try:
            assignments = (
                json.loads(cabin_assignments)
                if isinstance(cabin_assignments, str)
                else cabin_assignments
            )
        except json.JSONDecodeError:
            return json.dumps({
                "status": "error",
                "message": "Invalid cabin_assignments JSON",
            })
        return self._step("book_group", {
            "group_id": group_id,
            "flight_id": flight_id,
            "cabin_assignments": assignments,
        })

    def unbook_passenger(self, passenger_id: str) -> str:
        """
        Remove an existing booking, freeing the seat back to inventory.

        Args:
            passenger_id: The passenger ID to unbook.

        Returns:
            JSON string with unbook result and cost reversed.
        """
        return self._step("unbook_passenger", {"passenger_id": passenger_id})

    def finalize_plan(self) -> str:
        """
        End the episode and trigger final grading.

        Returns:
            JSON string with grader score and terminal breakdown.
        """
        return self._step("finalize_plan", {})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step(self, tool_name: str, args: dict) -> str:
        """Execute one step in the environment."""
        if self._done:
            return json.dumps({
                "status": "error",
                "message": "Episode already done",
            })

        obs = self._env.step(
            FlightRebookingAction(tool_name=tool_name, args=args)
        )
        self._obs = obs
        self.reward += obs.reward

        if obs.done:
            self._done = True
            if obs.tool_result and "grader_score" in obs.tool_result:
                self.grader_score = obs.tool_result["grader_score"]

        return self._format_result(obs)

    def _format_state(self, obs) -> str:
        """Format observation state as a string for the LLM."""
        parts = [
            f"Step {obs.step_count}/{obs.max_steps} | "
            f"Booked: {obs.passengers_booked}/{obs.passengers_total} | "
            f"Remaining: {obs.passengers_remaining} | "
            f"Cost: ${obs.total_cost:.0f} (budget: ${obs.compensation_budget:.0f})"
        ]

        if obs.events:
            parts.append("\n** EVENTS THIS STEP **")
            for evt in obs.events:
                parts.append(f"  [{evt['type']}] {evt.get('reason', '')}")
            parts.append("** Check bookings and adapt. **")

        if obs.reward_breakdown:
            non_zero = {k: v for k, v in obs.reward_breakdown.items() if v != 0.0}
            if non_zero:
                parts.append(f"\nReward breakdown: {non_zero}")

        if obs.booked_summary:
            parts.append("\nCurrent bookings:")
            for b in obs.booked_summary:
                parts.append(
                    f"  {b['passenger_id']} -> {b['flight_id']} ({b['cabin']})"
                )

        return "\n".join(parts)

    def _format_result(self, obs) -> str:
        """Format tool result + state for the LLM."""
        parts = []
        if obs.tool_result:
            parts.append(json.dumps(obs.tool_result, indent=2))
        parts.append(self._format_state(obs))
        return "\n".join(parts)
