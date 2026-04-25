"""
Pydantic models for the Flight Rebooking environment.

FlightRebookingAction      — what the agent sends on every step()
FlightRebookingObservation — what the environment returns to the agent
FlightRebookingState       — episode-level metadata (returned by state property)
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class FlightRebookingAction(Action):
    """
    Agent action — a tool call with its named arguments.

    Valid tool_name values:
        list_passengers, get_passenger_details, list_alternative_flights,
        get_flight_details, book_passenger, book_group, unbook_passenger,
        finalize_plan.
    Any other value is routed to the invalid-tool handler inside step().
    """

    tool_name: str = Field(
        ...,
        description=(
            "Tool to call: list_passengers | get_passenger_details | "
            "list_alternative_flights | get_flight_details | "
            "book_passenger | book_group | unbook_passenger | finalize_plan"
        ),
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Named arguments for the tool (values may be str or dict)",
    )


class FlightRebookingObservation(Observation):
    """
    Observation returned to the agent after every reset() and step().

    Inherited from Observation (do not redeclare): done, reward, metadata.
    """

    # Counters
    passengers_total: int = Field(
        default=0,
        description="Total number of passengers needing rebooking",
    )
    passengers_booked: int = Field(
        default=0,
        description="Number of passengers successfully booked so far",
    )
    passengers_remaining: int = Field(
        default=0,
        description="Passengers not yet booked onto an alternative flight",
    )

    # Step feedback
    tool_result: Optional[dict] = Field(
        default=None,
        description="Result dict from the last tool call; None after reset()",
    )
    reward_reason: str = Field(
        default="Episode started",
        description="Human-readable explanation of the reward for the last step",
    )
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=0, description="Step limit for this episode")
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of all rewards received so far in this episode",
    )

    # Summary view (lightweight — not full manifests)
    booked_summary: List[dict] = Field(
        default_factory=list,
        description="List of bookings made so far: [{passenger_id, flight_id, cabin}]",
    )
    flights_snapshot: Optional[List[dict]] = Field(
        default=None,
        description=(
            "Current flight availability snapshot; only populated after "
            "list_alternative_flights has been called"
        ),
    )

    # --- NEW: Decomposed reward breakdown ---
    reward_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Per-component reward breakdown for the last step: "
            "coverage_delta, cabin_match_delta, group_delta, deadline_delta, "
            "ssr_delta, cost_delta, loyalty_delta, opportunity_cost"
        ),
    )

    # --- NEW: Mid-episode events ---
    events: Optional[List[dict]] = Field(
        default=None,
        description=(
            "Mid-episode events that fired on this step (capacity changes, "
            "new passengers, SSR failures, deadline shifts, cancellations)"
        ),
    )

    # --- NEW: Cost tracking ---
    total_cost: float = Field(
        default=0.0,
        description="Total cost incurred so far (upgrades + compensation)",
    )
    compensation_budget: float = Field(
        default=0.0,
        description="Remaining compensation budget for this episode",
    )


class FlightRebookingState(State):
    """
    Episode-level metadata returned by the state property.

    Inherited from State (do not redeclare): episode_id, step_count.
    """

    total_passengers: int = Field(default=0)
    passengers_booked: int = Field(default=0)
    passengers_remaining: int = Field(default=0)
    cumulative_reward: float = Field(default=0.0)
    is_complete: bool = Field(default=False)
    total_cost: float = Field(default=0.0)
    compensation_budget_remaining: float = Field(default=0.0)
