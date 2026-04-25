"""
Core environment implementation for the Flight Rebooking task.

A flight has been cancelled. The agent must rebook passengers onto
alternative flights using 8 tools:
  - list_passengers           -> survey all passengers
  - get_passenger_details     -> inspect one passenger
  - list_alternative_flights  -> survey flight inventory
  - get_flight_details        -> inspect one flight
  - book_passenger            -> commit one passenger to a flight/cabin
  - book_group                -> commit an entire group atomically
  - unbook_passenger          -> undo a booking (frees seat, incurs cost)
  - finalize_plan             -> end the episode and trigger grading

Features:
  - Stochastic mid-episode events (capacity changes, new passengers,
    SSR equipment failures, deadline shifts, secondary cancellations)
  - Tighter step budgets (~2.0-2.5 steps per passenger at high difficulty)
  - Cost tracking (upgrade costs, downgrade compensation, loyalty entitlements)
  - Decomposed reward breakdowns per step
  - Opportunity cost signaling
  - Procedural data generation via seed
  - Progressive difficulty scaling
"""

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        FlightRebookingAction,
        FlightRebookingObservation,
        FlightRebookingState,
    )
    from .tools import (
        tool_list_passengers,
        tool_get_passenger_details,
        tool_list_alternative_flights,
        tool_get_flight_details,
        tool_book_passenger,
        tool_book_group,
        tool_unbook_passenger,
        tool_finalize_plan,
    )
    from .rewards import RewardComputer, REWARD_FINALIZE, REWARD_REPEATED_CALL
except ImportError:
    from models import (
        FlightRebookingAction,
        FlightRebookingObservation,
        FlightRebookingState,
    )
    from server.tools import (
        tool_list_passengers,
        tool_get_passenger_details,
        tool_list_alternative_flights,
        tool_get_flight_details,
        tool_book_passenger,
        tool_book_group,
        tool_unbook_passenger,
        tool_finalize_plan,
    )
    from server.rewards import RewardComputer, REWARD_FINALIZE, REWARD_REPEATED_CALL


# ---------------------------------------------------------------------------
# Internal episode state (not exposed to the agent)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    # Immutable data loaded from files
    passengers: Dict[str, dict] = field(default_factory=dict)
    flights: Dict[str, dict] = field(default_factory=dict)
    groups: Dict[str, List[str]] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    # Mutable booking state
    bookings: Dict[str, dict] = field(default_factory=dict)
    flight_availability: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Tracking for reward computation
    info_calls: Dict[str, int] = field(default_factory=dict)
    last_booking_step: int = 0
    passenger_details_fetched: Set[str] = field(default_factory=set)

    # Whether list_alternative_flights has been called (for snapshot in obs)
    flights_listed: bool = False

    # Episode metadata
    task_id: str = ""
    step_count: int = 0
    max_steps: int = 0
    cumulative_reward: float = 0.0
    done: bool = False

    # --- NEW: Cost tracking ---
    total_cost: float = 0.0
    compensation_budget: float = 0.0

    # --- NEW: Mid-episode events ---
    pending_events: List[dict] = field(default_factory=list)
    fired_events_log: List[dict] = field(default_factory=list)

    # --- NEW: Cancelled flights (secondary cancellations) ---
    cancelled_flights: Set[str] = field(default_factory=set)

    # --- NEW: Unbook tracking ---
    unbook_count: int = 0

    # --- NEW: Event tracking for current step ---
    events_this_step: List[dict] = field(default_factory=list)

    # --- NEW: Difficulty level ---
    difficulty: float = 0.5

    # --- NEW: Repeated tool call tracking ---
    # Stores recent consecutive (tool_name, args) pairs for repetition detection
    recent_tool_calls: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FlightRebookingEnvironment(Environment):
    """
    OpenEnv environment for airline flight rebooking after cancellation.

    An episode consists of an agent rebooking passengers from a cancelled
    flight onto alternative flights using 8 tools. The episode ends when
    finalize_plan is called, all passengers are booked, or the step limit
    is reached.

    Supports:
    - Static data loading from data/{task_id}/ directories
    - Procedural generation via seed parameter
    - Mid-episode stochastic events
    - Cost tracking and loyalty compliance
    - Decomposed reward feedback
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: Optional[EpisodeState] = None
        self._reward_computer: Optional[RewardComputer] = None
        self._data_dir = Path(__file__).resolve().parent.parent / "data"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, task_id: str = "medium"
    ) -> FlightRebookingObservation:
        """Load data, build lookup structures, and return the initial observation."""
        episode_id = str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)

        # --- Decide data source: procedural vs static ---
        if seed is not None and task_id.startswith("seed_"):
            # Procedural generation
            passengers_list, flights_list, config, pending_events = (
                self._generate_procedural(seed, task_id)
            )
        elif seed is not None:
            # Static task_id but with a seed: use procedural with difficulty
            # mapped from task_id
            difficulty_map = {"easy": 0.2, "medium": 0.5, "hard": 0.8}
            diff = difficulty_map.get(task_id, 0.5)
            passengers_list, flights_list, config, pending_events = (
                self._generate_procedural(seed, task_id, difficulty=diff)
            )
        else:
            # Static data from files
            passengers_list, flights_list, config, pending_events = (
                self._load_static(task_id)
            )

        # Build dicts
        passengers = {p["passenger_id"]: p for p in passengers_list}
        flights = {fl["flight_id"]: fl for fl in flights_list}

        # Build groups
        groups: Dict[str, List[str]] = {}
        for pid, pax in passengers.items():
            gid = pax.get("group_id")
            if gid:
                groups.setdefault(gid, []).append(pid)

        # Deep copy availability so bookings can decrement without mutating source
        flight_availability = {
            fid: copy.deepcopy(fl["cabin_availability"])
            for fid, fl in flights.items()
        }

        max_steps = config.get("max_steps", 60)
        compensation_budget = config.get("compensation_budget", 0.0)
        difficulty = config.get("difficulty", 0.5)

        self._episode = EpisodeState(
            passengers=passengers,
            flights=flights,
            groups=groups,
            config=config,
            bookings={},
            flight_availability=flight_availability,
            info_calls={},
            last_booking_step=0,
            passenger_details_fetched=set(),
            flights_listed=False,
            task_id=task_id,
            step_count=0,
            max_steps=max_steps,
            cumulative_reward=0.0,
            done=False,
            total_cost=0.0,
            compensation_budget=compensation_budget,
            pending_events=pending_events,
            fired_events_log=[],
            cancelled_flights=set(),
            unbook_count=0,
            events_this_step=[],
            difficulty=difficulty,
        )
        self._reward_computer = RewardComputer(
            total_passengers=len(passengers),
            max_steps=max_steps,
            difficulty=difficulty,
            compensation_budget=compensation_budget,
        )

        return self._build_observation(
            tool_result=None,
            reward=0.0,
            reward_reason="Episode started",
            done=False,
            reward_breakdown=None,
        )

    def step(
        self, action: FlightRebookingAction  # type: ignore[override]
    ) -> FlightRebookingObservation:
        """Execute one tool call and return the updated observation."""
        if self._episode is None:
            raise RuntimeError("step() called without reset()")
        if self._episode.done:
            raise RuntimeError("step() called on a terminated episode")

        ep = self._episode
        rc = self._reward_computer
        ep.step_count += 1
        self._state.step_count += 1

        # --- Fire any pending events for this step ---
        ep.events_this_step = []
        self._fire_events(ep)
        had_event = len(ep.events_this_step) > 0

        tool_name = action.tool_name
        args = action.args

        reward_breakdown = None

        # --- Route tool call ---
        try:
            if tool_name == "list_passengers":
                tool_result = tool_list_passengers(ep)
                reward, reason = rc.reward_for_info_call("list_passengers", ep)

            elif tool_name == "get_passenger_details":
                pid = args.get("passenger_id", "")
                already_booked = pid in ep.bookings
                tool_result = tool_get_passenger_details(ep, pid)
                if tool_result["status"] == "error":
                    reward, reason = rc.reward_for_failed_action(tool_result)
                elif already_booked:
                    reward, reason = rc.reward_for_info_call(
                        "get_passenger_details_booked", ep
                    )
                else:
                    reward, reason = rc.reward_for_info_call(
                        "get_passenger_details", ep
                    )

            elif tool_name == "list_alternative_flights":
                tool_result = tool_list_alternative_flights(ep)
                ep.flights_listed = True
                reward, reason = rc.reward_for_info_call(
                    "list_alternative_flights", ep
                )

            elif tool_name == "get_flight_details":
                fid = args.get("flight_id", "")
                tool_result = tool_get_flight_details(ep, fid)
                if tool_result["status"] == "error":
                    reward, reason = rc.reward_for_failed_action(tool_result)
                else:
                    reward, reason = rc.reward_for_info_call(
                        "get_flight_details", ep
                    )

            elif tool_name == "book_passenger":
                pid = args.get("passenger_id", "")
                fid = args.get("flight_id", "")
                cabin = args.get("cabin", "")
                tool_result = tool_book_passenger(ep, pid, fid, cabin)
                if tool_result["status"] == "success":
                    pax = ep.passengers.get(pid, {})
                    reward, reason = rc.reward_for_booking(
                        tool_result, pax, ep
                    )
                    # Decomposed breakdown
                    reward_breakdown = rc.compute_step_breakdown(
                        tool_result, pax, ep
                    )
                    # Opportunity cost
                    opp_cost, opp_explanation = rc.compute_opportunity_cost(
                        pax, fid, cabin, ep
                    )
                    if opp_cost != 0.0:
                        reward += opp_cost
                        reason += f" | Opportunity cost: {opp_explanation}"
                        reward_breakdown["opportunity_cost"] = opp_cost
                else:
                    reward, reason = rc.reward_for_failed_action(tool_result)

            elif tool_name == "book_group":
                gid = args.get("group_id", "")
                fid = args.get("flight_id", "")
                cabin_assignments = args.get("cabin_assignments", {})
                tool_result = tool_book_group(ep, gid, fid, cabin_assignments)
                if tool_result["status"] == "success":
                    group_pax = [
                        ep.passengers[pid]
                        for pid in ep.groups.get(gid, [])
                    ]
                    reward, reason = rc.reward_for_group_booking(
                        tool_result, group_pax, ep
                    )
                else:
                    reward, reason = rc.reward_for_failed_action(tool_result)

            elif tool_name == "unbook_passenger":
                pid = args.get("passenger_id", "")
                tool_result = tool_unbook_passenger(ep, pid)
                reward, reason = rc.reward_for_unbook(tool_result, had_event)

            elif tool_name == "finalize_plan":
                tool_result = tool_finalize_plan(ep)
                reward, reason = REWARD_FINALIZE, "Plan finalized"

            else:
                tool_result = {
                    "status": "error",
                    "message": f"Unknown tool: {tool_name!r}",
                }
                reward, reason = rc.reward_for_invalid_tool()

        except Exception as exc:
            tool_result = {"status": "error", "message": f"Internal error: {exc}"}
            reward, reason = rc.reward_for_invalid_tool()

        # --- Check repeated tool calls (same tool + same args >2 times in a row) ---
        current_call = {"tool_name": tool_name, "args": args}
        ep.recent_tool_calls.append(current_call)
        # Keep only the last 3 entries to check for 3-in-a-row
        if len(ep.recent_tool_calls) > 3:
            ep.recent_tool_calls = ep.recent_tool_calls[-3:]
        # Penalize if the last 3 calls are identical (i.e., >2 in a row)
        if len(ep.recent_tool_calls) >= 3:
            last_three = ep.recent_tool_calls[-3:]
            if (last_three[0] == last_three[1] == last_three[2]):
                reward += REWARD_REPEATED_CALL
                reason += " | Penalty: same tool called >2 times in a row with identical arguments"

        # --- Check termination ---
        all_booked = len(ep.bookings) >= len(ep.passengers)
        step_limit_reached = ep.step_count >= ep.max_steps
        done = bool(ep.done or all_booked or step_limit_reached)

        # --- Terminal grading ---
        if done:
            breakdown = rc.terminal_breakdown(
                ep.bookings, ep.passengers, ep.flights, ep.groups,
                ep.total_cost, ep.compensation_budget,
            )
            grader = rc.grader_score(
                ep.bookings, ep.passengers, ep.flights, ep.groups,
                ep.total_cost, ep.compensation_budget,
            )

            if step_limit_reached and not all_booked and not ep.done:
                reason += " | Episode timed out — not all passengers booked"
            elif all_booked and not ep.done:
                reason += " | All passengers booked — auto-finalized"
            else:
                reason += " | Episode complete"

            tool_result = {
                **(tool_result or {}),
                "terminal_breakdown": breakdown,
                "grader_score": grader,
            }
            ep.done = True

        ep.cumulative_reward += reward

        return self._build_observation(
            tool_result=tool_result,
            reward=reward,
            reward_reason=reason,
            done=done,
            reward_breakdown=reward_breakdown,
        )

    @property
    def state(self) -> FlightRebookingState:  # type: ignore[override]
        if self._episode is None:
            return FlightRebookingState()

        ep = self._episode
        n_booked = len(ep.bookings)

        return FlightRebookingState(
            episode_id=self._state.episode_id,
            step_count=ep.step_count,
            total_passengers=len(ep.passengers),
            passengers_booked=n_booked,
            passengers_remaining=len(ep.passengers) - n_booked,
            cumulative_reward=ep.cumulative_reward,
            is_complete=ep.done,
            total_cost=ep.total_cost,
            compensation_budget_remaining=ep.compensation_budget - ep.total_cost,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_static(self, task_id: str):
        """Load data from static JSON files in data/{task_id}/."""
        task_dir = self._data_dir / task_id
        if not task_dir.is_dir():
            raise ValueError(
                f"Unknown task_id {task_id!r}: no data directory at {task_dir}"
            )

        with open(task_dir / "config.json") as f:
            config = json.load(f)
        with open(task_dir / "passengers.json") as f:
            passengers_list = json.load(f)["passengers"]
        with open(task_dir / "flights.json") as f:
            flights_list = json.load(f)["flights"]

        # Add default fields for backward compatibility with old data files
        for pax in passengers_list:
            pax.setdefault("loyalty_status", "none")
        config.setdefault("compensation_budget", 0.0)
        config.setdefault("difficulty", {"easy": 0.2, "medium": 0.5, "hard": 0.8}.get(task_id, 0.5))
        config.setdefault("events_enabled", False)

        return passengers_list, flights_list, config, []

    def _generate_procedural(self, seed: int, task_id: str, difficulty: float = None):
        """Generate episode data procedurally from a seed."""
        import sys
        import os
        # Add data directory to path for generator import
        data_dir = str(self._data_dir)
        if data_dir not in sys.path:
            sys.path.insert(0, str(self._data_dir.parent))

        from data.generate import generate_episode_data, generate_events

        # Parse difficulty from task_id if not provided
        if difficulty is None:
            # Extract from task_id like "seed_42" or use 0.5
            difficulty = 0.5

        passengers_doc, flights_doc, config = generate_episode_data(
            seed=seed, difficulty=difficulty
        )

        passengers_list = passengers_doc["passengers"]
        flights_list = flights_doc["flights"]

        # Generate mid-episode events if enabled
        pending_events = []
        if config.get("events_enabled", False):
            rng = random.Random(seed + 1000)  # separate seed for events
            pending_events = generate_events(
                rng, passengers_list, flights_list,
                config["max_steps"], difficulty,
            )

        return passengers_list, flights_list, config, pending_events

    # ------------------------------------------------------------------
    # Mid-episode events
    # ------------------------------------------------------------------

    def _fire_events(self, ep: EpisodeState) -> None:
        """Check and fire any pending events for the current step."""
        if not ep.pending_events:
            return

        to_fire = [e for e in ep.pending_events if e["step"] == ep.step_count]
        ep.pending_events = [e for e in ep.pending_events if e["step"] != ep.step_count]

        for event in to_fire:
            self._apply_event(ep, event)
            ep.events_this_step.append(event)
            ep.fired_events_log.append(event)

    def _apply_event(self, ep: EpisodeState, event: dict) -> None:
        """Apply a single mid-episode event to the episode state."""
        etype = event["type"]

        if etype == "capacity_change":
            fid = event["flight_id"]
            cabin = event["cabin"]
            delta = event["delta"]
            if fid in ep.flight_availability and fid not in ep.cancelled_flights:
                current = ep.flight_availability[fid].get(cabin, 0)
                ep.flight_availability[fid][cabin] = max(0, current + delta)

        elif etype == "new_passenger":
            new_pax = event["passenger"]
            pid = new_pax["passenger_id"]
            ep.passengers[pid] = new_pax
            # New passenger is unbooked; agent must handle them

        elif etype == "ssr_equipment_failure":
            fid = event.get("flight_id")
            lost_ssr = event.get("lost_ssr")
            if fid and lost_ssr and fid in ep.flights and fid not in ep.cancelled_flights:
                ssr_list = ep.flights[fid]["supports_ssr"]
                if lost_ssr in ssr_list:
                    ssr_list.remove(lost_ssr)
                # Passengers already booked on this flight with that SSR
                # are now in violation — agent must detect and fix this
                # (the grader will penalize at terminal scoring)

        elif etype == "deadline_shift":
            pid = event.get("passenger_id")
            new_dl = event.get("new_deadline")
            if pid and pid in ep.passengers and new_dl:
                ep.passengers[pid]["downstream_deadline"] = new_dl

        elif etype == "secondary_cancellation":
            fid = event.get("flight_id")
            if fid and fid in ep.flights:
                ep.cancelled_flights.add(fid)
                # Unbook all passengers on this flight
                to_unbook = [
                    pid for pid, b in ep.bookings.items()
                    if b["flight_id"] == fid
                ]
                for pid in to_unbook:
                    booking = ep.bookings[pid]
                    ep.total_cost -= booking.get("cost", 0.0)
                    del ep.bookings[pid]
                if to_unbook:
                    event["unbooked_passengers"] = to_unbook

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        tool_result: Optional[dict],
        reward: float,
        reward_reason: str,
        done: bool,
        reward_breakdown: Optional[dict] = None,
    ) -> FlightRebookingObservation:
        ep = self._episode
        n_booked = len(ep.bookings)

        booked_summary = [
            {
                "passenger_id": pid,
                "flight_id": booking["flight_id"],
                "cabin": booking["cabin"],
            }
            for pid, booking in ep.bookings.items()
        ]

        # Only include flights snapshot if agent has called list_alternative_flights
        flights_snapshot = None
        if ep.flights_listed:
            flights_snapshot = [
                {
                    "flight_id": fid,
                    "departure_time": fl["departure_time"],
                    "arrival_time": fl["arrival_time"],
                    "cabin_availability": dict(ep.flight_availability[fid]),
                    "supports_ssr": fl["supports_ssr"],
                }
                for fid, fl in ep.flights.items()
                if fid not in ep.cancelled_flights
            ]

        # Events that fired this step
        events = ep.events_this_step if ep.events_this_step else None

        return FlightRebookingObservation(
            passengers_total=len(ep.passengers),
            passengers_booked=n_booked,
            passengers_remaining=len(ep.passengers) - n_booked,
            tool_result=tool_result,
            reward_reason=reward_reason,
            step_count=ep.step_count,
            max_steps=ep.max_steps,
            cumulative_reward=ep.cumulative_reward,
            booked_summary=booked_summary,
            flights_snapshot=flights_snapshot,
            done=done,
            reward=reward,
            reward_breakdown=reward_breakdown,
            events=events,
            total_cost=ep.total_cost,
            compensation_budget=ep.compensation_budget,
        )
