"""
Expert (scripted) policy for the Flight Rebooking Environment.

Implements a greedy-optimal solver that:
  1. Surveys passengers and flights
  2. Sorts passengers by constraint urgency (hard constraints first, then priority)
  3. Books each passenger onto the best valid flight/cabin
  4. Uses book_group for hard groups
  5. Handles booking failures by trying alternatives
  6. Finalizes when done

Each action includes chain-of-thought reasoning explaining WHY the action
was chosen — this reasoning becomes the training signal during SFT.

Used by collect_sft_data.py to generate expert trajectories for SFT training.
"""

from __future__ import annotations

import json
import random
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from server.environment import FlightRebookingEnvironment
from models import FlightRebookingAction, FlightRebookingObservation


# ---------------------------------------------------------------------------
# Cabin helpers
# ---------------------------------------------------------------------------

CABIN_RANK = {"economy": 0, "premium_economy": 1, "business": 2}


def _cabin_rank(cabin: str) -> int:
    return CABIN_RANK.get(cabin, 0)


def _parse_time(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


# ---------------------------------------------------------------------------
# Passenger sorting key
# ---------------------------------------------------------------------------

def _urgency_key(pax: dict) -> Tuple:
    """
    Sort key for booking order. Lower = book first.

    Priority order:
      1. Hard group members (must be booked together)
      2. Passengers with SSR flags (limited flight options)
      3. Passengers with deadlines (limited time options)
      4. Gold loyalty members (protection from downgrade)
      5. Higher priority tier (tier 1 first)
      6. Silver loyalty members
      7. Everyone else
    """
    has_hard_group = (pax.get("group_integrity") == "hard")
    has_ssr = len(pax.get("ssr_flags", [])) > 0
    has_deadline = pax.get("downstream_deadline") is not None
    is_gold = pax.get("loyalty_status") == "gold"
    is_silver = pax.get("loyalty_status") == "silver"
    tier = pax.get("priority_tier", 5)

    # Lower tuple = higher priority
    return (
        0 if has_hard_group else 1,
        0 if has_ssr else 1,
        0 if has_deadline else 1,
        0 if is_gold else 1,
        tier,
        0 if is_silver else 1,
    )


# ---------------------------------------------------------------------------
# Flight scoring for a given passenger
# ---------------------------------------------------------------------------

def _score_flight_cabin(
    pax: dict,
    flight: dict,
    cabin: str,
    availability: int,
) -> Optional[float]:
    """
    Score a (flight, cabin) option for a passenger. Returns None if invalid.
    Higher score = better assignment.
    """
    if availability <= 0:
        return None

    # SSR check
    if pax.get("ssr_flags"):
        supported = set(flight.get("supports_ssr", []))
        required = set(pax["ssr_flags"])
        if not required.issubset(supported):
            return None

    # Deadline check
    if pax.get("downstream_deadline"):
        arr = _parse_time(flight["arrival_time"])
        dl = _parse_time(pax["downstream_deadline"])
        if arr > dl:
            return None

    score = 0.0
    original = pax.get("original_cabin", "economy")
    orig_rank = _cabin_rank(original)
    new_rank = _cabin_rank(cabin)

    # Cabin match is highly desirable
    if cabin == original:
        score += 100.0
    elif new_rank > orig_rank:
        # Upgrade: acceptable but costs money
        score += 30.0
    else:
        # Downgrade: least desirable
        score += 10.0
        # Extra penalty for loyalty downgrades
        loyalty = pax.get("loyalty_status", "none")
        if loyalty == "gold":
            score -= 20.0
        elif loyalty == "silver":
            score -= 10.0

    # Prefer earlier flights (less disruption)
    dep_minutes = _parse_time(flight["departure_time"])
    score -= dep_minutes * 0.01

    # Prefer flights with more availability (preserve options for others)
    score += availability * 0.5

    return score


# ---------------------------------------------------------------------------
# Expert Policy
# ---------------------------------------------------------------------------

class ExpertPolicy:
    """
    Greedy-optimal expert solver for the flight rebooking environment.

    Produces a sequence of (tool_name, args) tuples that form an expert
    trajectory through the episode.
    """

    def __init__(self, env: FlightRebookingEnvironment):
        self.env = env

    def solve(self, seed: int, difficulty: float = 0.5) -> List[dict]:
        """
        Run a full episode and return the trajectory.

        Returns a list of turn dicts:
        [
            {
                "observation_text": str,    # formatted observation
                "action": {"tool_name": str, "args": dict},
                "reasoning": str,           # chain-of-thought explanation
                "reward": float,
                "reward_reason": str,
                "tool_result": dict or None,
            },
            ...
        ]
        """
        # RNG for reasoning variation (doesn't affect env)
        self._reasoning_rng = random.Random(seed + 999)

        # Map difficulty to a named task_id so reset() picks the right
        # difficulty level via the non-"seed_" code path.
        if difficulty <= 0.3:
            task_id = "easy"
        elif difficulty <= 0.6:
            task_id = "medium"
        else:
            task_id = "hard"
        obs = self.env.reset(seed=seed, task_id=task_id)
        turns: List[dict] = []

        # Step 1: list_passengers
        reasoning = self._reason_survey_start(obs)
        obs, turn = self._do_action(obs, "list_passengers", {}, reasoning)
        turns.append(turn)
        if obs.done:
            return turns

        passengers_summary = obs.tool_result.get("passengers", [])

        # Step 2: list_alternative_flights
        reasoning = self._reason_list_flights(passengers_summary)
        obs, turn = self._do_action(
            obs, "list_alternative_flights", {}, reasoning
        )
        turns.append(turn)
        if obs.done:
            return turns

        flights_info = obs.tool_result.get("flights", [])

        # Build working data structures
        passengers_by_id = {}
        for p in passengers_summary:
            passengers_by_id[p["passenger_id"]] = p

        flights_by_id = {}
        for f in flights_info:
            flights_by_id[f["flight_id"]] = f

        # Step 3: Get details for constrained passengers first
        constrained_pids = []
        other_pids = []
        for p in passengers_summary:
            pid = p["passenger_id"]
            if p.get("booked"):
                continue
            if p.get("has_ssr") or p.get("has_deadline"):
                constrained_pids.append(pid)
            else:
                other_pids.append(pid)

        remaining_steps = obs.max_steps - obs.step_count
        detail_budget = min(
            len(constrained_pids) + len(other_pids),
            int(remaining_steps * 0.4),
        )
        pids_to_fetch = constrained_pids + other_pids
        detailed_pax: Dict[str, dict] = {}

        for pid in pids_to_fetch[:detail_budget]:
            if obs.done:
                break
            p_summary = passengers_by_id.get(pid, {})
            reasoning = self._reason_get_details(pid, p_summary, constrained_pids)
            obs, turn = self._do_action(
                obs, "get_passenger_details", {"passenger_id": pid}, reasoning
            )
            turns.append(turn)
            if obs.tool_result and obs.tool_result.get("status") == "success":
                detailed_pax[pid] = obs.tool_result

        # Build full passenger info (merge summary + details)
        all_pax = self._build_passenger_list(passengers_summary, detailed_pax)

        # Sort by urgency
        all_pax.sort(key=_urgency_key)

        # Identify hard groups
        hard_groups: Dict[str, List[dict]] = {}
        for pax in all_pax:
            gid = pax.get("group_id")
            if gid and pax.get("group_integrity") == "hard":
                hard_groups.setdefault(gid, []).append(pax)

        # Track what's been booked
        booked_pids = set()
        booked_groups = set()

        # Step 4: Book hard groups first (atomically)
        for gid, members in hard_groups.items():
            if obs.done:
                break
            if gid in booked_groups:
                continue
            if any(m["passenger_id"] in booked_pids for m in members):
                continue

            obs, group_turns = self._book_hard_group(
                obs, gid, members, flights_by_id, booked_pids
            )
            turns.extend(group_turns)

            for m in members:
                if self._is_booked(obs, m["passenger_id"]):
                    booked_pids.add(m["passenger_id"])
            booked_groups.add(gid)

        # Step 5: Book remaining passengers individually
        for pax in all_pax:
            if obs.done:
                break
            pid = pax["passenger_id"]
            if pid in booked_pids:
                continue

            obs, booking_turns = self._book_individual(
                obs, pax, flights_by_id, booked_pids
            )
            turns.extend(booking_turns)

            if self._is_booked(obs, pid):
                booked_pids.add(pid)

            # Handle mid-episode events: refresh flights if events fired
            if obs.events and not obs.done:
                event_types = [e.get("type", "unknown") for e in obs.events]
                reasoning = (
                    f"Mid-episode events detected: {event_types}. "
                    f"I need to refresh flight availability to see what changed "
                    f"before continuing with bookings."
                )
                obs, turn = self._do_action(
                    obs, "list_alternative_flights", {}, reasoning
                )
                turns.append(turn)
                if obs.tool_result and obs.tool_result.get("status") == "success":
                    flights_info = obs.tool_result.get("flights", [])
                    flights_by_id = {f["flight_id"]: f for f in flights_info}

        # Step 6: Finalize
        if not obs.done:
            reasoning = self._reason_finalize(obs, booked_pids, all_pax)
            obs, turn = self._do_action(obs, "finalize_plan", {}, reasoning)
            turns.append(turn)

        return turns

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _do_action(
        self,
        obs: FlightRebookingObservation,
        tool_name: str,
        args: dict,
        reasoning: str = "",
    ) -> Tuple[FlightRebookingObservation, dict]:
        """Execute one action and return (new_obs, turn_dict)."""
        obs_text = self._format_observation(obs)

        action = FlightRebookingAction(tool_name=tool_name, args=args)
        new_obs = self.env.step(action)

        turn = {
            "observation_text": obs_text,
            "action": {"tool_name": tool_name, "args": args},
            "reasoning": reasoning,
            "reward": new_obs.reward,
            "reward_reason": new_obs.reward_reason,
            "tool_result": new_obs.tool_result,
        }

        return new_obs, turn

    # ------------------------------------------------------------------
    # Booking logic
    # ------------------------------------------------------------------

    def _book_hard_group(
        self,
        obs: FlightRebookingObservation,
        group_id: str,
        members: List[dict],
        flights_by_id: Dict[str, dict],
        booked_pids: set = None,
    ) -> Tuple[FlightRebookingObservation, List[dict]]:
        """Attempt to book a hard group atomically. Returns (obs, turns)."""
        turns = []
        if booked_pids is None:
            booked_pids = set()

        # Find best flight for the whole group
        best_flight = None
        best_score = -float("inf")
        best_assignments = {}
        rejected_flights: List[Tuple[str, str]] = []  # (fid, reason)

        for fid, fl in flights_by_id.items():
            # Check SSR compatibility for all members
            supported = set(fl.get("supports_ssr", []))
            all_ssr_ok = True
            for m in members:
                if m.get("ssr_flags"):
                    if not set(m["ssr_flags"]).issubset(supported):
                        all_ssr_ok = False
                        break
            if not all_ssr_ok:
                rejected_flights.append((fid, "SSR incompatible"))
                continue

            # Check deadline for all members
            all_deadline_ok = True
            for m in members:
                if m.get("downstream_deadline"):
                    arr = _parse_time(fl["arrival_time"])
                    dl = _parse_time(m["downstream_deadline"])
                    if arr > dl:
                        all_deadline_ok = False
                        break
            if not all_deadline_ok:
                rejected_flights.append((fid, "deadline violation"))
                continue

            # Try to assign cabins: prefer original cabin, fall back to others
            avail = dict(fl.get("cabin_availability", {}))
            assignments = {}
            flight_score = 0.0
            feasible = True

            for m in members:
                pid = m["passenger_id"]
                orig = m.get("original_cabin", "economy")

                cabin_order = self._cabin_preference(orig)
                assigned = False
                for c in cabin_order:
                    if avail.get(c, 0) > 0:
                        assignments[pid] = c
                        avail[c] -= 1
                        score = _score_flight_cabin(
                            m, fl, c, avail.get(c, 0) + 1
                        )
                        if score is not None:
                            flight_score += score
                        assigned = True
                        break

                if not assigned:
                    feasible = False
                    break

            if not feasible:
                rejected_flights.append((fid, "insufficient capacity"))
                continue

            if flight_score > best_score:
                best_score = flight_score
                best_flight = fid
                best_assignments = assignments

        if best_flight and best_assignments:
            reasoning = self._reason_book_group(
                group_id, members, best_flight, best_assignments,
                flights_by_id, rejected_flights,
            )
            obs, turn = self._do_action(obs, "book_group", {
                "group_id": group_id,
                "flight_id": best_flight,
                "cabin_assignments": best_assignments,
            }, reasoning)
            turns.append(turn)

            # If group booking failed, try individual bookings as fallback
            if obs.tool_result and obs.tool_result.get("status") == "error":
                for m in members:
                    if obs.done:
                        break
                    obs, ind_turns = self._book_individual(
                        obs, m, flights_by_id, booked_pids
                    )
                    turns.extend(ind_turns)
        else:
            # No valid flight for the whole group; book individually
            for m in members:
                if obs.done:
                    break
                obs, ind_turns = self._book_individual(
                    obs, m, flights_by_id, booked_pids
                )
                turns.extend(ind_turns)

        return obs, turns

    def _book_individual(
        self,
        obs: FlightRebookingObservation,
        pax: dict,
        flights_by_id: Dict[str, dict],
        booked_pids: set = None,
    ) -> Tuple[FlightRebookingObservation, List[dict]]:
        """Book one passenger onto the best available flight. Returns (obs, turns)."""
        turns = []
        if booked_pids is None:
            booked_pids = set()
        pid = pax["passenger_id"]

        # Already booked check
        if self._is_booked(obs, pid):
            return obs, turns

        # Score all (flight, cabin) options
        options: List[Tuple[float, str, str]] = []
        for fid, fl in flights_by_id.items():
            for cabin in ["economy", "premium_economy", "business"]:
                avail = fl.get("cabin_availability", {}).get(cabin, 0)
                score = _score_flight_cabin(pax, fl, cabin, avail)
                if score is not None:
                    options.append((score, fid, cabin))

        # Sort by score descending
        options.sort(key=lambda x: -x[0])

        # Try top options until one succeeds
        attempt = 0
        for score, fid, cabin in options[:3]:
            if obs.done:
                break
            attempt += 1
            reasoning = self._reason_book_individual(
                pax, fid, cabin, flights_by_id, obs, booked_pids, attempt,
            )
            obs, turn = self._do_action(obs, "book_passenger", {
                "passenger_id": pid,
                "flight_id": fid,
                "cabin": cabin,
            }, reasoning)
            turns.append(turn)

            if obs.tool_result and obs.tool_result.get("status") == "success":
                # Update local flight availability tracking
                if fid in flights_by_id:
                    fl_avail = flights_by_id[fid].get("cabin_availability", {})
                    fl_avail[cabin] = max(0, fl_avail.get(cabin, 0) - 1)
                break

        return obs, turns

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cabin_preference(original: str) -> List[str]:
        """Return cabin preference order: original first, then same-or-higher, then lower."""
        order = [original]
        rank = _cabin_rank(original)
        # Prefer upgrade over downgrade
        for c in ["premium_economy", "business", "economy"]:
            if c != original and c not in order:
                order.append(c)
        return order

    @staticmethod
    def _is_booked(obs: FlightRebookingObservation, pid: str) -> bool:
        """Check if a passenger appears in the booked summary."""
        for b in (obs.booked_summary or []):
            if b.get("passenger_id") == pid:
                return True
        return False

    @staticmethod
    def _build_passenger_list(
        summary: List[dict],
        details: Dict[str, dict],
    ) -> List[dict]:
        """
        Merge passenger summary with detailed records.
        For passengers with details, use the richer record.
        For others, synthesize a minimal record from the summary.
        """
        result = []
        for s in summary:
            pid = s["passenger_id"]
            if pid in details:
                d = details[pid]
                result.append({
                    "passenger_id": pid,
                    "name": d.get("name", ""),
                    "priority_tier": d.get("priority_tier", s.get("priority_tier", 5)),
                    "original_cabin": d.get("original_cabin", "economy"),
                    "group_id": d.get("group_id", s.get("group_id")),
                    "group_integrity": d.get("group_integrity"),
                    "group_size": d.get("group_size"),
                    "ssr_flags": d.get("ssr_flags", []),
                    "downstream_deadline": d.get("downstream_deadline"),
                    "loyalty_status": d.get("loyalty_status", "none"),
                    "paid_window": d.get("paid_window", False),
                    "paid_legroom": d.get("paid_legroom", False),
                })
            else:
                # Minimal record from summary
                result.append({
                    "passenger_id": pid,
                    "name": "",
                    "priority_tier": s.get("priority_tier", 5),
                    "original_cabin": "economy",  # unknown, default
                    "group_id": s.get("group_id"),
                    "group_integrity": None,
                    "group_size": None,
                    "ssr_flags": [],
                    "downstream_deadline": None if not s.get("has_deadline") else "23:59",
                    "loyalty_status": s.get("loyalty_status", "none"),
                    "paid_window": False,
                    "paid_legroom": False,
                })
        return result

    # ------------------------------------------------------------------
    # Reasoning generation
    # ------------------------------------------------------------------

    def _reason_survey_start(self, obs: FlightRebookingObservation) -> str:
        """Reasoning for the initial list_passengers call."""
        templates = [
            "I need to start by surveying all passengers to understand the scope of this rebooking task. "
            f"There are {obs.passengers_total} passengers to rebook within {obs.max_steps} steps. "
            "Let me list all passengers first to identify who has constraints.",

            f"First step: get an overview of all {obs.passengers_total} passengers. "
            f"With a budget of ${obs.compensation_budget:.0f} and {obs.max_steps} steps, "
            "I need to plan efficiently. Listing passengers will reveal SSR flags, deadlines, "
            "groups, and priority tiers.",

            f"Episode starts with {obs.passengers_total} passengers needing rebooking. "
            f"Step budget is {obs.max_steps}, compensation budget is ${obs.compensation_budget:.0f}. "
            "I'll begin by listing all passengers to assess constraint density and plan my booking order.",
        ]
        return self._reasoning_rng.choice(templates)

    def _reason_list_flights(self, passengers_summary: List[dict]) -> str:
        """Reasoning for listing flights after seeing passengers."""
        n_ssr = sum(1 for p in passengers_summary if p.get("has_ssr"))
        n_deadline = sum(1 for p in passengers_summary if p.get("has_deadline"))
        n_grouped = sum(1 for p in passengers_summary if p.get("group_id"))
        n_gold = sum(1 for p in passengers_summary if p.get("loyalty_status") == "gold")

        parts = [f"Now I have the passenger overview: {len(passengers_summary)} total"]
        constraints = []
        if n_ssr > 0:
            constraints.append(f"{n_ssr} with SSR requirements")
        if n_deadline > 0:
            constraints.append(f"{n_deadline} with connection deadlines")
        if n_grouped > 0:
            constraints.append(f"{n_grouped} in groups")
        if n_gold > 0:
            constraints.append(f"{n_gold} gold loyalty members")

        if constraints:
            parts.append(f" ({', '.join(constraints)})")
        parts.append(
            ". I need to see what flights are available, their cabin capacity, "
            "and SSR support to plan assignments."
        )
        return "".join(parts)

    def _reason_get_details(
        self, pid: str, p_summary: dict, constrained_pids: List[str],
    ) -> str:
        """Reasoning for getting one passenger's details."""
        is_constrained = pid in constrained_pids
        has_ssr = p_summary.get("has_ssr", False)
        has_deadline = p_summary.get("has_deadline", False)
        tier = p_summary.get("priority_tier", 5)
        loyalty = p_summary.get("loyalty_status", "none")
        group_id = p_summary.get("group_id")

        reasons = []
        if is_constrained:
            if has_ssr and has_deadline:
                reasons.append(f"has both SSR flags and a connection deadline — highly constrained")
            elif has_ssr:
                reasons.append(f"has SSR requirements that limit which flights are valid")
            elif has_deadline:
                reasons.append(f"has a downstream connection deadline")
        if group_id:
            reasons.append(f"is part of group {group_id}")
        if loyalty == "gold":
            reasons.append(f"is a gold loyalty member (must avoid downgrade)")
        elif loyalty == "silver":
            reasons.append(f"is a silver loyalty member")
        if tier <= 2:
            reasons.append(f"is priority tier {tier} (high priority)")

        if reasons:
            reason_str = ", ".join(reasons)
            return (
                f"Fetching details for {pid} because this passenger {reason_str}. "
                f"I need to know their exact cabin class, SSR flags, and preferences "
                f"to make the right booking decision."
            )
        return (
            f"Getting details for {pid} (tier {tier}) to learn their original cabin "
            f"and preferences before booking."
        )

    def _reason_book_group(
        self,
        group_id: str,
        members: List[dict],
        best_flight: str,
        assignments: Dict[str, str],
        flights_by_id: Dict[str, dict],
        rejected_flights: List[Tuple[str, str]],
    ) -> str:
        """Reasoning for booking a hard group."""
        member_pids = [m["passenger_id"] for m in members]
        fl = flights_by_id.get(best_flight, {})

        parts = [
            f"Hard group {group_id} has {len(members)} members ({', '.join(member_pids)}) "
            f"that MUST fly together on the same flight."
        ]

        if rejected_flights:
            reject_summary = "; ".join(
                f"{fid}: {reason}" for fid, reason in rejected_flights[:3]
            )
            parts.append(f" Rejected flights: {reject_summary}.")

        cabin_set = set(assignments.values())
        parts.append(
            f" Best option is {best_flight} (departs {fl.get('departure_time', '?')}, "
            f"arrives {fl.get('arrival_time', '?')})"
        )
        if len(cabin_set) == 1:
            parts.append(f" with all members in {cabin_set.pop()}.")
        else:
            cabin_detail = ", ".join(
                f"{pid}: {c}" for pid, c in assignments.items()
            )
            parts.append(f" with cabin assignments: {cabin_detail}.")

        return "".join(parts)

    def _reason_book_individual(
        self,
        pax: dict,
        flight_id: str,
        cabin: str,
        flights_by_id: Dict[str, dict],
        obs: FlightRebookingObservation,
        booked_pids: set,
        attempt: int,
    ) -> str:
        """Reasoning for booking an individual passenger."""
        pid = pax["passenger_id"]
        orig_cabin = pax.get("original_cabin", "economy")
        tier = pax.get("priority_tier", 5)
        loyalty = pax.get("loyalty_status", "none")
        ssr = pax.get("ssr_flags", [])
        deadline = pax.get("downstream_deadline")
        fl = flights_by_id.get(flight_id, {})
        remaining = obs.passengers_remaining

        parts = []

        if attempt > 1:
            parts.append(
                f"Previous booking attempt for {pid} failed. Trying alternative: "
            )

        # Why this passenger now
        priority_reasons = []
        if ssr:
            priority_reasons.append(f"SSR {ssr}")
        if deadline:
            priority_reasons.append(f"deadline {deadline}")
        if loyalty == "gold":
            priority_reasons.append("gold loyalty")
        if tier <= 2:
            priority_reasons.append(f"high priority (tier {tier})")

        if priority_reasons:
            parts.append(
                f"Booking {pid} ({', '.join(priority_reasons)})"
            )
        else:
            parts.append(f"Booking {pid} (tier {tier}, {orig_cabin})")

        # Why this flight+cabin
        parts.append(f" onto {flight_id}")
        if cabin == orig_cabin:
            parts.append(f" in {cabin} (matches original cabin — no cost)")
        elif _cabin_rank(cabin) > _cabin_rank(orig_cabin):
            parts.append(f" in {cabin} (upgrade from {orig_cabin} — costs extra but necessary)")
        else:
            parts.append(f" in {cabin} (downgrade from {orig_cabin})")
            if loyalty == "gold":
                parts.append(" — gold member downgrade incurs compensation")

        if ssr:
            parts.append(f". Flight supports required SSR {fl.get('supports_ssr', [])}")
        if deadline:
            parts.append(
                f". Flight arrives at {fl.get('arrival_time', '?')} "
                f"which is before deadline {deadline}"
            )

        parts.append(f". {remaining} passengers still remaining.")
        return "".join(parts)

    def _reason_finalize(
        self,
        obs: FlightRebookingObservation,
        booked_pids: set,
        all_pax: List[dict],
    ) -> str:
        """Reasoning for finalizing the plan."""
        n_booked = obs.passengers_booked
        n_total = obs.passengers_total
        cost = obs.total_cost
        budget = obs.compensation_budget

        if n_booked == n_total:
            return (
                f"All {n_total} passengers have been rebooked successfully. "
                f"Total cost: ${cost:.0f} out of ${budget:.0f} budget. "
                f"Finalizing the plan."
            )
        unbooked = n_total - n_booked
        return (
            f"{n_booked}/{n_total} passengers booked, {unbooked} could not be placed "
            f"(likely due to capacity or constraint conflicts). "
            f"Total cost: ${cost:.0f}/${budget:.0f}. "
            f"Step budget is exhausted or no valid options remain. Finalizing."
        )

    @staticmethod
    def _format_observation(obs: FlightRebookingObservation) -> str:
        """Format observation as the user message the LLM would see."""
        from inference import format_state, format_main_task, format_result

        parts = []

        # On step 0 (initial obs), include task description
        if obs.step_count == 0:
            parts.append(format_main_task("procedural"))
            parts.append(format_state(obs))
            parts.append("Choose your next tool call. Respond with ONLY a JSON object.")
        else:
            # Format previous tool result
            result_parts = []
            if obs.tool_result is not None:
                result_parts.append(
                    f"Last tool result: {json.dumps(obs.tool_result, indent=2)}"
                )
            if obs.reward is not None:
                result_parts.append(
                    f"Reward: {obs.reward:.2f} ({obs.reward_reason})"
                )
            if result_parts:
                parts.append("\n".join(result_parts))

            parts.append(format_state(obs))
            parts.append("Choose your next tool call. Respond with ONLY a JSON object.")

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_expert_episode(
    seed: int,
    difficulty: float = 0.5,
) -> Tuple[List[dict], float, float]:
    """
    Run one expert episode and return (turns, cumulative_reward, grader_score).
    """
    env = FlightRebookingEnvironment()
    policy = ExpertPolicy(env)
    turns = policy.solve(seed=seed, difficulty=difficulty)

    # Extract score from the last turn's tool_result
    grader_score = 0.0
    cumulative_reward = 0.0
    if turns:
        last_result = turns[-1].get("tool_result", {})
        if last_result and "grader_score" in last_result:
            grader_score = last_result["grader_score"]
        cumulative_reward = sum(t.get("reward", 0.0) for t in turns)

    return turns, cumulative_reward, grader_score


if __name__ == "__main__":
    """Quick test: run one episode at each difficulty."""
    for diff in [0.2, 0.5, 0.8]:
        turns, reward, score = run_expert_episode(seed=42, difficulty=diff)
        n_bookings = sum(
            1 for t in turns
            if t["action"]["tool_name"] in ("book_passenger", "book_group")
            and t.get("tool_result", {}).get("status") == "success"
        )
        print(
            f"difficulty={diff:.1f} | steps={len(turns)} | "
            f"bookings={n_bookings} | reward={reward:.3f} | score={score:.4f}"
        )
