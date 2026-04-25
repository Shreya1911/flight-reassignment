"""
Reward computation for the Flight Rebooking Environment.

Three-layer reward system:
  1. Step-level rewards (shaping signal per action)
  2. Decomposed reward breakdown (per-component deltas per step)
  3. Terminal grader score (end-of-episode evaluation)

Includes:
  - Cost efficiency scoring (upgrade/downgrade economics)
  - Loyalty/compensation policy adherence
  - Opportunity cost computation
  - Progressive difficulty scaling
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-4

PRIORITY_WEIGHTS = {1: 1.5, 2: 1.3, 3: 1.0, 4: 0.8, 5: 0.6}

# Step rewards — info tools
REWARD_LIST_PASSENGERS_FIRST = 0.02
REWARD_LIST_PASSENGERS_CHURN = -0.01
REWARD_GET_DETAILS_UNBOOKED = 0.02
REWARD_GET_DETAILS_BOOKED = -0.01
REWARD_LIST_FLIGHTS = 0.01
REWARD_GET_FLIGHT_DETAILS = 0.01

# Step rewards — booking outcomes
REWARD_SAME_CABIN_GROUP = 0.3
REWARD_UPGRADE = 0.10
REWARD_SPLIT_CABIN_SAME_FLIGHT = -0.02
REWARD_DOWNGRADE = -0.02
REWARD_DEADLINE_BONUS = 0.05
REWARD_HARD_VIOLATION = -0.30
REWARD_FAILED_BOOKING = -0.50

# Step rewards — unbook
REWARD_UNBOOK_BASE = -0.05  # small penalty for disruption
REWARD_UNBOOK_RECOVERY = 0.03  # partial offset if recovering from an event

# Step rewards — other
REWARD_INVALID_TOOL = -0.20
REWARD_FINALIZE = 0.0
REWARD_REPEATED_CALL = -0.05  # penalty for calling same tool with same args >2 times in a row

# Grader component weights (sum to 1.0)
GRADER_W_COVERAGE = 0.25
GRADER_W_CABIN_MATCH = 0.15
GRADER_W_GROUP_INTEGRITY = 0.12
GRADER_W_DEADLINE = 0.13
GRADER_W_SSR_INTEGRITY = 0.15
GRADER_W_COST_EFFICIENCY = 0.10
GRADER_W_LOYALTY_COMPLIANCE = 0.10

# Hard-constraint penalty (subtracted from final grader score per violation)
GRADER_HARD_PENALTY = 0.15

# Group integrity scores
GROUP_SAME_FLIGHT_SAME_CABIN = 0.7
GROUP_SAME_FLIGHT_DIFF_CABIN = 0.5
GROUP_SPLIT_FLIGHTS_HARD = 0.0
GROUP_SPLIT_FLIGHTS_SOFT = 0.04

# Cost efficiency thresholds
COST_EFFICIENCY_IDEAL_PER_PAX = 100.0  # ideal average cost per passenger


def priority_weight(tier: int) -> float:
    """Return the priority multiplier for a given tier."""
    return PRIORITY_WEIGHTS.get(tier, 1.0)


# Cabin ordering for upgrade/downgrade detection
_CABIN_RANK = {"economy": 0, "premium_economy": 1, "business": 2}


def _cabin_rank(cabin: str) -> int:
    return _CABIN_RANK.get(cabin, 0)


# ---------------------------------------------------------------------------
# RewardComputer
# ---------------------------------------------------------------------------

class RewardComputer:
    """
    Reward computer with decomposed rewards, opportunity cost, and
    progressive difficulty scaling.
    """

    def __init__(
        self,
        total_passengers: int,
        max_steps: int,
        difficulty: float = 0.5,
        compensation_budget: float = 0.0,
    ):
        self.total_passengers = total_passengers
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.compensation_budget = compensation_budget

        # Progressive difficulty: scale rewards harsher at higher difficulty
        # At difficulty 0.0: multiplier = 1.0 (lenient)
        # At difficulty 1.0: multiplier = 0.6 (only good decisions rewarded well)
        self._reward_scale = max(0.6, 1.0 - 0.4 * difficulty)

    # ------------------------------------------------------------------
    # Step-level: info calls
    # ------------------------------------------------------------------

    def reward_for_info_call(
        self, tool_name: str, ep_state
    ) -> Tuple[float, str]:
        """Compute reward for an information-gathering tool call."""

        if tool_name == "list_passengers":
            count = ep_state.info_calls.get("list_passengers", 0)
            if count >= 3 and ep_state.last_booking_step < ep_state.step_count - (count - 1):
                return (REWARD_LIST_PASSENGERS_CHURN,
                        "Redundant list_passengers call with no intervening bookings")
            if count == 1:
                return (REWARD_LIST_PASSENGERS_FIRST,
                        "First list_passengers call — good planning")
            return (REWARD_LIST_PASSENGERS_FIRST,
                    "list_passengers call")

        if tool_name == "get_passenger_details":
            return (REWARD_GET_DETAILS_UNBOOKED,
                    "Fetched passenger details")

        if tool_name == "get_passenger_details_booked":
            return (REWARD_GET_DETAILS_BOOKED,
                    "Fetched details for already-booked passenger")

        if tool_name == "list_alternative_flights":
            return (REWARD_LIST_FLIGHTS,
                    "Listed alternative flights")

        if tool_name == "get_flight_details":
            return (REWARD_GET_FLIGHT_DETAILS,
                    "Fetched flight details")

        return (0.0, f"Info call: {tool_name}")

    # ------------------------------------------------------------------
    # Step-level: booking
    # ------------------------------------------------------------------

    def reward_for_booking(
        self,
        tool_result: dict,
        passenger: dict,
        ep_state,
    ) -> Tuple[float, str]:
        """Compute reward for a book_passenger call that succeeded."""
        if tool_result["status"] != "success":
            return (REWARD_FAILED_BOOKING,
                    f"Booking failed: {tool_result.get('message', '')}")

        pw = priority_weight(passenger["priority_tier"])
        assigned_cabin = tool_result["cabin"]
        original_cabin = passenger["original_cabin"]

        assigned_rank = _cabin_rank(assigned_cabin)
        original_rank = _cabin_rank(original_cabin)

        if assigned_cabin == original_cabin:
            reward = REWARD_SAME_CABIN_GROUP * pw
            reason = "Booking: same cabin"
        elif assigned_rank > original_rank:
            reward = REWARD_UPGRADE * pw
            reason = "Booking: cabin upgrade"
        else:
            reward = REWARD_DOWNGRADE * pw
            reason = "Booking: cabin downgrade"

        # Deadline bonus
        if tool_result.get("deadline_met"):
            reward += REWARD_DEADLINE_BONUS * pw
            reason += " + deadline met"

        # Apply progressive difficulty scaling
        if reward > 0:
            reward *= self._reward_scale

        return (reward, reason)

    def reward_for_group_booking(
        self,
        tool_result: dict,
        group_passengers: List[dict],
        ep_state,
    ) -> Tuple[float, str]:
        """Compute reward for a book_group call that succeeded."""
        if tool_result["status"] != "success":
            return (REWARD_FAILED_BOOKING,
                    f"Group booking failed: {tool_result.get('message', '')}")

        total_reward = 0.0
        booked_list = tool_result["booked"]
        cabin_set = set()

        for entry in booked_list:
            pid = entry["passenger_id"]
            pax = next(p for p in group_passengers if p["passenger_id"] == pid)
            pw = priority_weight(pax["priority_tier"])

            assigned_cabin = entry["cabin"]
            original_cabin = pax["original_cabin"]
            cabin_set.add(assigned_cabin)

            assigned_rank = _cabin_rank(assigned_cabin)
            original_rank = _cabin_rank(original_cabin)

            if assigned_cabin == original_cabin:
                total_reward += REWARD_SAME_CABIN_GROUP * pw
            elif assigned_rank > original_rank:
                total_reward += REWARD_UPGRADE * pw
            else:
                total_reward += REWARD_DOWNGRADE * pw

            # Deadline bonus
            if pax.get("downstream_deadline"):
                total_reward += REWARD_DEADLINE_BONUS * pw

        # Apply progressive difficulty scaling
        if total_reward > 0:
            total_reward *= self._reward_scale

        if len(cabin_set) == 1:
            reason = "Group booking: all same cabin on same flight"
        else:
            reason = "Group booking: split cabin on same flight"

        return (total_reward, reason)

    def reward_for_failed_action(
        self, tool_result: dict
    ) -> Tuple[float, str]:
        """Reward for a booking action rejected by the environment."""
        return (REWARD_FAILED_BOOKING,
                f"Action failed: {tool_result.get('message', '')}")

    def reward_for_invalid_tool(self) -> Tuple[float, str]:
        """Reward when the agent submits an unrecognized tool name."""
        return (REWARD_INVALID_TOOL, "Unrecognized tool name")

    # ------------------------------------------------------------------
    # Step-level: unbook
    # ------------------------------------------------------------------

    def reward_for_unbook(
        self, tool_result: dict, had_event_this_step: bool
    ) -> Tuple[float, str]:
        """Reward for unbooking a passenger."""
        if tool_result["status"] != "success":
            return (REWARD_FAILED_BOOKING,
                    f"Unbook failed: {tool_result.get('message', '')}")

        reward = REWARD_UNBOOK_BASE
        reason = "Unbooked passenger (disruption cost)"

        # Partial offset if there was a mid-episode event justifying the unbook
        if had_event_this_step:
            reward += REWARD_UNBOOK_RECOVERY
            reason += " — event recovery"

        return (reward, reason)

    # ------------------------------------------------------------------
    # Step-level: decomposed breakdown
    # ------------------------------------------------------------------

    def compute_step_breakdown(
        self,
        booking_result: Optional[dict],
        passenger: Optional[dict],
        ep_state,
    ) -> Dict[str, float]:
        """
        Compute a per-component reward breakdown for a single booking step.

        Returns a dict with component deltas. Used by the observation to give
        the agent fine-grained feedback on what aspect of its decision was
        good or bad.
        """
        breakdown = {
            "coverage_delta": 0.0,
            "cabin_match_delta": 0.0,
            "group_delta": 0.0,
            "deadline_delta": 0.0,
            "ssr_delta": 0.0,
            "cost_delta": 0.0,
            "loyalty_delta": 0.0,
            "opportunity_cost": 0.0,
        }

        if booking_result is None or passenger is None:
            return breakdown

        if booking_result.get("status") != "success":
            return breakdown

        pw = priority_weight(passenger["priority_tier"])
        n = max(1, self.total_passengers)

        # Coverage: booking one passenger improves coverage by 1/N
        breakdown["coverage_delta"] = pw / n

        # Cabin match
        if booking_result.get("cabin_match"):
            breakdown["cabin_match_delta"] = pw / n
        elif _cabin_rank(booking_result.get("cabin", "")) > _cabin_rank(passenger.get("original_cabin", "")):
            breakdown["cabin_match_delta"] = 0.3 * pw / n  # upgrade, partial credit
        else:
            breakdown["cabin_match_delta"] = -0.5 * pw / n  # downgrade penalty

        # Deadline
        if booking_result.get("deadline_met"):
            breakdown["deadline_delta"] = pw / n

        # Cost
        cost = booking_result.get("booking_cost", 0.0)
        ideal = COST_EFFICIENCY_IDEAL_PER_PAX
        if cost <= ideal:
            breakdown["cost_delta"] = 0.02 * pw
        elif cost <= ideal * 3:
            breakdown["cost_delta"] = 0.0
        else:
            breakdown["cost_delta"] = -0.02 * pw

        # Loyalty compliance for downgrades
        loyalty = passenger.get("loyalty_status", "none")
        orig_rank = _cabin_rank(passenger.get("original_cabin", ""))
        new_rank = _cabin_rank(booking_result.get("cabin", ""))
        if new_rank < orig_rank and loyalty in ("gold", "silver"):
            # Downgrade of loyal passenger: negative signal
            breakdown["loyalty_delta"] = -0.03 * pw
        elif new_rank >= orig_rank and loyalty == "gold":
            # Gold member well-treated
            breakdown["loyalty_delta"] = 0.01 * pw

        return breakdown

    def compute_opportunity_cost(
        self,
        passenger: dict,
        booked_flight_id: str,
        booked_cabin: str,
        ep_state,
    ) -> Tuple[float, str]:
        """
        Compute what the agent gave up by making this booking.

        Checks if this booking consumed a scarce resource that another
        passenger needs more urgently.
        """
        cost = 0.0
        explanation_parts = []

        # Check: did we consume the last seat in a cabin on an SSR-compatible flight?
        remaining = ep_state.flight_availability.get(booked_flight_id, {}).get(booked_cabin, 0)
        if remaining == 0:
            # Find other unbooked passengers who need this cabin + flight's SSR support
            fl = ep_state.flights.get(booked_flight_id, {})
            flight_ssr = set(fl.get("supports_ssr", []))

            competing_pax = []
            for pid, pax in ep_state.passengers.items():
                if pid in ep_state.bookings:
                    continue
                if pid == passenger["passenger_id"]:
                    continue
                pax_ssr = set(pax.get("ssr_flags", []))
                if pax_ssr and pax_ssr.issubset(flight_ssr):
                    if pax["original_cabin"] == booked_cabin:
                        competing_pax.append(pax)

            if competing_pax:
                # Someone else needed this exact resource
                highest_tier = min(p["priority_tier"] for p in competing_pax)
                if highest_tier < passenger["priority_tier"]:
                    cost = -0.05
                    explanation_parts.append(
                        f"Last {booked_cabin} seat on SSR-compatible {booked_flight_id} "
                        f"consumed; {len(competing_pax)} unbooked passenger(s) with SSR "
                        f"need(s) this resource (highest tier: {highest_tier})"
                    )
                elif competing_pax:
                    cost = -0.02
                    explanation_parts.append(
                        f"Last {booked_cabin} seat on {booked_flight_id} consumed; "
                        f"{len(competing_pax)} SSR passenger(s) may be impacted"
                    )

        explanation = "; ".join(explanation_parts) if explanation_parts else "No significant opportunity cost"
        return (cost, explanation)

    # ------------------------------------------------------------------
    # Terminal: grader score
    # ------------------------------------------------------------------

    def grader_score(
        self,
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        flights: Dict[str, dict],
        groups: Dict[str, List[str]],
        total_cost: float = 0.0,
        compensation_budget: float = 0.0,
    ) -> float:
        """
        Compute the 0-1 grader score.

        Components (weights sum to 1.0):
          coverage_score        (0.25)
          cabin_match_score     (0.15)
          group_integrity       (0.12)
          deadline_score        (0.13)
          ssr_integrity         (0.15)
          cost_efficiency       (0.10)
          loyalty_compliance    (0.10)
        """
        breakdown = self.terminal_breakdown(
            bookings, passengers, flights, groups, total_cost, compensation_budget
        )

        score = (
            GRADER_W_COVERAGE * breakdown["coverage_score"]
            + GRADER_W_CABIN_MATCH * breakdown["cabin_match_score"]
            + GRADER_W_GROUP_INTEGRITY * breakdown["group_integrity_score"]
            + GRADER_W_DEADLINE * breakdown["deadline_score"]
            + GRADER_W_SSR_INTEGRITY * breakdown["ssr_integrity_score"]
            + GRADER_W_COST_EFFICIENCY * breakdown["cost_efficiency_score"]
            + GRADER_W_LOYALTY_COMPLIANCE * breakdown["loyalty_compliance_score"]
        )

        # Hard-constraint penalties
        score -= GRADER_HARD_PENALTY * breakdown["hard_violations"]

        return max(EPS, min(1.0 - EPS, score))

    def terminal_breakdown(
        self,
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        flights: Dict[str, dict],
        groups: Dict[str, List[str]],
        total_cost: float = 0.0,
        compensation_budget: float = 0.0,
    ) -> dict:
        """Compute all grader sub-scores and return them in a dict."""
        n_total = len(passengers)

        # --- Coverage ---
        n_booked = len(bookings)
        coverage_score = n_booked / n_total if n_total > 0 else 0.0

        # --- Cabin match (priority-weighted) ---
        cabin_match_score = self._cabin_match_score(bookings, passengers)

        # --- Group integrity ---
        group_integrity_score, hard_violations = self._group_integrity_score(
            bookings, passengers, groups
        )

        # --- Deadline score (priority-weighted) ---
        deadline_score = self._deadline_score(bookings, passengers, flights)

        # --- SSR integrity ---
        ssr_integrity_score, ssr_violations = self._ssr_integrity_score(
            bookings, passengers, flights
        )
        hard_violations += ssr_violations

        # --- Cost efficiency ---
        cost_efficiency_score = self._cost_efficiency_score(
            total_cost, n_booked, compensation_budget
        )

        # --- Loyalty compliance ---
        loyalty_compliance_score = self._loyalty_compliance_score(
            bookings, passengers
        )

        return {
            "coverage_score": coverage_score,
            "cabin_match_score": cabin_match_score,
            "group_integrity_score": group_integrity_score,
            "deadline_score": deadline_score,
            "ssr_integrity_score": ssr_integrity_score,
            "cost_efficiency_score": cost_efficiency_score,
            "loyalty_compliance_score": loyalty_compliance_score,
            "hard_violations": hard_violations,
        }

    # ------------------------------------------------------------------
    # Grader sub-score helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cabin_match_score(
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
    ) -> float:
        """Priority-weighted fraction of booked passengers in their original cabin."""
        total_weight = 0.0
        matched_weight = 0.0

        for pid, pax in passengers.items():
            pw = priority_weight(pax["priority_tier"])
            total_weight += pw
            if pid in bookings and bookings[pid]["cabin"] == pax["original_cabin"]:
                matched_weight += pw

        return matched_weight / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _group_integrity_score(
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        groups: Dict[str, List[str]],
    ) -> Tuple[float, int]:
        """
        Average per-group integrity score.
        Returns (avg_score, hard_violation_count).
        """
        if not groups:
            return 1.0, 0

        total_score = 0.0
        hard_violations = 0

        for gid, member_ids in groups.items():
            integrity = passengers[member_ids[0]]["group_integrity"]

            booked_flights = set()
            booked_cabins = set()
            all_booked = True
            for pid in member_ids:
                if pid in bookings:
                    booked_flights.add(bookings[pid]["flight_id"])
                    booked_cabins.add(bookings[pid]["cabin"])
                else:
                    all_booked = False

            if not booked_flights:
                total_score += 0.0
                continue

            if not all_booked:
                if integrity == "hard":
                    total_score += GROUP_SPLIT_FLIGHTS_HARD
                    hard_violations += 1
                else:
                    total_score += GROUP_SPLIT_FLIGHTS_SOFT
                continue

            if len(booked_flights) == 1 and len(booked_cabins) == 1:
                total_score += GROUP_SAME_FLIGHT_SAME_CABIN
            elif len(booked_flights) == 1:
                total_score += GROUP_SAME_FLIGHT_DIFF_CABIN
            else:
                if integrity == "hard":
                    total_score += GROUP_SPLIT_FLIGHTS_HARD
                    hard_violations += 1
                else:
                    total_score += GROUP_SPLIT_FLIGHTS_SOFT

        avg = total_score / len(groups) if groups else 1.0
        return avg, hard_violations

    @staticmethod
    def _deadline_score(
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        flights: Dict[str, dict],
    ) -> float:
        """Priority-weighted fraction of deadline-bearing passengers whose deadlines are met."""
        total_weight = 0.0
        met_weight = 0.0

        for pid, pax in passengers.items():
            if pax["downstream_deadline"] is None:
                continue
            pw = priority_weight(pax["priority_tier"])
            total_weight += pw

            if pid in bookings:
                fid = bookings[pid]["flight_id"]
                if fid in flights:
                    from server.tools import meets_deadline
                    if meets_deadline(flights[fid]["arrival_time"], pax["downstream_deadline"]):
                        met_weight += pw

        return met_weight / total_weight if total_weight > 0 else 1.0

    @staticmethod
    def _ssr_integrity_score(
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        flights: Dict[str, dict],
    ) -> Tuple[float, int]:
        """1.0 if no SSR violations; penalised per violation."""
        violations = 0

        for pid, pax in passengers.items():
            if not pax["ssr_flags"]:
                continue
            if pid not in bookings:
                continue

            fid = bookings[pid]["flight_id"]
            if fid not in flights:
                violations += 1
                continue

            fl = flights[fid]
            required = set(pax["ssr_flags"])
            supported = set(fl["supports_ssr"])
            if not required.issubset(supported):
                violations += 1

        score = max(0.0, 1.0 - 0.25 * violations)
        return score, violations

    @staticmethod
    def _cost_efficiency_score(
        total_cost: float,
        n_booked: int,
        compensation_budget: float,
    ) -> float:
        """
        Score based on how efficiently the agent spent money.

        - Under budget with good coverage = high score
        - Over budget = penalized
        - No bookings = 0
        """
        if n_booked == 0:
            return 0.0

        avg_cost = total_cost / max(1, n_booked)

        # Budget adherence (if budget is set)
        if compensation_budget > 0:
            budget_ratio = total_cost / compensation_budget
            if budget_ratio <= 0.8:
                budget_score = 1.0
            elif budget_ratio <= 1.0:
                budget_score = 1.0 - (budget_ratio - 0.8) * 2.5  # linear decline 1.0->0.5
            else:
                budget_score = max(0.0, 0.5 - (budget_ratio - 1.0) * 2.0)  # sharp penalty
        else:
            budget_score = 1.0

        # Per-passenger cost efficiency
        if avg_cost <= COST_EFFICIENCY_IDEAL_PER_PAX:
            cost_score = 1.0
        elif avg_cost <= COST_EFFICIENCY_IDEAL_PER_PAX * 3:
            cost_score = 1.0 - (avg_cost - COST_EFFICIENCY_IDEAL_PER_PAX) / (
                COST_EFFICIENCY_IDEAL_PER_PAX * 2
            )
        else:
            cost_score = 0.0

        return (budget_score + cost_score) / 2.0

    @staticmethod
    def _loyalty_compliance_score(
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
    ) -> float:
        """
        Score based on how well loyalty members were treated.

        Gold members should not be downgraded if possible.
        Silver members get mild penalty for downgrades.
        """
        total_weight = 0.0
        compliance_weight = 0.0

        for pid, pax in passengers.items():
            loyalty = pax.get("loyalty_status", "none")
            if loyalty == "none":
                continue

            lw = 2.0 if loyalty == "gold" else 1.0
            total_weight += lw

            if pid not in bookings:
                # Not booked at all — bad for loyalty
                continue

            original_rank = _cabin_rank(pax["original_cabin"])
            booked_rank = _cabin_rank(bookings[pid]["cabin"])

            if booked_rank >= original_rank:
                # Same or upgraded — compliant
                compliance_weight += lw
            elif booked_rank == original_rank - 1:
                # One-step downgrade — partial compliance
                compliance_weight += lw * 0.4
            # Two-step downgrade (business -> economy) = 0 compliance

        return compliance_weight / total_weight if total_weight > 0 else 1.0
