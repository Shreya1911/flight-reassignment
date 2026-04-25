"""
Integration tests for FlightRebookingEnvironment.

Tests instantiate the environment directly — no server, no WebSocket.

Easy task:   8 passengers, no groups, no SSR, no deadlines
Medium task: 15 passengers, 2 groups (1 hard, 1 soft), 2 SSR, 2 deadlines
Hard task:   25 passengers, 4 groups (2 hard, 2 soft), 6 SSR, 5 deadlines

Run with: pytest tests/test_environment.py -v
"""

import pytest

from models import FlightRebookingAction, FlightRebookingObservation
from server.environment import FlightRebookingEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(task_id: str = "medium") -> FlightRebookingEnvironment:
    env = FlightRebookingEnvironment()
    env.reset(task_id=task_id)
    return env


def call_tool(env, tool_name, **args) -> FlightRebookingObservation:
    return env.step(FlightRebookingAction(tool_name=tool_name, args=args))


def book(env, passenger_id, flight_id, cabin) -> FlightRebookingObservation:
    return call_tool(env, "book_passenger",
                     passenger_id=passenger_id, flight_id=flight_id, cabin=cabin)


def unbook(env, passenger_id) -> FlightRebookingObservation:
    return call_tool(env, "unbook_passenger", passenger_id=passenger_id)


def book_group(env, group_id, flight_id, cabin_assignments) -> FlightRebookingObservation:
    return call_tool(env, "book_group",
                     group_id=group_id, flight_id=flight_id,
                     cabin_assignments=cabin_assignments)


# ---------------------------------------------------------------------------
# Known-optimal assignments for each tier (grader ~ 1.0)
# All same-cabin bookings -> cost = 0
# ---------------------------------------------------------------------------

OPTIMAL_EASY = {
    "PAX-E001": ("FL-201", "business"),
    "PAX-E002": ("FL-201", "business"),
    "PAX-E003": ("FL-202", "business"),
    "PAX-E004": ("FL-201", "economy"),
    "PAX-E005": ("FL-201", "economy"),
    "PAX-E006": ("FL-202", "economy"),
    "PAX-E007": ("FL-201", "economy"),
    "PAX-E008": ("FL-203", "economy"),
}

# Medium: individuals (non-group members)
OPTIMAL_MEDIUM_INDIVIDUAL = {
    "PAX-M001": ("FL-201", "business"),
    "PAX-M002": ("FL-201", "business"),
    "PAX-M006": ("FL-201", "premium_economy"),
    "PAX-M015": ("FL-201", "premium_economy"),
    "PAX-M007": ("FL-201", "economy"),
    "PAX-M009": ("FL-201", "economy"),
    "PAX-M010": ("FL-202", "business"),
    "PAX-M011": ("FL-202", "business"),
    "PAX-M012": ("FL-202", "premium_economy"),
    "PAX-M008": ("FL-202", "economy"),
    "PAX-M013": ("FL-202", "economy"),
    "PAX-M014": ("FL-203", "economy"),
}

OPTIMAL_MEDIUM_GROUPS = {
    "GRP-M01": ("FL-201", {"PAX-M003": "economy", "PAX-M004": "economy", "PAX-M005": "economy"}),
}

# Hard: individuals (non-group members)
OPTIMAL_HARD_INDIVIDUAL = {
    "PAX-H003": ("FL-201", "business"),
    "PAX-H004": ("FL-202", "business"),
    "PAX-H005": ("FL-203", "business"),
    "PAX-H006": ("FL-202", "business"),
    "PAX-H009": ("FL-202", "premium_economy"),
    "PAX-H010": ("FL-203", "premium_economy"),
    "PAX-H017": ("FL-202", "economy"),
    "PAX-H018": ("FL-203", "economy"),
    "PAX-H019": ("FL-203", "economy"),
    "PAX-H020": ("FL-201", "economy"),
    "PAX-H021": ("FL-203", "economy"),
    "PAX-H022": ("FL-204", "economy"),
    "PAX-H023": ("FL-203", "economy"),
    "PAX-H024": ("FL-201", "economy"),
    "PAX-H025": ("FL-204", "economy"),
}

OPTIMAL_HARD_GROUPS = {
    "GRP-H02": ("FL-201", {"PAX-H001": "business", "PAX-H002": "business"}),
    "GRP-H04": ("FL-201", {"PAX-H007": "premium_economy", "PAX-H008": "premium_economy"}),
    "GRP-H01": ("FL-202", {"PAX-H011": "economy", "PAX-H012": "economy", "PAX-H013": "economy"}),
    "GRP-H03": ("FL-201", {"PAX-H014": "economy", "PAX-H015": "economy", "PAX-H016": "economy"}),
}


def run_optimal_easy(env):
    """Book all easy passengers optimally, return final obs."""
    obs = None
    for pid, (fid, cabin) in OPTIMAL_EASY.items():
        obs = book(env, pid, fid, cabin)
    return obs


def run_optimal_medium(env):
    """Book all medium passengers optimally (groups + individuals), return final obs."""
    obs = None
    for gid, (fid, assignments) in OPTIMAL_MEDIUM_GROUPS.items():
        obs = book_group(env, gid, fid, assignments)
    for pid, (fid, cabin) in OPTIMAL_MEDIUM_INDIVIDUAL.items():
        obs = book(env, pid, fid, cabin)
    return obs


def run_optimal_hard(env):
    """Book all hard passengers optimally (groups + individuals), return final obs."""
    obs = None
    for gid, (fid, assignments) in OPTIMAL_HARD_GROUPS.items():
        obs = book_group(env, gid, fid, assignments)
    for pid, (fid, cabin) in OPTIMAL_HARD_INDIVIDUAL.items():
        obs = book(env, pid, fid, cabin)
    return obs


# ---------------------------------------------------------------------------
# 1. TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_basic_fields(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")

        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.passengers_remaining == 15
        assert obs.passengers_total == 15
        assert obs.tool_result is None
        assert obs.reward_reason == "Episode started"
        assert obs.step_count == 0

    def test_no_bookings_initially(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.booked_summary == []
        assert obs.passengers_booked == 0

    def test_max_steps_set(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.max_steps == 35

    def test_unknown_task_raises(self):
        env = FlightRebookingEnvironment()
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent")

    def test_flights_snapshot_initially_none(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.flights_snapshot is None

    def test_cost_fields_initialized(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.total_cost == 0.0
        assert obs.compensation_budget == 4000.0

    def test_events_initially_none(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.events is None

    def test_reward_breakdown_initially_none(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.reward_breakdown is None


# ---------------------------------------------------------------------------
# 2. TestListPassengers
# ---------------------------------------------------------------------------

class TestListPassengers:
    def test_returns_all_passengers(self):
        env = make_env("medium")
        obs = call_tool(env, "list_passengers")
        assert obs.tool_result["status"] == "success"
        assert len(obs.tool_result["passengers"]) == 15

    def test_returns_summary_not_full_details(self):
        env = make_env("medium")
        obs = call_tool(env, "list_passengers")
        entry = obs.tool_result["passengers"][0]
        assert "passenger_id" in entry
        assert "priority_tier" in entry
        assert "group_id" in entry
        assert "has_ssr" in entry
        assert "has_deadline" in entry
        # Should NOT have full details
        assert "original_cabin" not in entry
        assert "ssr_flags" not in entry

    def test_reward_is_small_positive_first_call(self):
        env = make_env("medium")
        obs = call_tool(env, "list_passengers")
        assert obs.reward > 0

    def test_includes_loyalty_status(self):
        env = make_env("medium")
        obs = call_tool(env, "list_passengers")
        entry = obs.tool_result["passengers"][0]
        assert "loyalty_status" in entry

    def test_includes_booked_flag(self):
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        obs = call_tool(env, "list_passengers")
        for pax in obs.tool_result["passengers"]:
            if pax["passenger_id"] == "PAX-E001":
                assert pax["booked"] is True
            else:
                assert pax["booked"] is False


# ---------------------------------------------------------------------------
# 3. TestGetPassengerDetails
# ---------------------------------------------------------------------------

class TestGetPassengerDetails:
    def test_returns_full_record(self):
        env = make_env("medium")
        obs = call_tool(env, "get_passenger_details", passenger_id="PAX-M001")
        r = obs.tool_result
        assert r["status"] == "success"
        assert r["passenger_id"] == "PAX-M001"
        assert r["original_cabin"] == "business"
        assert r["ssr_flags"] == ["UM"]
        assert r["priority_tier"] == 1
        assert "group_id" in r
        assert "downstream_deadline" in r

    def test_includes_loyalty_and_preferences(self):
        env = make_env("medium")
        obs = call_tool(env, "get_passenger_details", passenger_id="PAX-M001")
        r = obs.tool_result
        assert r["loyalty_status"] == "gold"
        assert "paid_window" in r
        assert "paid_legroom" in r

    def test_nonexistent_passenger_errors(self):
        env = make_env("medium")
        obs = call_tool(env, "get_passenger_details", passenger_id="PAX-FAKE")
        assert obs.tool_result["status"] == "error"

    def test_already_booked_passenger_penalty(self):
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        obs = call_tool(env, "get_passenger_details", passenger_id="PAX-E001")
        assert obs.tool_result["status"] == "success"
        assert obs.reward < 0  # penalty for querying already-booked

    def test_booked_passenger_shows_current_booking(self):
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        obs = call_tool(env, "get_passenger_details", passenger_id="PAX-E001")
        r = obs.tool_result
        assert "current_booking" in r
        assert r["current_booking"]["flight_id"] == "FL-201"
        assert r["current_booking"]["cabin"] == "business"
        assert "cost" in r["current_booking"]


# ---------------------------------------------------------------------------
# 4. TestListAlternativeFlights
# ---------------------------------------------------------------------------

class TestListAlternativeFlights:
    def test_returns_all_flights(self):
        env = make_env("medium")
        obs = call_tool(env, "list_alternative_flights")
        assert obs.tool_result["status"] == "success"
        assert len(obs.tool_result["flights"]) == 4

    def test_availability_decrements_after_booking(self):
        env = make_env("easy")
        obs1 = call_tool(env, "list_alternative_flights")
        biz_before = None
        for fl in obs1.tool_result["flights"]:
            if fl["flight_id"] == "FL-201":
                biz_before = fl["cabin_availability"]["business"]
                break

        book(env, "PAX-E001", "FL-201", "business")
        obs2 = call_tool(env, "list_alternative_flights")
        biz_after = None
        for fl in obs2.tool_result["flights"]:
            if fl["flight_id"] == "FL-201":
                biz_after = fl["cabin_availability"]["business"]
                break

        assert biz_after == biz_before - 1

    def test_includes_ssr_support(self):
        env = make_env("medium")
        obs = call_tool(env, "list_alternative_flights")
        for fl in obs.tool_result["flights"]:
            assert "supports_ssr" in fl

    def test_populates_flights_snapshot(self):
        env = make_env("easy")
        obs = call_tool(env, "list_alternative_flights")
        assert obs.flights_snapshot is not None
        assert len(obs.flights_snapshot) == 3


# ---------------------------------------------------------------------------
# 5. TestGetFlightDetails
# ---------------------------------------------------------------------------

class TestGetFlightDetails:
    def test_returns_full_flight(self):
        env = make_env("medium")
        obs = call_tool(env, "get_flight_details", flight_id="FL-201")
        r = obs.tool_result
        assert r["status"] == "success"
        assert r["flight_id"] == "FL-201"
        assert "cabin_availability" in r
        assert "departure_time" in r
        assert "arrival_time" in r
        assert "supports_ssr" in r

    def test_nonexistent_flight_errors(self):
        env = make_env("medium")
        obs = call_tool(env, "get_flight_details", flight_id="FL-999")
        assert obs.tool_result["status"] == "error"


# ---------------------------------------------------------------------------
# 6. TestBookPassenger
# ---------------------------------------------------------------------------

class TestBookPassenger:
    def test_successful_booking(self):
        env = make_env("easy")
        obs = book(env, "PAX-E001", "FL-201", "business")
        assert obs.tool_result["status"] == "success"
        assert obs.passengers_remaining == 7

    def test_cabin_availability_decremented(self):
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        obs = call_tool(env, "get_flight_details", flight_id="FL-201")
        assert obs.tool_result["cabin_availability"]["business"] == 3  # was 4

    def test_same_cabin_positive_reward(self):
        env = make_env("easy")
        obs = book(env, "PAX-E001", "FL-201", "business")  # business -> business
        assert obs.reward > 0

    def test_upgrade_positive_reward(self):
        env = make_env("easy")
        obs = book(env, "PAX-E004", "FL-201", "business")  # economy -> business
        assert obs.reward > 0

    def test_downgrade_negative_reward(self):
        """Downgrading a gold member from business to economy is penalized."""
        env = make_env("easy")
        obs = book(env, "PAX-E001", "FL-201", "economy")  # business -> economy (gold member)
        assert obs.reward < 0

    def test_double_booking_errors(self):
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        obs = book(env, "PAX-E001", "FL-202", "business")
        assert obs.tool_result["status"] == "error"
        assert "already booked" in obs.tool_result["message"].lower()

    def test_no_availability_errors(self):
        """Fill all business seats on FL-203 (only 2), then try a third."""
        env = make_env("easy")
        book(env, "PAX-E001", "FL-203", "business")
        book(env, "PAX-E002", "FL-203", "business")
        obs = book(env, "PAX-E003", "FL-203", "business")
        assert obs.tool_result["status"] == "error"
        assert "no" in obs.tool_result["message"].lower() and "available" in obs.tool_result["message"].lower()

    def test_ssr_mismatch_errors(self):
        """PAX-M001 has UM SSR. FL-202 does not support UM."""
        env = make_env("medium")
        obs = book(env, "PAX-M001", "FL-202", "business")
        assert obs.tool_result["status"] == "error"
        assert "ssr" in obs.tool_result["message"].lower()

    def test_deadline_violation_errors(self):
        """PAX-M006 has deadline 14:30. FL-204 arrives 18:15."""
        env = make_env("medium")
        obs = book(env, "PAX-M006", "FL-204", "premium_economy")
        assert obs.tool_result["status"] == "error"
        assert "deadline" in obs.tool_result["message"].lower()

    def test_nonexistent_passenger_errors(self):
        env = make_env("easy")
        obs = book(env, "PAX-FAKE", "FL-201", "economy")
        assert obs.tool_result["status"] == "error"

    def test_nonexistent_flight_errors(self):
        env = make_env("easy")
        obs = book(env, "PAX-E001", "FL-999", "economy")
        assert obs.tool_result["status"] == "error"

    def test_invalid_cabin_errors(self):
        env = make_env("easy")
        obs = book(env, "PAX-E001", "FL-201", "first_class")
        assert obs.tool_result["status"] == "error"

    def test_booked_summary_updated(self):
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        obs = book(env, "PAX-E002", "FL-201", "business")
        assert len(obs.booked_summary) == 2
        pids = {b["passenger_id"] for b in obs.booked_summary}
        assert pids == {"PAX-E001", "PAX-E002"}

    def test_booking_cost_in_result(self):
        """Successful bookings should include booking_cost."""
        env = make_env("easy")
        obs = book(env, "PAX-E001", "FL-201", "business")  # same cabin -> cost=0
        assert obs.tool_result["booking_cost"] == 0.0

    def test_upgrade_has_cost(self):
        """Economy -> business upgrade should cost money."""
        env = make_env("easy")
        obs = book(env, "PAX-E004", "FL-201", "business")  # economy -> business
        assert obs.tool_result["booking_cost"] > 0

    def test_downgrade_has_compensation(self):
        """Business -> economy downgrade should have compensation cost."""
        env = make_env("easy")
        obs = book(env, "PAX-E001", "FL-201", "economy")  # business -> economy
        assert obs.tool_result["booking_cost"] > 0

    def test_total_cost_tracks_bookings(self):
        """Total cost in observation should accumulate."""
        env = make_env("easy")
        obs1 = book(env, "PAX-E004", "FL-201", "business")  # economy -> business = upgrade cost
        cost1 = obs1.tool_result["booking_cost"]
        assert obs1.total_cost == cost1
        obs2 = book(env, "PAX-E005", "FL-201", "business")  # economy -> business
        cost2 = obs2.tool_result["booking_cost"]
        assert obs2.total_cost == pytest.approx(cost1 + cost2)

    def test_reward_breakdown_on_success(self):
        """Successful booking should populate reward_breakdown."""
        env = make_env("easy")
        obs = book(env, "PAX-E001", "FL-201", "business")
        assert obs.reward_breakdown is not None
        expected_keys = {
            "coverage_delta", "cabin_match_delta", "group_delta",
            "deadline_delta", "ssr_delta", "cost_delta",
            "loyalty_delta", "opportunity_cost",
        }
        assert set(obs.reward_breakdown.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 7. TestBookGroup
# ---------------------------------------------------------------------------

class TestBookGroup:
    def test_successful_group_booking(self):
        env = make_env("medium")
        obs = book_group(env, "GRP-M01", "FL-201",
                         {"PAX-M003": "economy", "PAX-M004": "economy", "PAX-M005": "economy"})
        assert obs.tool_result["status"] == "success"
        assert len(obs.tool_result["booked"]) == 3
        assert obs.passengers_remaining == 12  # 15 - 3

    def test_atomic_failure(self):
        """If capacity insufficient for all, none are booked."""
        env2 = make_env("medium")
        # FL-203 has 2 business seats. Fill them first.
        book(env2, "PAX-M001", "FL-203", "business")
        book(env2, "PAX-M002", "FL-203", "business")
        # Now FL-203 has 0 business seats. Try to book group
        obs = book_group(env2, "GRP-M02", "FL-203",
                         {"PAX-M010": "business", "PAX-M011": "business"})
        assert obs.tool_result["status"] == "error"
        # Verify neither member was booked
        assert obs.passengers_booked == 2  # only M001 and M002

    def test_hard_group_same_flight(self):
        """Hard group members end up on the same flight by design (single flight_id arg)."""
        env = make_env("medium")
        obs = book_group(env, "GRP-M01", "FL-201",
                         {"PAX-M003": "economy", "PAX-M004": "economy", "PAX-M005": "economy"})
        assert obs.tool_result["status"] == "success"
        # All on FL-201
        for entry in obs.tool_result["booked"]:
            assert entry["passenger_id"] in {"PAX-M003", "PAX-M004", "PAX-M005"}

    def test_split_cabin_allowed(self):
        """Different cabin assignments per member on same flight is valid."""
        env = make_env("medium")
        obs = book_group(env, "GRP-M02", "FL-201",
                         {"PAX-M010": "business", "PAX-M011": "premium_economy"})
        assert obs.tool_result["status"] == "success"

    def test_ssr_check_all_members(self):
        """GRP-H01 has PAX-H013 with WCHR. FL-203 doesn't support WCHR -> fail all."""
        env = make_env("hard")
        obs = book_group(env, "GRP-H01", "FL-203",
                         {"PAX-H011": "economy", "PAX-H012": "economy", "PAX-H013": "economy"})
        assert obs.tool_result["status"] == "error"
        assert "ssr" in obs.tool_result["message"].lower()
        assert obs.passengers_booked == 0

    def test_nonexistent_group_errors(self):
        env = make_env("medium")
        obs = book_group(env, "GRP-FAKE", "FL-201", {"PAX-M001": "business"})
        assert obs.tool_result["status"] == "error"

    def test_partially_booked_group_errors(self):
        """If one member is already booked, the whole group booking fails."""
        env = make_env("medium")
        book(env, "PAX-M003", "FL-201", "economy")  # book one member individually
        obs = book_group(env, "GRP-M01", "FL-201",
                         {"PAX-M003": "economy", "PAX-M004": "economy", "PAX-M005": "economy"})
        assert obs.tool_result["status"] == "error"
        assert "already booked" in obs.tool_result["message"].lower()

    def test_group_booking_cost_tracked(self):
        """Group booking should report total_group_cost."""
        env = make_env("medium")
        obs = book_group(env, "GRP-M01", "FL-201",
                         {"PAX-M003": "economy", "PAX-M004": "economy", "PAX-M005": "economy"})
        assert "total_group_cost" in obs.tool_result
        # Same cabin bookings -> 0 cost
        assert obs.tool_result["total_group_cost"] == 0.0


# ---------------------------------------------------------------------------
# 8. TestUnbookPassenger
# ---------------------------------------------------------------------------

class TestUnbookPassenger:
    def test_successful_unbook(self):
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        obs = unbook(env, "PAX-E001")
        assert obs.tool_result["status"] == "success"
        assert obs.tool_result["passenger_id"] == "PAX-E001"
        assert obs.tool_result["freed_flight"] == "FL-201"
        assert obs.tool_result["freed_cabin"] == "business"

    def test_unbook_frees_seat(self):
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        unbook(env, "PAX-E001")
        obs = call_tool(env, "get_flight_details", flight_id="FL-201")
        # Should be back to original 4 business seats
        assert obs.tool_result["cabin_availability"]["business"] == 4

    def test_unbook_updates_remaining(self):
        env = make_env("easy")
        obs1 = book(env, "PAX-E001", "FL-201", "business")
        assert obs1.passengers_remaining == 7
        obs2 = unbook(env, "PAX-E001")
        assert obs2.passengers_remaining == 8

    def test_unbook_reverses_cost(self):
        env = make_env("easy")
        obs1 = book(env, "PAX-E004", "FL-201", "business")  # economy -> business = upgrade cost
        cost_after_book = obs1.total_cost
        assert cost_after_book > 0
        obs2 = unbook(env, "PAX-E004")
        assert obs2.total_cost == pytest.approx(0.0)
        assert obs2.tool_result["cost_reversed"] == pytest.approx(cost_after_book)

    def test_unbook_not_booked_errors(self):
        env = make_env("easy")
        obs = unbook(env, "PAX-E001")
        assert obs.tool_result["status"] == "error"
        assert "not currently booked" in obs.tool_result["message"].lower()

    def test_unbook_nonexistent_errors(self):
        env = make_env("easy")
        obs = unbook(env, "PAX-FAKE")
        assert obs.tool_result["status"] == "error"

    def test_unbook_then_rebook(self):
        """After unbooking, the passenger can be rebooked."""
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        unbook(env, "PAX-E001")
        obs = book(env, "PAX-E001", "FL-202", "business")
        assert obs.tool_result["status"] == "success"
        assert obs.tool_result["flight_id"] == "FL-202"

    def test_unbook_reward_is_negative(self):
        """Unbooking should have a small negative reward (disruption cost)."""
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        obs = unbook(env, "PAX-E001")
        assert obs.reward < 0


# ---------------------------------------------------------------------------
# 9. TestFinalizePlan
# ---------------------------------------------------------------------------

class TestFinalizePlan:
    def test_triggers_done(self):
        env = make_env("easy")
        obs = call_tool(env, "finalize_plan")
        assert obs.done is True

    def test_grader_score_present(self):
        env = make_env("easy")
        obs = call_tool(env, "finalize_plan")
        assert "grader_score" in obs.tool_result

    def test_grader_score_in_range(self):
        env = make_env("easy")
        obs = call_tool(env, "finalize_plan")
        score = obs.tool_result["grader_score"]
        assert 0.0 < score < 1.0

    def test_step_after_done_raises(self):
        env = make_env("easy")
        call_tool(env, "finalize_plan")
        with pytest.raises(RuntimeError, match="terminated"):
            call_tool(env, "list_passengers")


# ---------------------------------------------------------------------------
# 10. TestEasyTask
# ---------------------------------------------------------------------------

class TestEasyTask:
    def test_reset_fields(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="easy")
        assert obs.passengers_total == 8
        assert obs.passengers_remaining == 8
        assert obs.max_steps == 20

    def test_full_optimal_booking(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        obs = run_optimal_easy(env)
        assert obs.done is True
        assert obs.passengers_remaining == 0
        score = obs.tool_result["grader_score"]
        assert score > 0.90

    def test_no_groups_no_ssr(self):
        """Easy data has no groups and no SSR."""
        env = make_env("easy")
        obs = call_tool(env, "list_passengers")
        for pax in obs.tool_result["passengers"]:
            assert pax["group_id"] is None
            assert pax["has_ssr"] is False
            assert pax["has_deadline"] is False

    def test_step_limit_terminates(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        obs = None
        for _ in range(20):
            obs = call_tool(env, "list_passengers")
        assert obs.done is True
        assert obs.step_count == 20

    def test_compensation_budget(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="easy")
        assert obs.compensation_budget == 5000.0


# ---------------------------------------------------------------------------
# 11. TestMediumTask
# ---------------------------------------------------------------------------

class TestMediumTask:
    def test_reset_fields(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.passengers_total == 15
        assert obs.passengers_remaining == 15
        assert obs.max_steps == 35

    def test_group_booking_works(self):
        env = make_env("medium")
        obs = book_group(env, "GRP-M01", "FL-201",
                         {"PAX-M003": "economy", "PAX-M004": "economy", "PAX-M005": "economy"})
        assert obs.tool_result["status"] == "success"

    def test_ssr_respected(self):
        """PAX-M001 has UM SSR. Can book on FL-201 (supports UM) but not FL-202."""
        env = make_env("medium")
        obs_bad = book(env, "PAX-M001", "FL-202", "business")
        assert obs_bad.tool_result["status"] == "error"

    def test_deadline_respected(self):
        """PAX-M006 has deadline 14:30. FL-201 arrives 12:15 -> OK. FL-204 arrives 18:15 -> fail."""
        env = make_env("medium")
        obs = book(env, "PAX-M006", "FL-201", "premium_economy")
        assert obs.tool_result["status"] == "success"

    def test_full_optimal_booking(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="medium")
        obs = run_optimal_medium(env)
        assert obs.done is True
        assert obs.passengers_remaining == 0
        score = obs.tool_result["grader_score"]
        assert score > 0.90

    def test_compensation_budget(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.compensation_budget == 4000.0


# ---------------------------------------------------------------------------
# 12. TestHardTask
# ---------------------------------------------------------------------------

class TestHardTask:
    def test_reset_fields(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="hard")
        assert obs.passengers_total == 25
        assert obs.passengers_remaining == 25
        assert obs.max_steps == 55

    def test_multiple_groups(self):
        """Hard data has 4 groups: 2 hard, 2 soft."""
        env = make_env("hard")
        obs = call_tool(env, "list_passengers")
        groups = set()
        for pax in obs.tool_result["passengers"]:
            if pax["group_id"]:
                groups.add(pax["group_id"])
        assert len(groups) == 4

    def test_ssr_scarcity(self):
        """Not all flights support all SSRs. PAX-H010 (pet_cabin) can only go on FL-201/FL-203."""
        env = make_env("hard")
        obs = book(env, "PAX-H010", "FL-202", "premium_economy")
        assert obs.tool_result["status"] == "error"
        assert "ssr" in obs.tool_result["message"].lower()

    def test_capacity_pressure(self):
        """FL-201 has only 3 business seats. Booking 4 should fail on the 4th."""
        env = make_env("hard")
        book(env, "PAX-H003", "FL-201", "business")
        book(env, "PAX-H004", "FL-201", "business")
        book(env, "PAX-H005", "FL-201", "business")
        obs = book(env, "PAX-H006", "FL-201", "business")
        assert obs.tool_result["status"] == "error"
        assert "available" in obs.tool_result["message"].lower()

    def test_full_optimal_booking(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="hard")
        obs = run_optimal_hard(env)
        assert obs.done is True
        assert obs.passengers_remaining == 0
        score = obs.tool_result["grader_score"]
        assert score > 0.90


# ---------------------------------------------------------------------------
# 13. TestInvalidTool
# ---------------------------------------------------------------------------

class TestInvalidTool:
    def test_unknown_tool_error(self):
        env = make_env("easy")
        obs = env.step(FlightRebookingAction(tool_name="teleport_passenger", args={}))
        assert obs.tool_result["status"] == "error"
        assert "teleport_passenger" in obs.tool_result["message"]

    def test_empty_args_handled(self):
        env = make_env("easy")
        obs = env.step(FlightRebookingAction(tool_name="book_passenger", args={}))
        assert obs.tool_result["status"] == "error"
        # Should error gracefully (empty passenger_id), not crash

    def test_unknown_tool_gives_negative_reward(self):
        env = make_env("easy")
        obs = env.step(FlightRebookingAction(tool_name="nonexistent", args={}))
        assert obs.reward < 0


# ---------------------------------------------------------------------------
# 14. TestEpisodeCompletion
# ---------------------------------------------------------------------------

class TestEpisodeCompletion:
    def test_auto_finalize_on_all_booked(self):
        """Episode ends automatically when all passengers are booked."""
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        obs = run_optimal_easy(env)
        assert obs.done is True
        assert "grader_score" in obs.tool_result

    def test_state_is_complete(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        run_optimal_easy(env)
        s = env.state
        assert s.is_complete is True
        assert s.passengers_booked == 8
        assert s.passengers_remaining == 0

    def test_state_tracks_cost(self):
        """State property should expose cost tracking."""
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        run_optimal_easy(env)
        s = env.state
        assert hasattr(s, "total_cost")
        assert hasattr(s, "compensation_budget_remaining")

    def test_timeout_terminates(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        obs = None
        for _ in range(20):
            obs = call_tool(env, "list_passengers")
        assert obs.done is True
        assert "timed out" in obs.reward_reason.lower()

    def test_grader_present_on_timeout(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        obs = None
        for _ in range(20):
            obs = call_tool(env, "list_passengers")
        assert "grader_score" in obs.tool_result

    def test_terminal_breakdown_keys(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        obs = run_optimal_easy(env)
        bd = obs.tool_result["terminal_breakdown"]
        assert set(bd.keys()) == {
            "coverage_score", "cabin_match_score", "group_integrity_score",
            "deadline_score", "ssr_integrity_score", "cost_efficiency_score",
            "loyalty_compliance_score", "hard_violations",
        }

    def test_cumulative_reward_accumulates(self):
        env = make_env("easy")
        obs1 = book(env, "PAX-E001", "FL-201", "business")
        obs2 = book(env, "PAX-E002", "FL-201", "business")
        assert obs2.cumulative_reward == pytest.approx(obs1.reward + obs2.reward)


# ---------------------------------------------------------------------------
# 15. TestCostTracking
# ---------------------------------------------------------------------------

class TestCostTracking:
    def test_same_cabin_zero_cost(self):
        """Same-cabin bookings should cost nothing."""
        env = make_env("easy")
        book(env, "PAX-E001", "FL-201", "business")
        obs = book(env, "PAX-E004", "FL-201", "economy")
        # E001 business->business = 0, E004 economy->economy = 0
        assert obs.total_cost == 0.0

    def test_upgrade_adds_cost(self):
        """Economy -> business upgrade should add to total cost."""
        env = make_env("easy")
        obs = book(env, "PAX-E004", "FL-201", "business")  # economy -> business
        assert obs.total_cost > 0

    def test_downgrade_adds_compensation(self):
        """Business -> economy downgrade has compensation cost."""
        env = make_env("easy")
        obs = book(env, "PAX-E001", "FL-201", "economy")  # business -> economy
        assert obs.total_cost > 0

    def test_gold_downgrade_extra_compensation(self):
        """Gold member downgrade incurs loyalty compensation on top of base."""
        env = make_env("easy")
        # PAX-E001 is gold, business -> economy
        obs_gold = book(env, "PAX-E001", "FL-201", "economy")
        gold_cost = obs_gold.tool_result["booking_cost"]

        env2 = make_env("easy")
        # PAX-E003 is none, business -> economy
        obs_none = book(env2, "PAX-E003", "FL-201", "economy")
        none_cost = obs_none.tool_result["booking_cost"]

        # Gold member should cost more due to loyalty entitlements
        assert gold_cost > none_cost
