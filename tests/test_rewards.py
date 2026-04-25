"""
Tests for server/rewards.py

All data is constructed inline — no file I/O.
Run with: pytest tests/test_rewards.py -v
"""

import pytest

from server.rewards import (
    RewardComputer,
    EPS,
    PRIORITY_WEIGHTS,
    priority_weight,
    REWARD_LIST_PASSENGERS_FIRST,
    REWARD_LIST_PASSENGERS_CHURN,
    REWARD_GET_DETAILS_UNBOOKED,
    REWARD_GET_DETAILS_BOOKED,
    REWARD_LIST_FLIGHTS,
    REWARD_GET_FLIGHT_DETAILS,
    REWARD_SAME_CABIN_GROUP,
    REWARD_UPGRADE,
    REWARD_DOWNGRADE,
    REWARD_DEADLINE_BONUS,
    REWARD_FAILED_BOOKING,
    REWARD_INVALID_TOOL,
    REWARD_UNBOOK_BASE,
    REWARD_UNBOOK_RECOVERY,
    GRADER_W_COVERAGE,
    GRADER_W_CABIN_MATCH,
    GRADER_W_GROUP_INTEGRITY,
    GRADER_W_DEADLINE,
    GRADER_W_SSR_INTEGRITY,
    GRADER_W_COST_EFFICIENCY,
    GRADER_W_LOYALTY_COMPLIANCE,
    GRADER_HARD_PENALTY,
    GROUP_SAME_FLIGHT_SAME_CABIN,
    GROUP_SAME_FLIGHT_DIFF_CABIN,
    GROUP_SPLIT_FLIGHTS_HARD,
    GROUP_SPLIT_FLIGHTS_SOFT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rc():
    """RewardComputer with 10 passengers, 60 max steps, 0.5 difficulty."""
    return RewardComputer(total_passengers=10, max_steps=60, difficulty=0.5, compensation_budget=5000)


@pytest.fixture
def rc_hard():
    """RewardComputer at high difficulty for progressive scaling tests."""
    return RewardComputer(total_passengers=10, max_steps=30, difficulty=0.9, compensation_budget=2000)


# Reusable test data
PASSENGERS = {
    "P1": {"passenger_id": "P1", "name": "A", "priority_tier": 1, "original_cabin": "business",
           "group_id": None, "group_integrity": None, "group_size": None,
           "ssr_flags": ["UM"], "downstream_deadline": "14:00", "loyalty_status": "gold",
           "paid_window": False, "paid_legroom": False},
    "P2": {"passenger_id": "P2", "name": "B", "priority_tier": 2, "original_cabin": "business",
           "group_id": "G1", "group_integrity": "hard", "group_size": 2,
           "ssr_flags": [], "downstream_deadline": None, "loyalty_status": "silver",
           "paid_window": False, "paid_legroom": False},
    "P3": {"passenger_id": "P3", "name": "C", "priority_tier": 3, "original_cabin": "economy",
           "group_id": "G1", "group_integrity": "hard", "group_size": 2,
           "ssr_flags": [], "downstream_deadline": None, "loyalty_status": "none",
           "paid_window": False, "paid_legroom": False},
    "P4": {"passenger_id": "P4", "name": "D", "priority_tier": 4, "original_cabin": "economy",
           "group_id": "G2", "group_integrity": "soft", "group_size": 2,
           "ssr_flags": [], "downstream_deadline": None, "loyalty_status": "none",
           "paid_window": False, "paid_legroom": False},
    "P5": {"passenger_id": "P5", "name": "E", "priority_tier": 5, "original_cabin": "economy",
           "group_id": "G2", "group_integrity": "soft", "group_size": 2,
           "ssr_flags": ["WCHR"], "downstream_deadline": "16:00", "loyalty_status": "none",
           "paid_window": False, "paid_legroom": False},
}

GROUPS = {
    "G1": ["P2", "P3"],
    "G2": ["P4", "P5"],
}

FLIGHTS = {
    "FL-A": {"flight_id": "FL-A", "departure_time": "09:00", "arrival_time": "12:00",
             "cabin_availability": {"economy": 5, "business": 3},
             "supports_ssr": ["UM", "WCHR"]},
    "FL-B": {"flight_id": "FL-B", "departure_time": "13:00", "arrival_time": "16:00",
             "cabin_availability": {"economy": 3, "business": 2},
             "supports_ssr": ["WCHR"]},
}


# ===========================================================================
# 1. TestInfoCallRewards
# ===========================================================================

class TestInfoCallRewards:
    def test_first_list_passengers_positive(self, rc):
        class FakeEp:
            info_calls = {"list_passengers": 1}
            last_booking_step = 0
            step_count = 1
        reward, reason = rc.reward_for_info_call("list_passengers", FakeEp())
        assert reward == REWARD_LIST_PASSENGERS_FIRST
        assert reward > 0

    def test_repeated_list_passengers_negative(self, rc):
        class FakeEp:
            info_calls = {"list_passengers": 5}
            last_booking_step = 0
            step_count = 10
        reward, reason = rc.reward_for_info_call("list_passengers", FakeEp())
        assert reward == REWARD_LIST_PASSENGERS_CHURN
        assert reward < 0

    def test_get_details_unbooked_positive(self, rc):
        class FakeEp:
            pass
        reward, _ = rc.reward_for_info_call("get_passenger_details", FakeEp())
        assert reward == REWARD_GET_DETAILS_UNBOOKED
        assert reward > 0

    def test_get_details_booked_negative(self, rc):
        class FakeEp:
            pass
        reward, _ = rc.reward_for_info_call("get_passenger_details_booked", FakeEp())
        assert reward == REWARD_GET_DETAILS_BOOKED
        assert reward < 0

    def test_list_flights_always_small_positive(self, rc):
        class FakeEp:
            info_calls = {"list_alternative_flights": 3}
        reward, _ = rc.reward_for_info_call("list_alternative_flights", FakeEp())
        assert reward == REWARD_LIST_FLIGHTS
        assert reward > 0


# ===========================================================================
# 2. TestBookingRewards
# ===========================================================================

class TestBookingRewards:
    def test_same_cabin_positive_reward(self, rc):
        pax = {"priority_tier": 1, "original_cabin": "business", "loyalty_status": "gold"}
        result = {"status": "success", "cabin": "business", "original_cabin": "business"}
        reward, _ = rc.reward_for_booking(result, pax, None)
        assert reward > 0

    def test_upgrade_positive_reward(self, rc):
        pax = {"priority_tier": 3, "original_cabin": "economy", "loyalty_status": "none"}
        result = {"status": "success", "cabin": "business", "original_cabin": "economy"}
        reward, _ = rc.reward_for_booking(result, pax, None)
        assert reward > 0

    def test_priority_weight_scaling(self, rc):
        result = {"status": "success", "cabin": "economy", "original_cabin": "economy"}
        pax1 = {"priority_tier": 1, "original_cabin": "economy", "loyalty_status": "none"}
        pax5 = {"priority_tier": 5, "original_cabin": "economy", "loyalty_status": "none"}
        r1, _ = rc.reward_for_booking(result, pax1, None)
        r5, _ = rc.reward_for_booking(result, pax5, None)
        assert r1 > r5

    def test_deadline_met_bonus(self, rc):
        pax = {"priority_tier": 2, "original_cabin": "economy", "loyalty_status": "none"}
        result_with = {"status": "success", "cabin": "economy", "original_cabin": "economy",
                       "deadline_met": True}
        result_without = {"status": "success", "cabin": "economy", "original_cabin": "economy"}
        reward_with, _ = rc.reward_for_booking(result_with, pax, None)
        reward_without, _ = rc.reward_for_booking(result_without, pax, None)
        assert reward_with > reward_without

    def test_failed_booking_penalty(self, rc):
        result = {"status": "error", "message": "No seats available"}
        pax = {"priority_tier": 3, "original_cabin": "economy", "loyalty_status": "none"}
        reward, _ = rc.reward_for_booking(result, pax, None)
        assert reward == REWARD_FAILED_BOOKING
        assert reward < 0

    def test_invalid_tool_penalty(self, rc):
        reward, _ = rc.reward_for_invalid_tool()
        assert reward == REWARD_INVALID_TOOL
        assert reward < 0


# ===========================================================================
# 3. TestUnbookRewards
# ===========================================================================

class TestUnbookRewards:
    def test_unbook_base_penalty(self, rc):
        result = {"status": "success", "passenger_id": "P1"}
        reward, _ = rc.reward_for_unbook(result, had_event_this_step=False)
        assert reward == REWARD_UNBOOK_BASE
        assert reward < 0

    def test_unbook_event_recovery_offset(self, rc):
        result = {"status": "success", "passenger_id": "P1"}
        reward, reason = rc.reward_for_unbook(result, had_event_this_step=True)
        assert reward == REWARD_UNBOOK_BASE + REWARD_UNBOOK_RECOVERY
        assert "event recovery" in reason

    def test_unbook_failed(self, rc):
        result = {"status": "error", "message": "Not booked"}
        reward, _ = rc.reward_for_unbook(result, had_event_this_step=False)
        assert reward == REWARD_FAILED_BOOKING


# ===========================================================================
# 4. TestProgressiveDifficulty
# ===========================================================================

class TestProgressiveDifficulty:
    def test_high_difficulty_scales_down_positive_rewards(self, rc, rc_hard):
        pax = {"priority_tier": 1, "original_cabin": "business", "loyalty_status": "none"}
        result = {"status": "success", "cabin": "business", "original_cabin": "business"}
        reward_normal, _ = rc.reward_for_booking(result, pax, None)
        reward_hard, _ = rc_hard.reward_for_booking(result, pax, None)
        assert reward_hard < reward_normal

    def test_difficulty_does_not_affect_penalties(self, rc, rc_hard):
        result = {"status": "error", "message": "No seats"}
        pax = {"priority_tier": 1, "original_cabin": "business", "loyalty_status": "none"}
        r1, _ = rc.reward_for_booking(result, pax, None)
        r2, _ = rc_hard.reward_for_booking(result, pax, None)
        assert r1 == r2 == REWARD_FAILED_BOOKING


# ===========================================================================
# 5. TestDecomposedBreakdown
# ===========================================================================

class TestDecomposedBreakdown:
    def test_breakdown_has_all_keys(self, rc):
        pax = PASSENGERS["P1"]
        result = {"status": "success", "cabin": "business", "cabin_match": True,
                  "booking_cost": 0.0}
        bd = rc.compute_step_breakdown(result, pax, None)
        expected_keys = {
            "coverage_delta", "cabin_match_delta", "group_delta",
            "deadline_delta", "ssr_delta", "cost_delta",
            "loyalty_delta", "opportunity_cost",
        }
        assert set(bd.keys()) == expected_keys

    def test_cabin_match_positive_delta(self, rc):
        pax = PASSENGERS["P1"]
        result = {"status": "success", "cabin": "business", "cabin_match": True,
                  "booking_cost": 0.0}
        bd = rc.compute_step_breakdown(result, pax, None)
        assert bd["cabin_match_delta"] > 0

    def test_downgrade_negative_cabin_delta(self, rc):
        pax = PASSENGERS["P1"]
        result = {"status": "success", "cabin": "economy", "cabin_match": False,
                  "booking_cost": 700.0}
        bd = rc.compute_step_breakdown(result, pax, None)
        assert bd["cabin_match_delta"] < 0

    def test_gold_downgrade_negative_loyalty(self, rc):
        pax = PASSENGERS["P1"]  # gold member
        result = {"status": "success", "cabin": "economy", "cabin_match": False,
                  "booking_cost": 700.0}
        bd = rc.compute_step_breakdown(result, pax, None)
        assert bd["loyalty_delta"] < 0

    def test_failed_booking_returns_zeros(self, rc):
        result = {"status": "error", "message": "fail"}
        bd = rc.compute_step_breakdown(result, PASSENGERS["P1"], None)
        assert all(v == 0.0 for v in bd.values())


# ===========================================================================
# 6. TestGraderScore
# ===========================================================================

class TestGraderScore:
    def test_zero_coverage(self, rc):
        bookings = {}
        score = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        assert EPS < score < 0.5

    def test_partial_coverage(self, rc):
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
            "P2": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
        }
        score = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        assert EPS < score < 1.0 - EPS

    def test_ssr_violation_penalizes(self, rc):
        bookings_bad = {
            "P1": {"flight_id": "FL-B", "cabin": "business", "cost": 0},
            "P2": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
            "P3": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
            "P4": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
            "P5": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
        }
        bookings_good = dict(bookings_bad)
        bookings_good["P1"] = {"flight_id": "FL-A", "cabin": "business", "cost": 0}
        score_bad = rc.grader_score(bookings_bad, PASSENGERS, FLIGHTS, GROUPS)
        score_good = rc.grader_score(bookings_good, PASSENGERS, FLIGHTS, GROUPS)
        assert score_bad < score_good

    def test_group_split_penalizes(self, rc):
        bookings_split = {
            "P1": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
            "P2": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
            "P3": {"flight_id": "FL-B", "cabin": "economy", "cost": 0},
            "P4": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
            "P5": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
        }
        bookings_together = {
            "P1": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
            "P2": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
            "P3": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
            "P4": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
            "P5": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
        }
        score_split = rc.grader_score(bookings_split, PASSENGERS, FLIGHTS, GROUPS)
        score_together = rc.grader_score(bookings_together, PASSENGERS, FLIGHTS, GROUPS)
        assert score_split < score_together

    def test_grader_is_deterministic(self, rc):
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
            "P2": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
            "P3": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
            "P4": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
            "P5": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
        }
        s1 = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        s2 = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        assert s1 == s2

    def test_grader_clamped_to_eps_range(self, rc):
        score_zero = rc.grader_score({}, PASSENGERS, FLIGHTS, GROUPS)
        assert score_zero >= EPS
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
            "P2": {"flight_id": "FL-A", "cabin": "business", "cost": 0},
            "P3": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
            "P4": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
            "P5": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
        }
        score_full = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        assert score_full <= 1.0 - EPS


# ===========================================================================
# 7. TestCostEfficiency
# ===========================================================================

class TestCostEfficiency:
    def test_zero_cost_high_score(self):
        score = RewardComputer._cost_efficiency_score(0.0, 10, 5000)
        assert score > 0.8

    def test_over_budget_low_score(self):
        score = RewardComputer._cost_efficiency_score(10000, 5, 5000)
        assert score < 0.5

    def test_no_bookings_zero(self):
        score = RewardComputer._cost_efficiency_score(0, 0, 5000)
        assert score == 0.0


# ===========================================================================
# 8. TestLoyaltyCompliance
# ===========================================================================

class TestLoyaltyCompliance:
    def test_gold_same_cabin_high_score(self):
        pax = {"P1": PASSENGERS["P1"]}
        bookings = {"P1": {"flight_id": "FL-A", "cabin": "business", "cost": 0}}
        score = RewardComputer._loyalty_compliance_score(bookings, pax)
        assert score == pytest.approx(1.0)

    def test_gold_downgraded_low_score(self):
        pax = {"P1": PASSENGERS["P1"]}
        bookings = {"P1": {"flight_id": "FL-A", "cabin": "economy", "cost": 700}}
        score = RewardComputer._loyalty_compliance_score(bookings, pax)
        assert score < 0.5

    def test_no_loyalty_passengers_perfect(self):
        pax = {"P3": PASSENGERS["P3"], "P4": PASSENGERS["P4"]}
        bookings = {
            "P3": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
            "P4": {"flight_id": "FL-A", "cabin": "economy", "cost": 0},
        }
        score = RewardComputer._loyalty_compliance_score(bookings, pax)
        assert score == pytest.approx(1.0)


# ===========================================================================
# 9. TestGraderWeightsSum
# ===========================================================================

def test_grader_weights_sum_to_one():
    total = (GRADER_W_COVERAGE + GRADER_W_CABIN_MATCH +
             GRADER_W_GROUP_INTEGRITY + GRADER_W_DEADLINE +
             GRADER_W_SSR_INTEGRITY + GRADER_W_COST_EFFICIENCY +
             GRADER_W_LOYALTY_COMPLIANCE)
    assert total == pytest.approx(1.0)


# ===========================================================================
# 10. TestPriorityWeights
# ===========================================================================

class TestPriorityWeights:
    def test_tier1_highest(self):
        assert priority_weight(1) == 1.5
        for tier in [2, 3, 4, 5]:
            assert priority_weight(1) > priority_weight(tier)

    def test_tier5_lowest(self):
        assert priority_weight(5) == 0.6

    def test_unknown_tier_defaults_to_1(self):
        assert priority_weight(99) == 1.0
