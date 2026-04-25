"""
Procedural data generator for the Flight Rebooking Environment.

Produces passengers.json, flights.json, and config.json for a given seed.
Supports adversarial scenario generation (greedy traps, distractor flights,
false scarcity, priority inversion) and Pareto-conflict constraint sets.

Usage:
    from data.generate import generate_episode_data
    passengers, flights, config = generate_episode_data(seed=42, difficulty=0.6)

    # difficulty is a float 0.0-1.0 controlling scarcity, constraint density, etc.
    # seed ensures reproducibility.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Aarav", "Priya", "Vikram", "Neha", "Rohan", "Ananya", "Arjun", "Diya",
    "Kabir", "Zara", "Aditya", "Ishaan", "Meera", "Ravi", "Sanya", "Vivaan",
    "Kavya", "Arnav", "Nisha", "Tanvi", "Om", "Pooja", "Yash", "Rajesh",
    "Sunita", "Amit", "Deepa", "Suresh", "Lakshmi", "Farhan", "Sana", "Nikhil",
    "Rekha", "Manish", "Sneha", "Ritu", "Gaurav", "Bhavna", "Chirag", "Jaya",
    "Pankaj", "Divya", "Rahul", "Anjali", "Kiran", "Mohan", "Swati", "Tarun",
    "Harini", "Dev", "Shreya", "Kunal", "Tara", "Varun", "Mira", "Siddharth",
]

LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Gupta", "Mehta", "Reddy", "Kumar", "Joshi",
    "Malhotra", "Khan", "Verma", "Nair", "Bhatia", "Shah", "Iyer", "Chopra",
    "Desai", "Saxena", "Rao", "Tiwari", "Kapoor", "Banerjee", "Menon", "Pillai",
    "Rajan", "Ahmed", "Das", "Mishra", "Yadav", "Pandey", "Krishnan", "Dubey",
    "Shetty", "Chauhan", "Thakur", "Nayak", "Agarwal", "Bhatt", "Goel",
]

CABINS = ["economy", "premium_economy", "business"]
CABIN_RANK = {"economy": 0, "premium_economy": 1, "business": 2}
SSR_TYPES = ["UM", "WCHR", "pet_cabin", "pet_cargo"]
LOYALTY_LEVELS = ["none", "silver", "gold"]
DESTINATIONS = ["DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "PNQ"]

# Costs per cabin upgrade/downgrade
UPGRADE_COST = {
    ("economy", "premium_economy"): 200,
    ("economy", "business"): 800,
    ("premium_economy", "business"): 500,
}
DOWNGRADE_COMPENSATION = {
    ("business", "premium_economy"): 400,
    ("business", "economy"): 700,
    ("premium_economy", "economy"): 200,
}


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_episode_data(
    seed: int,
    difficulty: float = 0.5,
    *,
    # Override ranges (None = use difficulty-derived defaults)
    n_passengers: Optional[int] = None,
    n_flights: Optional[int] = None,
    force_pareto_conflict: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Generate a complete episode dataset.

    Returns:
        (passengers_doc, flights_doc, config_doc)
        Each is a dict ready to be serialized to JSON.
    """
    rng = random.Random(seed)
    d = max(0.0, min(1.0, difficulty))

    # --- Derive counts from difficulty ---
    if n_passengers is None:
        n_passengers = _lerp_int(d, 8, 45, rng)
    if n_flights is None:
        n_flights = _lerp_int(d, 3, 8, rng)

    # Constraint densities
    ssr_density = _lerp(d, 0.0, 0.35) + rng.uniform(-0.05, 0.05)
    ssr_density = max(0.0, min(0.4, ssr_density))

    group_density = _lerp(d, 0.0, 0.35) + rng.uniform(-0.05, 0.05)
    group_density = max(0.0, min(0.4, group_density))

    deadline_density = _lerp(d, 0.0, 0.30) + rng.uniform(-0.05, 0.05)
    deadline_density = max(0.0, min(0.35, deadline_density))

    # Capacity scarcity: ratio of total seats to passengers
    # Low difficulty = 2.0x surplus, high difficulty = 0.85x (under-provisioned)
    surplus_ratio = _lerp(d, 2.0, 0.85)

    # Step budget: tighter at high difficulty
    steps_per_pax = _lerp(d, 3.5, 2.0)
    max_steps = max(15, int(n_passengers * steps_per_pax))

    # Compensation budget: tighter at high difficulty
    comp_budget_per_pax = _lerp(d, 500.0, 150.0)
    compensation_budget = int(n_passengers * comp_budget_per_pax)

    destination = rng.choice(DESTINATIONS)

    # --- Generate flights ---
    flights = _generate_flights(
        rng, n_flights, n_passengers, surplus_ratio, d
    )

    # --- Generate passengers ---
    passengers = _generate_passengers(
        rng, n_passengers, flights, d,
        ssr_density, group_density, deadline_density,
    )

    # --- Inject adversarial scenarios at higher difficulty ---
    if d >= 0.4:
        _inject_adversarial(rng, passengers, flights, d)

    # --- Force Pareto conflicts if requested or at high difficulty ---
    if force_pareto_conflict or d >= 0.6:
        _inject_pareto_conflicts(rng, passengers, flights, d)

    # --- Build output dicts ---
    passengers_doc = {"passengers": passengers}
    flights_doc = {"flights": flights}
    config_doc = {
        "task_id": f"seed_{seed}",
        "max_steps": max_steps,
        "cancelled_flight": "FL-100",
        "destination": destination,
        "compensation_budget": compensation_budget,
        "difficulty": round(d, 3),
        "events_enabled": d >= 0.3,
    }

    return passengers_doc, flights_doc, config_doc


# ---------------------------------------------------------------------------
# Flight generation
# ---------------------------------------------------------------------------

def _generate_flights(
    rng: random.Random,
    n_flights: int,
    n_passengers: int,
    surplus_ratio: float,
    difficulty: float,
) -> List[dict]:
    """Generate alternative flights with varied SSR support and capacity."""

    # Total seats needed across all cabins
    total_seats_target = int(n_passengers * surplus_ratio)

    # Cabin split: ~60% economy, ~25% premium, ~15% business
    economy_total = int(total_seats_target * 0.60)
    premium_total = int(total_seats_target * 0.25)
    business_total = total_seats_target - economy_total - premium_total

    flights = []
    base_hour = rng.randint(7, 10)
    flight_duration_minutes = rng.randint(150, 210)

    # Decide SSR coverage: at high difficulty, SSRs are scarce
    # Each flight supports a random subset of SSRs
    all_ssr = list(SSR_TYPES)

    for i in range(n_flights):
        fid = f"FL-{201 + i}"
        dep_hour = base_hour + i * rng.randint(1, 3)
        dep_min = rng.choice([0, 15, 30, 45])
        arr_minutes = dep_hour * 60 + dep_min + flight_duration_minutes + rng.randint(-15, 30)
        arr_hour = arr_minutes // 60
        arr_min = arr_minutes % 60

        dep_time = f"{dep_hour:02d}:{dep_min:02d}"
        arr_time = f"{arr_hour:02d}:{arr_min:02d}"

        # Distribute seats unevenly across flights (earlier flights get more)
        weight = max(0.3, 1.0 - (i / n_flights) * 0.6 + rng.uniform(-0.15, 0.15))
        econ = max(1, int(economy_total * weight / n_flights) + rng.randint(-2, 2))
        prem = max(0, int(premium_total * weight / n_flights) + rng.randint(-1, 1))
        biz = max(0, int(business_total * weight / n_flights) + rng.randint(-1, 1))

        # SSR support: higher difficulty = fewer SSRs per flight
        if difficulty < 0.3:
            supported_ssr = list(all_ssr)
        else:
            n_ssr = rng.randint(
                max(0, int(len(all_ssr) * (1.0 - difficulty))),
                len(all_ssr),
            )
            supported_ssr = sorted(rng.sample(all_ssr, min(n_ssr, len(all_ssr))))

        # Window/legroom features
        seat_features = {
            "economy": {
                "window": max(0, econ // 3 + rng.randint(-1, 1)),
                "legroom": max(0, econ // 5 + rng.randint(-1, 1)),
            },
            "premium_economy": {
                "window": max(0, prem // 2 + rng.randint(0, 1)),
                "legroom": max(0, prem // 2 + rng.randint(0, 1)),
            },
            "business": {
                "window": max(0, biz // 2 + rng.randint(0, 1)),
                "legroom": biz,
            },
        }

        flights.append({
            "flight_id": fid,
            "departure_time": dep_time,
            "arrival_time": arr_time,
            "cabin_availability": {
                "economy": econ,
                "premium_economy": prem,
                "business": biz,
            },
            "seat_features": seat_features,
            "supports_ssr": supported_ssr,
        })

    # Ensure at least one flight supports each SSR (so the problem is solvable at low difficulty)
    if difficulty < 0.7:
        for ssr in all_ssr:
            if not any(ssr in fl["supports_ssr"] for fl in flights):
                target = rng.choice(flights)
                target["supports_ssr"].append(ssr)
                target["supports_ssr"].sort()

    return flights


# ---------------------------------------------------------------------------
# Passenger generation
# ---------------------------------------------------------------------------

def _generate_passengers(
    rng: random.Random,
    n_passengers: int,
    flights: List[dict],
    difficulty: float,
    ssr_density: float,
    group_density: float,
    deadline_density: float,
) -> List[dict]:
    """Generate passengers with varied constraints."""

    used_names: set = set()
    passengers: List[dict] = []

    # Pre-compute groups
    n_in_groups = int(n_passengers * group_density)
    groups = _make_groups(rng, n_in_groups, difficulty)
    group_assignments = {}  # pax_index -> (group_id, integrity, size)
    idx = 0
    for gid, (integrity, size) in groups.items():
        for _ in range(size):
            if idx < n_passengers:
                group_assignments[idx] = (gid, integrity, size)
                idx += 1

    # Shuffle which indices get groups
    group_indices = list(group_assignments.keys())
    non_group_indices = [i for i in range(n_passengers) if i not in group_assignments]
    # Re-assign group data to random indices
    shuffled_group_indices = rng.sample(range(n_passengers), len(group_indices))
    new_group_assignments = {}
    for new_idx, old_idx in zip(shuffled_group_indices, group_indices):
        new_group_assignments[new_idx] = group_assignments[old_idx]
    group_assignments = new_group_assignments

    # Compute flight time range for deadlines
    latest_arrival = max(
        _parse_time(fl["arrival_time"]) for fl in flights
    )
    earliest_arrival = min(
        _parse_time(fl["arrival_time"]) for fl in flights
    )

    for i in range(n_passengers):
        pid = f"PAX-{i + 1:03d}"

        # Name
        name = _unique_name(rng, used_names)
        used_names.add(name)

        # Priority tier (weighted: more mid-tiers)
        tier = rng.choices([1, 2, 3, 4, 5], weights=[8, 15, 30, 25, 22])[0]

        # Original cabin (correlated with tier)
        if tier <= 2:
            cabin = rng.choices(CABINS, weights=[15, 30, 55])[0]
        elif tier == 3:
            cabin = rng.choices(CABINS, weights=[40, 35, 25])[0]
        else:
            cabin = rng.choices(CABINS, weights=[65, 25, 10])[0]

        # Group
        if i in group_assignments:
            gid, integrity, gsize = group_assignments[i]
        else:
            gid, integrity, gsize = None, None, None

        # SSR flags
        ssr_flags = []
        if rng.random() < ssr_density:
            n_ssr = rng.choices([1, 2], weights=[80, 20])[0]
            # Avoid conflicting SSRs (pet_cabin + pet_cargo together is unlikely)
            pool = list(SSR_TYPES)
            chosen = []
            for _ in range(n_ssr):
                if not pool:
                    break
                s = rng.choice(pool)
                chosen.append(s)
                pool.remove(s)
                # Remove conflicting
                if s == "pet_cabin" and "pet_cargo" in pool:
                    pool.remove("pet_cargo")
                elif s == "pet_cargo" and "pet_cabin" in pool:
                    pool.remove("pet_cabin")
            ssr_flags = sorted(chosen)

        # Deadline
        deadline = None
        if rng.random() < deadline_density:
            # Set deadline between earliest and latest arrival + some buffer
            buffer = rng.randint(0, 60)
            dl_minutes = rng.randint(earliest_arrival + 30, latest_arrival + buffer)
            dl_h = dl_minutes // 60
            dl_m = dl_minutes % 60
            deadline = f"{dl_h:02d}:{dl_m:02d}"

        # Loyalty
        if tier == 1:
            loyalty = rng.choices(LOYALTY_LEVELS, weights=[5, 25, 70])[0]
        elif tier == 2:
            loyalty = rng.choices(LOYALTY_LEVELS, weights=[20, 45, 35])[0]
        elif tier == 3:
            loyalty = rng.choices(LOYALTY_LEVELS, weights=[50, 35, 15])[0]
        else:
            loyalty = rng.choices(LOYALTY_LEVELS, weights=[75, 20, 5])[0]

        # Preferences
        paid_window = rng.random() < 0.25
        paid_legroom = rng.random() < 0.20

        passengers.append({
            "passenger_id": pid,
            "name": name,
            "priority_tier": tier,
            "original_cabin": cabin,
            "group_id": gid,
            "group_integrity": integrity,
            "group_size": gsize,
            "ssr_flags": ssr_flags,
            "downstream_deadline": deadline,
            "loyalty_status": loyalty,
            "paid_window": paid_window,
            "paid_legroom": paid_legroom,
        })

    return passengers


def _make_groups(
    rng: random.Random, n_in_groups: int, difficulty: float
) -> Dict[str, Tuple[str, int]]:
    """Create group specs. Returns {group_id: (integrity, size)}."""
    groups = {}
    remaining = n_in_groups
    gid_counter = 1

    while remaining >= 2:
        size = rng.choices([2, 3, 4], weights=[50, 35, 15])[0]
        size = min(size, remaining)
        if size < 2:
            break

        # Higher difficulty = more hard groups
        hard_prob = _lerp(difficulty, 0.2, 0.6)
        integrity = "hard" if rng.random() < hard_prob else "soft"

        gid = f"GRP-{gid_counter:03d}"
        groups[gid] = (integrity, size)
        gid_counter += 1
        remaining -= size

    return groups


# ---------------------------------------------------------------------------
# Adversarial injection
# ---------------------------------------------------------------------------

def _inject_adversarial(
    rng: random.Random,
    passengers: List[dict],
    flights: List[dict],
    difficulty: float,
) -> None:
    """Inject adversarial patterns into the data."""

    # --- Greedy trap: make the "obvious" best flight scarce for critical passengers ---
    # Find the earliest flight (most attractive to greedy agents)
    flights_sorted = sorted(flights, key=lambda f: _parse_time(f["departure_time"]))
    if len(flights_sorted) >= 2:
        best_flight = flights_sorted[0]
        # Reduce its business seats to create pressure
        if difficulty >= 0.5:
            best_flight["cabin_availability"]["business"] = max(
                1, best_flight["cabin_availability"]["business"] // 2
            )

    # --- Distractor flight: a flight with lots of seats but no SSR support ---
    if difficulty >= 0.5 and len(flights) >= 3:
        distractor = rng.choice(flights[1:])
        # Boost seats but strip SSR
        distractor["cabin_availability"]["economy"] += rng.randint(3, 8)
        if difficulty >= 0.6:
            distractor["supports_ssr"] = []

    # --- Priority inversion trap: low-tier pax with critical SSR ---
    if difficulty >= 0.4:
        # Find a low-priority passenger and give them a rare SSR
        low_tier_pax = [p for p in passengers if p["priority_tier"] >= 4 and not p["ssr_flags"]]
        if low_tier_pax:
            target = rng.choice(low_tier_pax)
            rare_ssr = _find_rarest_ssr(flights)
            if rare_ssr:
                target["ssr_flags"] = [rare_ssr]


def _find_rarest_ssr(flights: List[dict]) -> Optional[str]:
    """Find the SSR supported by the fewest flights."""
    counts: Dict[str, int] = {s: 0 for s in SSR_TYPES}
    for fl in flights:
        for s in fl["supports_ssr"]:
            counts[s] = counts.get(s, 0) + 1
    # Return the rarest that has at least 1 supporting flight
    valid = {s: c for s, c in counts.items() if c > 0}
    if not valid:
        return None
    return min(valid, key=valid.get)


# ---------------------------------------------------------------------------
# Pareto conflict injection
# ---------------------------------------------------------------------------

def _inject_pareto_conflicts(
    rng: random.Random,
    passengers: List[dict],
    flights: List[dict],
    difficulty: float,
) -> None:
    """
    Engineer data so that satisfying all constraints simultaneously is impossible.
    Forces the agent to make trade-offs.
    """

    # --- Conflict 1: SSR passengers outnumber SSR-compatible cabin seats ---
    # Find passengers with SSR flags
    ssr_pax = [p for p in passengers if p["ssr_flags"]]
    if len(ssr_pax) >= 3 and difficulty >= 0.6:
        # Pick the most common SSR among passengers
        ssr_counts: Dict[str, int] = {}
        for p in ssr_pax:
            for s in p["ssr_flags"]:
                ssr_counts[s] = ssr_counts.get(s, 0) + 1
        if ssr_counts:
            most_common_ssr = max(ssr_counts, key=ssr_counts.get)
            pax_with_ssr = [p for p in ssr_pax if most_common_ssr in p["ssr_flags"]]

            # Count total business seats on flights supporting this SSR
            compatible_flights = [
                fl for fl in flights if most_common_ssr in fl["supports_ssr"]
            ]
            total_biz = sum(fl["cabin_availability"]["business"] for fl in compatible_flights)

            # If there are more business-class SSR pax than business seats, conflict exists
            biz_ssr_pax = [p for p in pax_with_ssr if p["original_cabin"] == "business"]
            if len(biz_ssr_pax) > total_biz and total_biz > 0:
                pass  # Already a conflict
            elif len(biz_ssr_pax) >= 2 and compatible_flights:
                # Create the conflict by reducing business seats
                for fl in compatible_flights:
                    fl["cabin_availability"]["business"] = max(
                        0, fl["cabin_availability"]["business"] - 1
                    )

    # --- Conflict 2: Hard group needs more seats than any single flight has ---
    hard_groups = {}
    for p in passengers:
        if p["group_id"] and p["group_integrity"] == "hard":
            hard_groups.setdefault(p["group_id"], []).append(p)

    if hard_groups and difficulty >= 0.65:
        largest_group_id = max(hard_groups, key=lambda g: len(hard_groups[g]))
        group = hard_groups[largest_group_id]
        group_cabin = group[0]["original_cabin"]

        # Reduce capacity on all flights so no single flight can hold the whole group
        for fl in flights:
            current = fl["cabin_availability"].get(group_cabin, 0)
            if current >= len(group):
                fl["cabin_availability"][group_cabin] = max(
                    1, len(group) - 1
                )

    # --- Conflict 3: Deadline passengers competing for the same early flight ---
    deadline_pax = [p for p in passengers if p["downstream_deadline"]]
    if len(deadline_pax) >= 2 and difficulty >= 0.5:
        # Find the earliest flight
        earliest = min(flights, key=lambda f: _parse_time(f["arrival_time"]))
        # Tighten deadlines so only the earliest flight works
        earliest_arr = _parse_time(earliest["arrival_time"])
        for p in deadline_pax[:min(3, len(deadline_pax))]:
            tight_dl = earliest_arr + rng.randint(5, 30)
            p["downstream_deadline"] = f"{tight_dl // 60:02d}:{tight_dl % 60:02d}"
        # Reduce seats on that flight
        total_deadline_pax = min(3, len(deadline_pax))
        for cabin in CABINS:
            current = earliest["cabin_availability"].get(cabin, 0)
            if current > total_deadline_pax:
                earliest["cabin_availability"][cabin] = max(
                    1, total_deadline_pax - 1
                )


# ---------------------------------------------------------------------------
# Mid-episode event generation
# ---------------------------------------------------------------------------

def generate_events(
    rng: random.Random,
    passengers: List[dict],
    flights: List[dict],
    max_steps: int,
    difficulty: float,
) -> List[dict]:
    """
    Generate a list of mid-episode events that fire at specific steps.

    Event types:
    - capacity_change: a flight gains or loses seats
    - new_passenger: a new passenger is injected
    - ssr_equipment_failure: a flight loses SSR support
    - deadline_shift: a passenger's deadline changes
    - secondary_cancellation: a flight is removed entirely
    """
    events = []
    d = difficulty

    # Number of events: 0 at low difficulty, up to 4 at high
    n_events = max(0, int(_lerp(d, 0, 4) + rng.uniform(-0.5, 0.5)))
    if n_events == 0:
        return events

    # Pick event steps (not too early, not at the very end)
    min_step = max(3, max_steps // 5)
    max_step = max_steps - 3
    if min_step >= max_step:
        return events

    event_steps = sorted(rng.sample(
        range(min_step, max_step),
        min(n_events, max_step - min_step),
    ))

    event_types = ["capacity_change", "new_passenger", "ssr_equipment_failure",
                   "deadline_shift", "secondary_cancellation"]

    for step in event_steps:
        etype = rng.choices(
            event_types,
            weights=[30, 20, 20, 20, 10],
        )[0]

        event: Dict[str, Any] = {"step": step, "type": etype}

        if etype == "capacity_change":
            fl = rng.choice(flights)
            cabin = rng.choice(CABINS)
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
            event["flight_id"] = fl["flight_id"]
            event["cabin"] = cabin
            event["delta"] = delta
            event["reason"] = (
                f"Crew deadheading claimed {abs(delta)} {cabin} seat(s)"
                if delta < 0
                else f"Larger aircraft swapped in: +{delta} {cabin} seat(s)"
            )

        elif etype == "new_passenger":
            new_pid = f"PAX-NEW-{step:03d}"
            tier = rng.choices([1, 2, 3], weights=[50, 30, 20])[0]
            cabin = rng.choices(CABINS, weights=[30, 30, 40])[0]
            event["passenger"] = {
                "passenger_id": new_pid,
                "name": f"Emergency Pax {step}",
                "priority_tier": tier,
                "original_cabin": cabin,
                "group_id": None,
                "group_integrity": None,
                "group_size": None,
                "ssr_flags": [],
                "downstream_deadline": None,
                "loyalty_status": rng.choice(LOYALTY_LEVELS),
                "paid_window": False,
                "paid_legroom": False,
            }
            event["reason"] = (
                f"Missed connection passenger {new_pid} added (tier {tier}, {cabin})"
            )

        elif etype == "ssr_equipment_failure":
            fl = rng.choice(flights)
            if fl["supports_ssr"]:
                lost_ssr = rng.choice(fl["supports_ssr"])
                event["flight_id"] = fl["flight_id"]
                event["lost_ssr"] = lost_ssr
                event["reason"] = (
                    f"{lost_ssr} equipment failure on {fl['flight_id']}"
                )
            else:
                event["type"] = "capacity_change"
                event["flight_id"] = fl["flight_id"]
                event["cabin"] = "economy"
                event["delta"] = -2
                event["reason"] = "Equipment issue reduced economy capacity"

        elif etype == "deadline_shift":
            deadline_pax = [p for p in passengers if p["downstream_deadline"]]
            if deadline_pax:
                target = rng.choice(deadline_pax)
                old_dl = target["downstream_deadline"]
                shift = rng.choice([-30, -15, 15, 30, 60])
                old_minutes = _parse_time(old_dl)
                new_minutes = max(0, old_minutes + shift)
                new_dl = f"{new_minutes // 60:02d}:{new_minutes % 60:02d}"
                event["passenger_id"] = target["passenger_id"]
                event["old_deadline"] = old_dl
                event["new_deadline"] = new_dl
                event["reason"] = (
                    f"Connecting flight {'delayed' if shift > 0 else 'advanced'}: "
                    f"{target['passenger_id']} deadline {old_dl} -> {new_dl}"
                )
            else:
                event["type"] = "capacity_change"
                fl = rng.choice(flights)
                event["flight_id"] = fl["flight_id"]
                event["cabin"] = "economy"
                event["delta"] = -1
                event["reason"] = "Minor capacity adjustment"

        elif etype == "secondary_cancellation":
            if len(flights) > 2:
                cancelled = rng.choice(flights[1:])  # Don't cancel the first flight
                event["flight_id"] = cancelled["flight_id"]
                event["reason"] = (
                    f"Flight {cancelled['flight_id']} cancelled (mechanical issue)"
                )
            else:
                event["type"] = "capacity_change"
                fl = rng.choice(flights)
                event["flight_id"] = fl["flight_id"]
                event["cabin"] = "economy"
                event["delta"] = -3
                event["reason"] = "Severe capacity reduction on flight"

        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp(d: float, low: float, high: float) -> float:
    return low + d * (high - low)


def _lerp_int(d: float, low: int, high: int, rng: random.Random) -> int:
    base = low + d * (high - low)
    return max(low, min(high, int(base) + rng.randint(-2, 2)))


def _parse_time(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _unique_name(rng: random.Random, used: set) -> str:
    for _ in range(100):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if name not in used:
            return name
    # Fallback
    return f"Passenger {rng.randint(1000, 9999)}"
