---
title: Seat Reassignment Environment
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# ✈️ Airline Seat Reassignment — OpenEnv Environment

**Team Agentic Troop** · [HF Space](https://huggingface.co/spaces/vansh-ar-0-ra/seat-reassignment) · [GitHub](https://github.com/vansh-ar0ra/seat_reassignment)

An OpenEnv-compliant RL environment that simulates **airline seat reassignment after an aircraft swap** — a real operational task performed daily by airline ground staff. An agent must reassign all passengers from Aircraft-1 (AC-1) to Aircraft-2 (AC-2) using tool calls, respecting cabin class, paid window-seat preferences, and paid extra-legroom preferences depending on the difficulty level.

---

## Motivation & Real-World Utility

Aircraft swaps are a routine disruption in airline operations. When a scheduled aircraft is substituted, gate agents must manually reassign every passenger to the replacement aircraft's different seating layout while honouring cabin class, paid upgrades, and accessibility needs — all under time pressure. This environment models that exact task, making it directly useful for training and evaluating agents on constrained multi-step planning with heterogeneous passenger requirements.

---

## Environment Overview

Each episode begins with a set of passengers occupying seats on AC-1. The agent must move every passenger to an appropriate seat on AC-2 (which has a different layout) using three tools. The episode ends when all passengers are reassigned or the step limit is reached.

### Action Space

The agent's action is a JSON object specifying one tool call per step:

| Tool | Arguments | Description |
|---|---|---|
| `get_passenger_details` | `seat_id` | Inspect an AC-1 seat to learn the passenger's ID, cabin, and paid preferences. |
| `assign_seat` | `passenger_id`, `target_seat_id` | Move a passenger from AC-1 to a specific empty AC-2 seat. Returns cabin match and preference satisfaction status. |
| `swap_seats` | `passenger_id_1`, `passenger_id_2` | Swap two passengers who are both already assigned on AC-2. Useful for correcting earlier misplacements. |

**Action format:** `{"tool_name": "assign_seat", "args": {"passenger_id": "PAX-001", "target_seat_id": "2A"}}`

### Observation Space

After every `reset()` and `step()`, the agent receives a `SeatReassignmentObservation` containing:

- **AC-1 and AC-2 layouts** — full seat configurations (cabin, seat type, legroom) for both aircraft.
- **AC-1 seats still occupied** — passengers not yet moved.
- **AC-2 seat assignments** — current mapping of AC-2 seats to passengers.
- **AC-2 seats available** — empty seats on AC-2.
- **Tool result** — structured output from the last tool call (success/error, cabin match, preference satisfaction).
- **Step-level feedback** — `reward`, `reward_reason`, `step_count`, `max_steps`, `cumulative_reward`, `passengers_remaining`.

All models are typed Pydantic classes (`SeatReassignmentAction`, `SeatReassignmentObservation`, `SeatReassignmentState`) inheriting from OpenEnv base types.

---

## Task Difficulty Design

The environment provides **3 tasks: easy, medium, and hard**. A critical design choice is that **difficulty is not introduced by varying the user-facing prompt** — the core instruction to the agent remains the same (reassign all passengers from AC-1 to AC-2). Instead, **difficulty is driven entirely by the underlying data**: the number of passengers, the number and type of constraints encoded in the passenger records, and the scarcity of qualifying seats on AC-2. The agent's system prompt guides it to account for whatever constraints it discovers in the data, but it is the data itself that defines how challenging each task is.

### Easy — Cabin Match Only

| Attribute | Value |
|---|---|
| Passengers | 8 |
| AC-2 seats | 10 |
| Max steps | 24 |
| Constraints | Cabin class only (business → business, economy → economy) |
| Passenger preferences | None — no `paid_window` or `paid_legroom` flags are set |
| Tools available | `get_passenger_details`, `assign_seat` |
| Grader | `grader_score = cabin_score` |

The data contains 8 passengers with no paid preferences. The agent simply needs to respect cabin class boundaries. There is seat surplus on AC-2 (10 seats for 8 passengers), making the task straightforward.

### Medium — Cabin + Window Preferences

| Attribute | Value |
|---|---|
| Passengers | 20 |
| AC-2 seats | 24 |
| Max steps | 60 |
| Constraints | Cabin class + paid window-seat preferences |
| Passenger preferences | A subset of passengers have `paid_window=True` |
| Tools available | `get_passenger_details`, `assign_seat`, `swap_seats` |
| Grader | `grader_score = (cabin_score + preference_score) / 2` |

The data now includes 20 passengers, some of whom have paid for window seats. The agent must learn each passenger's preferences (via `get_passenger_details`) and plan assignments so that paid-window passengers land on window seats — while still respecting cabin class. The `swap_seats` tool becomes available to correct mistakes.

### Hard — Cabin + Window + Legroom Preferences

| Attribute | Value |
|---|---|
| Passengers | 20 |
| AC-2 seats | 24 |
| Max steps | 60 |
| Constraints | Cabin class + paid window + paid extra legroom |
| Passenger preferences | 6 with `paid_window`, 6 with `paid_legroom`, 2 with both |
| Tools available | `get_passenger_details`, `assign_seat`, `swap_seats` |
| Grader | `grader_score = (cabin_score + preference_score) / 2`, where `preference_score` averages across both window and legroom dimensions |

The data introduces a second preference dimension (`paid_legroom`) and a deliberate scarcity constraint: AC-2 has **only 3 economy legroom seats** for 3 economy legroom-paying passengers, and legroom rows are in **different positions** between AC-1 and AC-2 (row 1 on AC-1 vs. row 2 on AC-2 for business; row 3 on AC-1 vs. seats 4A/4B/4H on AC-2 for economy). The agent must plan assignments carefully — wasting a scarce legroom seat on a non-paying passenger makes it impossible to achieve a perfect score. Two passengers have *both* `paid_window` and `paid_legroom`, requiring intersection of window-type and legroom-equipped seats — a genuine combinatorial planning challenge.

---

## Reward Design

The reward function provides **per-step signal**, not just end-of-episode scores:

| Event | Reward | Notes |
|---|---|---|
| Correct cabin + all preferences satisfied | +0.35 | Best per-assignment outcome |
| Correct cabin + no preferences applicable | +0.20 | Passenger had no paid prefs |
| Correct cabin + preferences missed | +0.10 | Cabin right, but pref not honoured |
| Cabin mismatch | −0.10 | Business ↔ economy violation |
| Successful improvement swap | +0.25 | Swap increased constraint satisfaction |
| Neutral / worsening swap | −0.05 / −0.15 | Discourages pointless or harmful swaps |
| Redundant fetch | −0.05 | Re-querying an already-fetched seat |
| Error (invalid seat, bad args, etc.) | −0.10 to −0.40 | Penalises clearly undesirable actions |
| Incomplete assignment at episode end | −1.0 | Strong penalty for leaving passengers unassigned |

**Terminal reward** is a weighted combination of cabin correctness (weight 1.5), preference satisfaction (weight 1.0), and step efficiency (weight 0.5), providing a rich multi-dimensional signal.

---

## Grader

Each task has a deterministic grader producing a score in **[0.0, 1.0]**:

- **Easy:** `grader_score = cabin_score` (fraction of passengers in the correct cabin).
- **Medium/Hard:** `grader_score = (cabin_score + preference_score) / 2`, where `preference_score` averages across all active preference dimensions (window, and legroom if applicable). Unassigned passengers count as unsatisfied for both components.

The grader is deterministic and reproducible — given the same assignment state, it always returns the same score.

---

## Baseline Scores

| Task | Model | Grader Score |
|---|---|---|
| Easy | Gemini 2.5 Pro | **1.00** |
| Medium | Gemini 2.5 Pro | **1.00** |
| Hard | Gemini 2.5 Pro | **0.96** |

Produced by `inference.py` using the OpenAI-compatible client.

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- Docker (for containerised execution)

### 1. Start the Environment Server

```bash
uv run server
```

The server starts at `http://localhost:8000` with the OpenEnv-compliant API (`reset()`, `step()`, `state()` endpoints).

### 2. Run Baseline Inference

Set the required environment variables and run:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token_here"

python inference.py
```

The script uses the **OpenAI client** to call the LLM, runs all 3 tasks (easy, medium, hard), and outputs grader scores.

### 3. Docker

```bash
docker build -t seat-reassignment .
docker run -p 8000:8000 seat-reassignment
```

---

## Project Structure

```
├── inference.py              # Baseline inference script (OpenAI client, all 3 tasks)
├── models.py                 # Typed Pydantic models (Action, Observation, State)
├── openenv.yaml              # OpenEnv spec metadata
├── Dockerfile                # Containerised deployment
├── pyproject.toml            # Dependencies and package config
├── client/                   # WebSocket client for interacting with the environment
├── server/
│   ├── app.py                # FastAPI application
│   ├── environment.py        # Core environment logic (reset, step, state)
│   ├── tools.py              # Tool implementations (get_passenger_details, assign_seat, swap_seats)
│   └── rewards.py            # Reward computation and grader
├── data/
│   ├── easy/                 # 8 passengers, no preferences
│   ├── medium/               # 20 passengers, window preferences
│   └── hard/                 # 20 passengers, window + legroom preferences
└── tests/
    ├── test_environment.py   # Integration tests for all 3 tasks
    └── test_rewards.py       # Unit tests for reward and grader logic
```

---

## OpenEnv Spec Compliance

- **Typed models:** `SeatReassignmentAction`, `SeatReassignmentObservation`, `SeatReassignmentState` — all Pydantic classes extending OpenEnv base types.
- **Endpoints:** `step(action)`, `reset(task_id)`, `state()` — fully implemented.
- **`openenv.yaml`:** Present with spec version, runtime, and port configuration.
- **Graders:** 3 tasks with deterministic graders returning scores in [0.0, 1.0].
- **Dockerfile:** Builds and runs cleanly.
- **HF Space:** Deployed and responding at [vansh-ar-0-ra/seat-reassignment](https://huggingface.co/spaces/vansh-ar-0-ra/seat-reassignment).