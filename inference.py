import asyncio
import json
import os
import re
import subprocess
import sys
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "openai>=1.0.0", "--quiet"],
            timeout=120,
        )
    except Exception:
        pass
    from openai import OpenAI

try:
    from client import FlightRebookingEnv
    from models import FlightRebookingAction
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import FlightRebookingEnv
    from models import FlightRebookingAction

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
IMAGE_NAME = os.getenv("IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL") or os.getenv("SERVER_URL") or "http://localhost:8000"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BENCHMARK = "flight_rebooking"
TEMPERATURE = 0.3
MAX_TOKENS = 1000
RESULTS_DIR = os.getenv("RESULTS_DIR") or "results"

# ---------------------------------------------------------------------------
# System prompt (constant across all difficulty tiers)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an airline operations agent. A scheduled flight has been cancelled and all passengers must be rebooked onto alternative flights to the same destination. You operate at the inventory level — placing passengers into available cabin buckets on flights, NOT assigning specific seats.

YOUR GOAL:
Produce a rebooking plan that gets every passenger to their destination while respecting constraints, managing costs, and treating loyalty members fairly. You must handle trade-offs — it may be impossible to satisfy all constraints simultaneously.

CONSTRAINT PRIORITY (highest to lowest):
1. HARD CONSTRAINTS (must not violate):
   - SSR compatibility: passengers with special service requirements (UM, WCHR, pet_cabin, pet_cargo) can only go on flights that support those SSRs.
   - Hard group integrity: passengers in a "hard" group must all be on the same flight.
   - Downstream deadlines: if a passenger has a connection deadline, their new flight must arrive by that time.

2. COVERAGE: every passenger should be rebooked onto some flight.

3. COST EFFICIENCY: bookings have costs. Upgrades cost the airline money; downgrades require compensation. Stay within the compensation budget. Avoid unnecessary upgrades.

4. LOYALTY COMPLIANCE: gold/silver members should not be downgraded if avoidable. Gold members downgraded incur extra compensation (lounge + meal). Treat loyalty members with priority when making trade-offs.

5. CABIN MATCHING: place passengers in their original cabin class when possible.

6. PRIORITY TIERS: higher-priority passengers (tier 1 is highest, tier 5 is lowest) should get better outcomes when trade-offs are needed.

7. SOFT GROUP INTEGRITY: passengers in a "soft" group should be kept together when possible, but splitting is acceptable.

TRADE-OFF REASONING:
Not all constraints can always be satisfied. When conflicts arise:
- Prefer violating soft constraints over hard constraints.
- A tier-5 passenger with a critical SSR may need to be booked before a tier-1 passenger without constraints.
- Downgrading a gold member is worse than downgrading a non-loyalty passenger.
- Spending $800 to upgrade one passenger may not be worth it if it exhausts the compensation budget.
- Sometimes unbooking an earlier decision is the right call if circumstances change.

MID-EPISODE EVENTS:
The environment may inject events during the episode:
- Flight capacity changes (crew deadheading, aircraft swaps)
- New passengers added (missed connections)
- SSR equipment failures (flight loses support for a service)
- Deadline shifts (connecting flights delayed or advanced)
- Secondary flight cancellations (passengers on that flight become unbooked)

When events occur, they appear in the observation. You must adapt — check what changed, assess impact on existing bookings, unbook/rebook affected passengers if needed.

TOOLS AVAILABLE (8 tools):
Each turn you must call exactly one tool.

1. list_passengers()
   - Returns summary: ID, priority tier, group ID, loyalty status, SSR/deadline flags.

2. get_passenger_details(passenger_id)
   - Returns full details including loyalty status, paid preferences, exact SSR flags.

3. list_alternative_flights()
   - Returns all active flights with per-cabin seat counts, times, SSR support.
   - Cancelled flights are excluded. Call again to refresh after events.

4. get_flight_details(flight_id)
   - Returns details for one specific flight including current availability.

5. book_passenger(passenger_id, flight_id, cabin)
   - Books one passenger. Returns booking cost (upgrade/downgrade/compensation).
   - Will be rejected if: no seats, SSR mismatch, deadline violation, already booked, flight cancelled.

6. book_group(group_id, flight_id, cabin_assignments)
   - Books an entire group atomically. Returns total group cost.

7. unbook_passenger(passenger_id)
   - Removes an existing booking, freeing the seat back to inventory.
   - Use when events invalidate a booking, or to make room for a higher-priority passenger.
   - Incurs a small disruption penalty.

8. finalize_plan()
   - Call when done. Triggers final scoring.

ACTION FORMAT:
Respond with ONLY a raw JSON object. No reasoning, no markdown, no extra text.
Examples:
{"tool_name": "list_passengers", "args": {}}
{"tool_name": "get_passenger_details", "args": {"passenger_id": "PAX-001"}}
{"tool_name": "list_alternative_flights", "args": {}}
{"tool_name": "get_flight_details", "args": {"flight_id": "FL-201"}}
{"tool_name": "book_passenger", "args": {"passenger_id": "PAX-001", "flight_id": "FL-201", "cabin": "business"}}
{"tool_name": "book_group", "args": {"group_id": "GRP-001", "flight_id": "FL-201", "cabin_assignments": {"PAX-002": "economy", "PAX-003": "economy"}}}
{"tool_name": "unbook_passenger", "args": {"passenger_id": "PAX-001"}}
{"tool_name": "finalize_plan", "args": {}}

STRATEGY:
1. Start with list_passengers and list_alternative_flights to survey the situation.
2. Identify constrained passengers: SSR flags, deadlines, hard groups, loyalty status.
3. Assess capacity scarcity: which cabins/flights are tight? Which SSRs are rare?
4. Book the most constrained passengers first (hard groups, SSR+deadline combos).
5. Consider cost: match cabin when possible, avoid unnecessary upgrades.
6. Protect loyalty members from downgrades when alternatives exist.
7. Use book_group for groups (especially hard groups) to keep them together atomically.
8. After events, check if existing bookings are still valid. Use unbook_passenger if needed.
9. When all passengers are booked (or you've done your best), call finalize_plan.
10. If a booking fails, analyze why and adapt — try a different flight, cabin, or booking order.

STEP BUDGET IS TIGHT — be efficient. Don't inspect every passenger if the summary tells you enough. Prioritize investigation of constrained passengers.

IMPORTANT:
- Hard constraints have severe penalties.
- The grader evaluates: coverage, cabin match, group integrity, deadlines, SSR integrity, cost efficiency, and loyalty compliance.
- Unbooked passengers hurt your score, but violating hard constraints hurts more.
- Cost overruns and loyalty mistreatment are graded separately."""

# ---------------------------------------------------------------------------
# Task definitions: (task_name, task_id, max_steps)
# ---------------------------------------------------------------------------
TASKS = [
    ("task_easy",   "easy",   20),
    ("task_medium", "medium", 35),
    ("task_hard",   "hard",   55),
]

# ---------------------------------------------------------------------------
# State and prompt formatting
# ---------------------------------------------------------------------------

def format_main_task(task_id: str) -> str:
    return (
        "Task: A flight has been cancelled. Rebook all passengers onto "
        "alternative flights, respecting constraints and priorities."
    )


def format_state(obs) -> str:
    parts = [
        f"=== Step {obs.step_count}/{obs.max_steps} | "
        f"Booked: {obs.passengers_booked}/{obs.passengers_total} | "
        f"Remaining: {obs.passengers_remaining} | "
        f"Cost: ${obs.total_cost:.0f} (budget: ${obs.compensation_budget:.0f}) ==="
    ]

    # Show mid-episode events if any fired this step
    if obs.events:
        parts.append("\n** EVENTS THIS STEP **")
        for evt in obs.events:
            parts.append(f"  [{evt['type']}] {evt.get('reason', '')}")
            if evt["type"] == "secondary_cancellation" and "unbooked_passengers" in evt:
                parts.append(f"    Passengers unbooked: {evt['unbooked_passengers']}")
        parts.append("** Check bookings and adapt. **")

    # Show reward breakdown if available
    if obs.reward_breakdown:
        bd = obs.reward_breakdown
        non_zero = {k: v for k, v in bd.items() if v != 0.0}
        if non_zero:
            parts.append(f"\nReward breakdown: {non_zero}")

    if obs.booked_summary:
        parts.append("\nCurrent bookings:")
        for b in obs.booked_summary:
            parts.append(f"  {b['passenger_id']} -> {b['flight_id']} ({b['cabin']})")

    if obs.flights_snapshot:
        parts.append("\nFlight availability:")
        for fl in obs.flights_snapshot:
            avail = ", ".join(
                f"{c}={n}" for c, n in fl["cabin_availability"].items() if n > 0
            )
            parts.append(
                f"  {fl['flight_id']} dep={fl['departure_time']} arr={fl['arrival_time']} "
                f"SSR={fl['supports_ssr']} [{avail}]"
            )

    if obs.passengers_remaining > 0 and obs.booked_summary:
        booked_pids = {b["passenger_id"] for b in obs.booked_summary}
        # The model needs to know WHO is remaining, not just the count
        parts.append(f"\nBooked pids: {booked_pids}")
        parts.append(f"\nStill need booking: {obs.passengers_remaining} passengers (checklist_passengers)")

    return "\n".join(parts)


def format_instruction() -> str:
    return "Choose your next tool call. Respond with ONLY a JSON object."


def format_result(item: dict) -> str:
    parts = []
    if item.get("result") is not None:
        parts.append(f"Last tool result: {json.dumps(item['result'], indent=2)}")
    if item.get("reward") is not None:
        parts.append(
            f"Reward: {item['reward']:.2f} ({item.get('reward_reason', 'unknown')})"
        )
    if not parts:
        return "Tool executed."
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> Optional[dict]:
    """Parse LLM response into a tool call dict. Returns None on failure."""
    text = response_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if "tool_name" in parsed and "args" in parsed:
            return parsed
        return None
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"tool_name"[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI, obs, conversation_history: list, task_id: str
) -> Optional[dict]:
    """Call the LLM to get the next action."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if not conversation_history:
        user_content = "\n\n".join([
            format_main_task(task_id),
            format_state(obs),
            format_instruction(),
        ])
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": format_main_task(task_id)})

        recent_history = conversation_history[-12:]
        for i, item in enumerate(recent_history):
            messages.append(
                {"role": "assistant", "content": json.dumps(item["action"])}
            )

            user_parts = [format_result(item)]
            if i == len(recent_history) - 1:
                user_parts.append(format_state(obs))
                user_parts.append(format_instruction())

            messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(response_text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fallback action
# ---------------------------------------------------------------------------

def fallback_action(obs) -> dict:
    """Simple fallback: list passengers if nothing booked yet, else list flights."""
    if obs.passengers_booked == 0:
        return {"tool_name": "list_passengers", "args": {}}
    return {"tool_name": "list_alternative_flights", "args": {}}


# ---------------------------------------------------------------------------
# Mandatory logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def save_assignments(
    task_name: str,
    task_id: str,
    obs,
    steps_taken: int,
    score: float,
    rewards: List[float],
) -> None:
    """Save final passenger assignments and episode summary to a JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build per-passenger assignment records from booked_summary
    assignments = []
    booked_pids = set()
    if obs and obs.booked_summary:
        for b in obs.booked_summary:
            assignments.append({
                "passenger_id": b["passenger_id"],
                "flight_id": b["flight_id"],
                "cabin": b["cabin"],
                "status": "booked",
            })
            booked_pids.add(b["passenger_id"])

    # Mark unbooked passengers
    n_unbooked = (obs.passengers_total - len(booked_pids)) if obs else 0
    # We don't have IDs for unbooked passengers from the observation alone,
    # so record the count in the summary.

    # Terminal breakdown and grader score from tool_result
    terminal_breakdown = None
    if obs and obs.tool_result and "terminal_breakdown" in obs.tool_result:
        terminal_breakdown = obs.tool_result["terminal_breakdown"]

    result = {
        "task_name": task_name,
        "task_id": task_id,
        "model": MODEL_NAME,
        "score": round(score, 4),
        "success": score >= 0.5,
        "steps_taken": steps_taken,
        "passengers_total": obs.passengers_total if obs else 0,
        "passengers_booked": len(booked_pids),
        "passengers_unbooked": n_unbooked,
        "total_cost": round(obs.total_cost, 2) if obs else 0.0,
        "compensation_budget": obs.compensation_budget if obs else 0.0,
        "cumulative_reward": round(sum(rewards), 4),
        "terminal_breakdown": terminal_breakdown,
        "assignments": assignments,
    }

    filepath = os.path.join(RESULTS_DIR, f"{task_id}_assignments.json")
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[SAVE] {filepath} ({len(assignments)} assignments)", flush=True)


# ---------------------------------------------------------------------------
# Per-task episode runner
# ---------------------------------------------------------------------------

async def run_task(
    task_name: str,
    task_id: str,
    max_steps: int,
    client: OpenAI,
) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    conversation_history: list = []
    last_obs = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = None
    try:
        env = FlightRebookingEnv(base_url=ENV_URL)
        result = await env.reset(task_id=task_id)
        obs = result.observation

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_dict = get_agent_action(client, obs, conversation_history, task_id)

            if action_dict is None:
                action_dict = fallback_action(obs)

            action_summary = (
                f"{action_dict['tool_name']}({json.dumps(action_dict['args'])})"
            )

            result = await env.step(
                FlightRebookingAction(
                    tool_name=action_dict["tool_name"],
                    args=action_dict["args"],
                )
            )
            obs = result.observation
            last_obs = obs
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            error = (
                obs.tool_result.get("message")
                if obs.tool_result and obs.tool_result.get("status") == "error"
                else None
            )
            log_step(
                step=step, action=action_summary,
                reward=reward, done=done, error=error,
            )

            conversation_history.append({
                "action": action_dict,
                "result": obs.tool_result,
                "reward": reward,
                "reward_reason": obs.reward_reason,
            })

            if done:
                if obs.tool_result and "grader_score" in obs.tool_result:
                    score = obs.tool_result["grader_score"]
                break

        success = score >= 0.5
        score = min(max(score, 0.0), 1.0)

    except Exception:
        pass

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        if last_obs is not None:
            try:
                save_assignments(
                    task_name, task_id, last_obs,
                    steps_taken, score, rewards,
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main — loop over all tasks
# ---------------------------------------------------------------------------

async def main() -> None:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[ERROR] Failed to create OpenAI client: {exc}", flush=True)
        return
    for task_name, task_id, max_steps in TASKS:
        try:
            await run_task(task_name, task_id, max_steps, client)
        except Exception as exc:
            print(f"[ERROR] task={task_name} failed: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
