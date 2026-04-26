"""
Build a prompt dataset for GRPO training.

Each prompt is a conversational message list that TRL's GRPOTrainer will
format using the model's chat template. The environment_factory handles
the actual environment interaction during training.

Usage:
    python -m training.build_grpo_prompts \
        --n_prompts 5000 \
        --output_dir training/grpo_prompts
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# GRPO system prompt — domain knowledge only, no tool format instructions.
# TRL injects tool schemas and output format via the model's chat template.
# The inference.py SYSTEM_PROMPT includes ACTION FORMAT (raw JSON examples)
# which conflicts with TRL's <tool_call> format — so we use a separate prompt.
# ---------------------------------------------------------------------------

GRPO_SYSTEM_PROMPT = """You are an airline operations agent. A scheduled flight has been cancelled and all passengers must be rebooked onto alternative flights to the same destination. You operate at the inventory level — placing passengers into available cabin buckets on flights, NOT assigning specific seats.

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
# Difficulty schedule (more varied than SFT to explore the full space)
# ---------------------------------------------------------------------------

DIFFICULTIES = [0.15, 0.25, 0.35, 0.50, 0.60, 0.75, 0.90]


def build_prompt_dataset(
    n_prompts: int = 5000,
    output_dir: str = "training/grpo_prompts",
) -> None:
    """
    Build a dataset of initial prompts for GRPO training.

    Each row has:
        - prompt: list of message dicts (conversational format for TRL)
        - difficulty: float
        - seed: int
    """
    try:
        from datasets import Dataset
    except ImportError:
        print("ERROR: `datasets` package not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    rows = []
    for i in range(n_prompts):
        difficulty = DIFFICULTIES[i % len(DIFFICULTIES)]
        seed = i + 1

        user_text = (
            "Task: A flight has been cancelled. Rebook all passengers onto "
            "alternative flights, respecting constraints and priorities.\n\n"
            f"=== Step 0 | Episode seed: {seed} | Difficulty: {difficulty} ===\n\n"
            "Use the available tools to rebook passengers, then call finalize_plan when done."
        )

        # Conversational prompt for TRL's GRPOTrainer
        # TRL injects tool schemas via the chat template automatically
        prompt_messages = [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        rows.append({
            "prompt": prompt_messages,
            "difficulty": difficulty,
            "seed": seed,
        })

    dataset = Dataset.from_list(rows)

    # Stats
    diff_counts = {}
    for r in rows:
        d = r["difficulty"]
        diff_counts[d] = diff_counts.get(d, 0) + 1

    print(f"Built {len(rows)} prompts")
    print(f"  Difficulty distribution: {diff_counts}")

    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"  Saved to {output_dir}")

    # Preview
    preview_path = os.path.join(output_dir, "preview.jsonl")
    with open(preview_path, "w") as f:
        for row in rows[:5]:
            f.write(json.dumps(row) + "\n")
    print(f"  Preview saved to {preview_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build GRPO prompt dataset"
    )
    parser.add_argument(
        "--n_prompts", type=int, default=5000,
        help="Number of prompts to generate (default: 5000)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="training/grpo_prompts",
        help="Output directory (default: training/grpo_prompts)",
    )
    args = parser.parse_args()

    build_prompt_dataset(
        n_prompts=args.n_prompts,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
