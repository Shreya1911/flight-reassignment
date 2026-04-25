"""
Build a prompt dataset for GRPO training.

Each prompt is a conversation start: system message + initial task description.
The environment_factory handles the actual environment interaction during training.

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

from inference import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Difficulty schedule (more varied than SFT to explore the full space)
# ---------------------------------------------------------------------------

DIFFICULTIES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def build_prompt_dataset(
    n_prompts: int = 5000,
    output_dir: str = "training/grpo_prompts",
) -> None:
    """
    Build a dataset of initial prompts for GRPO training.

    Each row has:
        - prompt: list of messages (system + user)
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

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "A flight has been cancelled. Rebook all passengers onto "
                "alternative flights, respecting constraints and priorities.\n\n"
                f"=== Step 0 | Episode seed: {seed} | Difficulty: {difficulty} ===\n\n"
                "Call list_passengers to begin."
            )},
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
