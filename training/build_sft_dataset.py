"""
Build a HuggingFace Dataset from collected expert episodes for SFTTrainer.

Converts episode JSON files into the conversational format expected by TRL:
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": observation_text},
            {"role": "assistant", "content": '{"tool_name": "...", "args": {...}}'},
            ...
        ]
    }

Usage:
    python -m training.build_sft_dataset \
        --episodes_dir data/sft_episodes \
        --output_dir training/sft_dataset \
        --min_score 0.8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference import SYSTEM_PROMPT


def load_episodes(episodes_dir: str, min_score: float = 0.8) -> List[dict]:
    """Load and filter episode JSON files."""
    episodes = []
    ep_dir = Path(episodes_dir)

    if not ep_dir.is_dir():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")

    for filepath in sorted(ep_dir.glob("*.json")):
        try:
            with open(filepath) as f:
                episode = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping {filepath.name}: {e}")
            continue

        score = episode.get("score", 0.0)
        if score < min_score:
            continue

        turns = episode.get("turns", [])
        if len(turns) < 2:
            continue

        episodes.append(episode)

    return episodes


def episode_to_messages(episode: dict) -> List[dict]:
    """
    Convert an episode dict to a list of chat messages.

    Format:
        [system, user, assistant, user, assistant, ..., assistant]

    The first user message contains the initial observation.
    Subsequent user messages contain tool results + updated state.
    Each assistant message is the JSON tool call.
    """
    turns = episode.get("turns", [])
    if not turns:
        return []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, turn in enumerate(turns):
        # User message: the observation text
        obs_text = turn.get("observation_text", "")
        if obs_text:
            messages.append({"role": "user", "content": obs_text})

        # Assistant message: the tool call as JSON
        action = turn.get("action", {})
        action_json = json.dumps(action, separators=(",", ":"))
        messages.append({"role": "assistant", "content": action_json})

    return messages


def build_dataset(
    episodes_dir: str,
    output_dir: str,
    min_score: float = 0.8,
    max_episodes: int = 0,
) -> None:
    """
    Build HF Dataset from episode JSONs and save to disk.

    Args:
        episodes_dir: Directory containing episode JSON files.
        output_dir: Where to save the HF Dataset.
        min_score: Minimum grader score to include (default: 0.8).
        max_episodes: Max episodes to include (0 = all).
    """
    try:
        from datasets import Dataset
    except ImportError:
        print("ERROR: `datasets` package not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    print(f"Loading episodes from {episodes_dir} (min_score={min_score})...")
    episodes = load_episodes(episodes_dir, min_score=min_score)

    if max_episodes > 0:
        episodes = episodes[:max_episodes]

    print(f"  Loaded {len(episodes)} episodes")

    if not episodes:
        print("No episodes found. Run collect_sft_data.py first.")
        return

    # Convert to messages format
    rows = []
    score_sum = 0.0
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}

    for episode in episodes:
        messages = episode_to_messages(episode)
        if not messages:
            continue

        rows.append({"messages": messages})

        score_sum += episode.get("score", 0.0)
        diff = episode.get("difficulty", 0.5)
        if diff <= 0.3:
            difficulty_counts["easy"] += 1
        elif diff <= 0.6:
            difficulty_counts["medium"] += 1
        else:
            difficulty_counts["hard"] += 1

    print(f"  Converted {len(rows)} episodes to messages format")
    print(f"  Avg score: {score_sum / len(rows):.4f}")
    print(f"  Difficulty distribution: {difficulty_counts}")

    # Message length stats
    msg_counts = [len(r["messages"]) for r in rows]
    print(f"  Messages per episode: min={min(msg_counts)}, "
          f"max={max(msg_counts)}, avg={sum(msg_counts)/len(msg_counts):.1f}")

    # Build HF Dataset
    dataset = Dataset.from_list(rows)
    print(f"\n  Dataset: {dataset}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"  Saved to {output_dir}")

    # Also save a JSONL version for inspection
    jsonl_path = os.path.join(output_dir, "preview.jsonl")
    with open(jsonl_path, "w") as f:
        for row in rows[:10]:  # first 10 for preview
            f.write(json.dumps(row) + "\n")
    print(f"  Preview saved to {jsonl_path} (first 10 episodes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build SFT dataset from expert episodes"
    )
    parser.add_argument(
        "--episodes_dir", type=str, default="data/sft_episodes",
        help="Directory with episode JSON files (default: data/sft_episodes)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="training/sft_dataset",
        help="Output directory for HF Dataset (default: training/sft_dataset)",
    )
    parser.add_argument(
        "--min_score", type=float, default=0.8,
        help="Minimum grader score to include (default: 0.8)",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=0,
        help="Maximum episodes to include, 0 = all (default: 0)",
    )

    args = parser.parse_args()

    build_dataset(
        episodes_dir=args.episodes_dir,
        output_dir=args.output_dir,
        min_score=args.min_score,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
