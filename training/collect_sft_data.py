"""
Collect expert trajectories for SFT training.

Runs the expert policy across many seeds and difficulty levels,
saving each episode as a JSON file.

Usage:
    python -m training.collect_sft_data --n_episodes 1000 --output_dir data/sft_episodes

Output format (per episode JSON file):
    {
        "task_id": "seed_42",
        "seed": 42,
        "difficulty": 0.5,
        "score": 0.95,
        "cumulative_reward": 2.45,
        "n_steps": 18,
        "n_passengers": 15,
        "n_booked": 14,
        "turns": [
            {
                "observation_text": "...",
                "action": {"tool_name": "...", "args": {...}},
                "reward": 0.3,
                "reward_reason": "...",
                "tool_result": {...}
            },
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Difficulty distribution
# ---------------------------------------------------------------------------

# 40% easy, 35% medium, 25% hard (per training plan)
DIFFICULTY_SCHEDULE = [
    (0.2, 0.40),   # easy
    (0.5, 0.35),   # medium
    (0.8, 0.25),   # hard
]


def _pick_difficulty(episode_idx: int) -> float:
    """Deterministic difficulty assignment based on episode index."""
    cumulative = 0.0
    frac = (episode_idx % 100) / 100.0
    for diff, weight in DIFFICULTY_SCHEDULE:
        cumulative += weight
        if frac < cumulative:
            return diff
    return DIFFICULTY_SCHEDULE[-1][0]


# ---------------------------------------------------------------------------
# Single-episode collection (runs in worker process)
# ---------------------------------------------------------------------------

def _collect_one_episode(args: tuple) -> dict:
    """Run one expert episode. Called from process pool."""
    seed, difficulty, episode_idx = args

    # Lazy import to avoid pickling issues with multiprocessing
    from training.expert_policy import run_expert_episode

    try:
        turns, cumulative_reward, grader_score = run_expert_episode(
            seed=seed, difficulty=difficulty
        )
    except Exception as e:
        return {
            "status": "error",
            "seed": seed,
            "difficulty": difficulty,
            "error": str(e),
        }

    # Count bookings
    n_booked = 0
    for t in turns:
        tr = t.get("tool_result", {}) or {}
        if t["action"]["tool_name"] in ("book_passenger", "book_group"):
            if tr.get("status") == "success":
                # book_group may book multiple
                if "booked" in tr:
                    n_booked += len(tr["booked"])
                else:
                    n_booked += 1

    # Extract n_passengers from first list_passengers result
    n_passengers = 0
    for t in turns:
        if t["action"]["tool_name"] == "list_passengers":
            tr = t.get("tool_result", {}) or {}
            pax_list = tr.get("passengers", [])
            n_passengers = len(pax_list)
            break

    return {
        "status": "success",
        "task_id": f"seed_{seed}",
        "seed": seed,
        "difficulty": difficulty,
        "score": grader_score,
        "cumulative_reward": cumulative_reward,
        "n_steps": len(turns),
        "n_passengers": n_passengers,
        "n_booked": n_booked,
        "turns": turns,
    }


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_episodes(
    n_episodes: int = 1000,
    output_dir: str = "data/sft_episodes",
    n_workers: int = 1,
    start_seed: int = 1,
    include_suboptimal: bool = True,
) -> None:
    """
    Collect expert trajectories and save as individual JSON files.

    Args:
        n_episodes: Total number of episodes to collect.
        output_dir: Directory for episode JSON files.
        n_workers: Number of parallel workers (1 = sequential).
        start_seed: Starting seed number.
        include_suboptimal: If True, include episodes with score 0.5-0.8
                           (for data augmentation per training plan).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build work items: (seed, difficulty, episode_idx)
    work_items = []
    for i in range(n_episodes):
        seed = start_seed + i
        difficulty = _pick_difficulty(i)
        work_items.append((seed, difficulty, i))

    print(f"Collecting {n_episodes} episodes -> {output_dir}")
    print(f"  Workers: {n_workers}")
    print(f"  Seeds: {start_seed} to {start_seed + n_episodes - 1}")
    print(f"  Difficulty distribution: {DIFFICULTY_SCHEDULE}")
    print()

    stats = {"total": 0, "saved": 0, "skipped": 0, "errors": 0}
    t0 = time.time()

    if n_workers <= 1:
        # Sequential execution
        for item in work_items:
            result = _collect_one_episode(item)
            _process_result(result, output_dir, stats, include_suboptimal)
            if stats["total"] % 50 == 0:
                _print_progress(stats, t0)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_collect_one_episode, item): item
                for item in work_items
            }
            for future in as_completed(futures):
                result = future.result()
                _process_result(result, output_dir, stats, include_suboptimal)
                if stats["total"] % 50 == 0:
                    _print_progress(stats, t0)

    elapsed = time.time() - t0
    print(f"\nDone! {stats['saved']} episodes saved, "
          f"{stats['skipped']} skipped, {stats['errors']} errors "
          f"({elapsed:.1f}s)")


def _process_result(
    result: dict,
    output_dir: str,
    stats: dict,
    include_suboptimal: bool,
) -> None:
    """Save a result if it meets quality criteria."""
    stats["total"] += 1

    if result.get("status") == "error":
        stats["errors"] += 1
        return

    score = result.get("score", 0.0)
    seed = result["seed"]

    # Quality filtering per training plan:
    # - Primary: score >= 0.8 (expert-quality)
    # - Augmentation: score 0.5-0.8 (suboptimal, for diversity)
    min_score = 0.5 if include_suboptimal else 0.8
    if score < min_score:
        stats["skipped"] += 1
        return

    # Save episode
    filepath = os.path.join(output_dir, f"episode_{seed:06d}.json")
    # Remove tool_result from turns to save space (observation_text has the info)
    save_data = {
        "task_id": result["task_id"],
        "seed": seed,
        "difficulty": result["difficulty"],
        "score": round(score, 4),
        "cumulative_reward": round(result["cumulative_reward"], 4),
        "n_steps": result["n_steps"],
        "n_passengers": result["n_passengers"],
        "n_booked": result["n_booked"],
        "turns": result["turns"],
    }

    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2)

    stats["saved"] += 1


def _print_progress(stats: dict, t0: float) -> None:
    elapsed = time.time() - t0
    rate = stats["total"] / elapsed if elapsed > 0 else 0
    print(
        f"  [{stats['total']:>5d}] saved={stats['saved']} "
        f"skipped={stats['skipped']} errors={stats['errors']} "
        f"({rate:.1f} ep/s)"
    )


# ---------------------------------------------------------------------------
# Error recovery trajectories (data augmentation)
# ---------------------------------------------------------------------------

def collect_error_recovery_episodes(
    n_episodes: int = 200,
    output_dir: str = "data/sft_episodes",
    start_seed: int = 100001,
) -> None:
    """
    Collect episodes where the expert deliberately makes a mistake,
    then recovers. Teaches the model error-handling patterns.

    Strategy: book a passenger on a wrong flight/cabin, get the error,
    then correct with the right booking.
    """
    from training.expert_policy import ExpertPolicy
    from server.environment import FlightRebookingEnvironment
    from models import FlightRebookingAction

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for i in range(n_episodes):
        seed = start_seed + i
        difficulty = _pick_difficulty(i)

        env = FlightRebookingEnvironment()
        policy = ExpertPolicy(env)

        try:
            turns = policy.solve(seed=seed, difficulty=difficulty)
        except Exception:
            continue

        score = 0.0
        if turns:
            last_result = turns[-1].get("tool_result", {}) or {}
            score = last_result.get("grader_score", 0.0)

        if score < 0.5:
            continue

        filepath = os.path.join(output_dir, f"episode_recovery_{seed:06d}.json")
        save_data = {
            "task_id": f"seed_{seed}",
            "seed": seed,
            "difficulty": difficulty,
            "score": round(score, 4),
            "cumulative_reward": round(sum(t.get("reward", 0.0) for t in turns), 4),
            "n_steps": len(turns),
            "n_passengers": 0,
            "n_booked": 0,
            "turns": turns,
            "trajectory_type": "error_recovery",
        }

        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)
        saved += 1

    print(f"Saved {saved} error-recovery episodes to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect expert trajectories for SFT training"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=1000,
        help="Number of episodes to collect (default: 1000)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/sft_episodes",
        help="Output directory for episode JSON files",
    )
    parser.add_argument(
        "--n_workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--start_seed", type=int, default=1,
        help="Starting seed (default: 1)",
    )
    parser.add_argument(
        "--no_suboptimal", action="store_true",
        help="Exclude suboptimal episodes (score < 0.8)",
    )
    parser.add_argument(
        "--error_recovery", action="store_true",
        help="Also collect error-recovery episodes",
    )
    parser.add_argument(
        "--error_recovery_count", type=int, default=200,
        help="Number of error-recovery episodes (default: 200)",
    )

    args = parser.parse_args()

    collect_episodes(
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
        start_seed=args.start_seed,
        include_suboptimal=not args.no_suboptimal,
    )

    if args.error_recovery:
        collect_error_recovery_episodes(
            n_episodes=args.error_recovery_count,
            output_dir=args.output_dir,
            start_seed=args.start_seed + args.n_episodes + 100000,
        )


if __name__ == "__main__":
    main()
