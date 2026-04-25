"""
Evaluation script for trained Flight Rebooking models.

Runs the model against the environment across difficulty tiers and
procedural seeds, reporting grader scores, coverage, cost efficiency,
and hard constraint violation rates.

Usage:
    # Evaluate against static tiers
    python -m training.eval --model checkpoints/grpo/final

    # Evaluate against procedural seeds
    python -m training.eval --model checkpoints/grpo/final --procedural --n_episodes 50

    # Compare base vs SFT vs GRPO
    python -m training.eval --model Qwen/Qwen2.5-7B-Instruct --compare checkpoints/sft/final checkpoints/grpo/final
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from server.environment import FlightRebookingEnvironment
from models import FlightRebookingAction


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    seed: int = 0
    difficulty: float = 0.0
    grader_score: float = 0.0
    coverage: float = 0.0
    cost_efficiency: float = 0.0
    steps_used: int = 0
    max_steps: int = 0
    total_cost: float = 0.0
    hard_violations: int = 0
    passengers_total: int = 0
    passengers_booked: int = 0
    terminal_breakdown: Optional[dict] = None


@dataclass
class TierReport:
    tier: str = ""
    n_episodes: int = 0
    mean_score: float = 0.0
    pass_rate: float = 0.0
    mean_coverage: float = 0.0
    mean_cost_efficiency: float = 0.0
    mean_steps_ratio: float = 0.0
    mean_hard_violations: float = 0.0
    scores: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model-based inference (for HF models)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str):
    """Load a HF model + tokenizer for evaluation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        print("ERROR: transformers/peft not installed.")
        sys.exit(1)

    # Check if it's a LoRA adapter
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        with open(adapter_config) as f:
            config = json.load(f)
        base_model = config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
        print(f"  Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype="auto", device_map="auto"
        )
        print(f"  Loading adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        print(f"  Loading full model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    return model, tokenizer


def generate_action(model, tokenizer, messages: List[dict]) -> Optional[dict]:
    """Generate a single action from the model given conversation messages."""
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse JSON action
    from inference import parse_llm_response
    return parse_llm_response(response)


# ---------------------------------------------------------------------------
# Expert-policy evaluation (no model needed)
# ---------------------------------------------------------------------------

def eval_expert(
    seed: int,
    difficulty: float,
) -> EpisodeMetrics:
    """Run one episode with the expert policy and return metrics."""
    from training.expert_policy import run_expert_episode

    turns, cumulative_reward, grader_score = run_expert_episode(
        seed=seed, difficulty=difficulty
    )

    # Extract terminal breakdown
    terminal_breakdown = None
    hard_violations = 0
    coverage = 0.0
    cost_efficiency = 0.0
    total_cost = 0.0
    passengers_total = 0
    passengers_booked = 0

    if turns:
        last_result = turns[-1].get("tool_result", {}) or {}
        if "terminal_breakdown" in last_result:
            terminal_breakdown = last_result["terminal_breakdown"]
            hard_violations = terminal_breakdown.get("hard_violations", 0)
            coverage = terminal_breakdown.get("coverage_score", 0.0)
            cost_efficiency = terminal_breakdown.get("cost_efficiency_score", 0.0)

    return EpisodeMetrics(
        seed=seed,
        difficulty=difficulty,
        grader_score=grader_score,
        coverage=coverage,
        cost_efficiency=cost_efficiency,
        steps_used=len(turns),
        hard_violations=hard_violations,
        terminal_breakdown=terminal_breakdown,
    )


# ---------------------------------------------------------------------------
# Model-based evaluation
# ---------------------------------------------------------------------------

def eval_model_episode(
    model,
    tokenizer,
    seed: int,
    difficulty: float,
) -> EpisodeMetrics:
    """Run one episode with a trained model and return metrics."""
    from inference import (
        SYSTEM_PROMPT, format_main_task, format_state,
        format_result, format_instruction,
    )

    env = FlightRebookingEnvironment()
    obs = env.reset(seed=seed, task_id=f"seed_{seed}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Initial user message
    user_content = "\n\n".join([
        format_main_task("procedural"),
        format_state(obs),
        format_instruction(),
    ])
    messages.append({"role": "user", "content": user_content})

    max_steps = obs.max_steps
    for step in range(max_steps):
        if obs.done:
            break

        action_dict = generate_action(model, tokenizer, messages)
        if action_dict is None:
            action_dict = {"tool_name": "list_passengers", "args": {}}

        # Execute
        action = FlightRebookingAction(
            tool_name=action_dict["tool_name"],
            args=action_dict["args"],
        )
        obs = env.step(action)

        # Update conversation
        messages.append({
            "role": "assistant",
            "content": json.dumps(action_dict),
        })

        # Build next user message
        result_parts = []
        if obs.tool_result is not None:
            result_parts.append(
                f"Last tool result: {json.dumps(obs.tool_result, indent=2)}"
            )
        result_parts.append(
            f"Reward: {obs.reward:.2f} ({obs.reward_reason})"
        )
        result_parts.append(format_state(obs))
        result_parts.append(format_instruction())

        messages.append({
            "role": "user",
            "content": "\n\n".join(result_parts),
        })

        if obs.done:
            break

    # Finalize if not done
    if not obs.done:
        obs = env.step(FlightRebookingAction(
            tool_name="finalize_plan", args={}
        ))

    # Extract metrics
    grader_score = 0.0
    terminal_breakdown = None
    if obs.tool_result and "grader_score" in obs.tool_result:
        grader_score = obs.tool_result["grader_score"]
    if obs.tool_result and "terminal_breakdown" in obs.tool_result:
        terminal_breakdown = obs.tool_result["terminal_breakdown"]

    return EpisodeMetrics(
        seed=seed,
        difficulty=difficulty,
        grader_score=grader_score,
        coverage=terminal_breakdown.get("coverage_score", 0.0) if terminal_breakdown else 0.0,
        cost_efficiency=terminal_breakdown.get("cost_efficiency_score", 0.0) if terminal_breakdown else 0.0,
        steps_used=obs.step_count,
        max_steps=max_steps,
        total_cost=obs.total_cost,
        hard_violations=terminal_breakdown.get("hard_violations", 0) if terminal_breakdown else 0,
        passengers_total=obs.passengers_total,
        passengers_booked=obs.passengers_booked,
        terminal_breakdown=terminal_breakdown,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_tier_report(tier: str, metrics_list: List[EpisodeMetrics]) -> TierReport:
    """Aggregate episode metrics into a tier report."""
    if not metrics_list:
        return TierReport(tier=tier)

    n = len(metrics_list)
    scores = [m.grader_score for m in metrics_list]

    return TierReport(
        tier=tier,
        n_episodes=n,
        mean_score=sum(scores) / n,
        pass_rate=sum(1 for s in scores if s >= 0.5) / n,
        mean_coverage=sum(m.coverage for m in metrics_list) / n,
        mean_cost_efficiency=sum(m.cost_efficiency for m in metrics_list) / n,
        mean_steps_ratio=sum(
            m.steps_used / max(1, m.max_steps) for m in metrics_list
        ) / n,
        mean_hard_violations=sum(m.hard_violations for m in metrics_list) / n,
        scores=scores,
    )


def print_report(reports: List[TierReport], model_name: str) -> None:
    """Print a formatted evaluation report."""
    print(f"\n{'='*70}")
    print(f"  Evaluation Report: {model_name}")
    print(f"{'='*70}")

    header = (
        f"{'Tier':<12} {'N':>4} {'Score':>7} {'Pass%':>7} "
        f"{'Cover':>7} {'Cost$':>7} {'Steps%':>7} {'HardV':>7}"
    )
    print(header)
    print("-" * 70)

    for r in reports:
        row = (
            f"{r.tier:<12} {r.n_episodes:>4} "
            f"{r.mean_score:>7.3f} {r.pass_rate*100:>6.1f}% "
            f"{r.mean_coverage:>7.3f} {r.mean_cost_efficiency:>7.3f} "
            f"{r.mean_steps_ratio*100:>6.1f}% {r.mean_hard_violations:>7.2f}"
        )
        print(row)

    # Overall
    all_scores = [s for r in reports for s in r.scores]
    if all_scores:
        overall_mean = sum(all_scores) / len(all_scores)
        overall_pass = sum(1 for s in all_scores if s >= 0.5) / len(all_scores)
        print("-" * 70)
        print(
            f"{'OVERALL':<12} {len(all_scores):>4} "
            f"{overall_mean:>7.3f} {overall_pass*100:>6.1f}%"
        )
    print()


def save_report(
    reports: List[TierReport],
    model_name: str,
    output_path: str,
) -> None:
    """Save evaluation results to JSON."""
    result = {
        "model": model_name,
        "tiers": {},
    }
    for r in reports:
        result["tiers"][r.tier] = {
            "n_episodes": r.n_episodes,
            "mean_score": round(r.mean_score, 4),
            "pass_rate": round(r.pass_rate, 4),
            "mean_coverage": round(r.mean_coverage, 4),
            "mean_cost_efficiency": round(r.mean_cost_efficiency, 4),
            "mean_steps_ratio": round(r.mean_steps_ratio, 4),
            "mean_hard_violations": round(r.mean_hard_violations, 4),
            "scores": [round(s, 4) for s in r.scores],
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Report saved to {output_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    n_episodes: int = 50,
    use_expert: bool = False,
    procedural: bool = False,
    output_path: str = "results/eval_report.json",
) -> None:
    """Run full evaluation."""

    if not use_expert:
        print(f"Loading model from {model_path}...")
        model, tokenizer = load_model_and_tokenizer(model_path)
    else:
        model, tokenizer = None, None
        print("Using expert policy (no model)")

    # Define evaluation tiers
    if procedural:
        tiers = [
            ("easy_proc", 0.2, list(range(10001, 10001 + n_episodes))),
            ("medium_proc", 0.5, list(range(20001, 20001 + n_episodes))),
            ("hard_proc", 0.8, list(range(30001, 30001 + n_episodes))),
        ]
    else:
        tiers = [
            ("easy", 0.2, list(range(1, 1 + n_episodes))),
            ("medium", 0.5, list(range(1001, 1001 + n_episodes))),
            ("hard", 0.8, list(range(2001, 2001 + n_episodes))),
        ]

    reports = []
    for tier_name, difficulty, seeds in tiers:
        print(f"\nEvaluating {tier_name} (difficulty={difficulty}, "
              f"n={len(seeds)})...")
        metrics_list = []

        for i, seed in enumerate(seeds):
            try:
                if use_expert:
                    m = eval_expert(seed=seed, difficulty=difficulty)
                else:
                    m = eval_model_episode(
                        model, tokenizer,
                        seed=seed, difficulty=difficulty,
                    )
                metrics_list.append(m)

                if (i + 1) % 10 == 0:
                    avg = sum(mm.grader_score for mm in metrics_list) / len(metrics_list)
                    print(f"  [{i+1}/{len(seeds)}] avg_score={avg:.3f}")

            except Exception as e:
                print(f"  Seed {seed} failed: {e}")
                continue

        report = compute_tier_report(tier_name, metrics_list)
        reports.append(report)

    # Print and save
    print_report(reports, model_path if not use_expert else "expert_policy")
    save_report(reports, model_path if not use_expert else "expert_policy", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Flight Rebooking model"
    )
    parser.add_argument(
        "--model", type=str, default="checkpoints/grpo/final",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--expert", action="store_true",
        help="Evaluate the expert policy (no model needed)",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=50,
        help="Episodes per tier (default: 50)",
    )
    parser.add_argument(
        "--procedural", action="store_true",
        help="Use procedural seeds instead of static tiers",
    )
    parser.add_argument(
        "--output", type=str, default="results/eval_report.json",
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--compare", nargs="+", type=str, default=None,
        help="Additional model paths to compare against",
    )

    args = parser.parse_args()

    # Primary evaluation
    evaluate(
        model_path=args.model,
        n_episodes=args.n_episodes,
        use_expert=args.expert,
        procedural=args.procedural,
        output_path=args.output,
    )

    # Comparison runs
    if args.compare:
        for i, compare_path in enumerate(args.compare):
            compare_output = args.output.replace(
                ".json", f"_compare_{i+1}.json"
            )
            print(f"\n{'#'*70}")
            print(f"  Comparing: {compare_path}")
            print(f"{'#'*70}")
            evaluate(
                model_path=compare_path,
                n_episodes=args.n_episodes,
                procedural=args.procedural,
                output_path=compare_output,
            )


if __name__ == "__main__":
    main()
