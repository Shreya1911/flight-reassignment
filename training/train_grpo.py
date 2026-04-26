"""
GRPO (Group Relative Policy Optimization) training script.

Phase 2 of the training pipeline: reinforces the SFT-trained model against
the live environment using the grader score as reward, teaching trade-off
reasoning that can't be captured in static demonstrations.

Usage:
    python -m training.train_grpo
    python -m training.train_grpo --config training/configs/grpo_config.yaml

Requires:
    pip install trl>=1.2.0 peft transformers datasets accelerate
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULTS = {
    "model_name": "/app/checkpoints/sft/final",
    "dataset_dir": "training/grpo_prompts",
    "output_dir": "/app/checkpoints/grpo",

    # LoRA (smaller than SFT to reduce overfitting during RL)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"],

    # Training
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-7,
    "warmup_steps": 20,
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 5,
    "save_strategy": "steps",
    "save_steps": 200,

    # GRPO-specific
    "num_generations": 4,
    "max_completion_length": 4096,
    "beta": 0.0,
    "scale_rewards": True,
    "log_completions": True,

    # Reward weights
    "grader_reward_weight": 1.0,
    "efficiency_reward_weight": 0.5,

    # vLLM (optional)
    "use_vllm": False,
}


def load_config(config_path: str | None) -> dict:
    config = dict(DEFAULTS)
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def grader_reward(environments, **kwargs):
    """Primary reward: the environment's grader score (0 to 1)."""
    return [env.grader_score for env in environments]


def efficiency_reward(environments, **kwargs):
    """Bonus for finishing with fewer steps (max 0.1 extra)."""
    rewards = []
    for env in environments:
        obs = env._obs
        if obs is None:
            rewards.append(0.0)
            continue
        step_ratio = obs.step_count / max(1, obs.max_steps)
        rewards.append(max(0.0, 0.1 * (1.0 - step_ratio)))
    return rewards


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    """Run GRPO training."""
    from datasets import load_from_disk
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback

    from training.grpo_env import FlightRebookingGRPOEnv

    # Fix cache permission error - point to writable directory
    os.environ["HF_HOME"] = "/app/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/app/hf_cache"

    # Login without saving token to disk
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    # Load prompt dataset
    dataset_dir = config["dataset_dir"]
    dataset = load_from_disk(dataset_dir)

    # LoRA config
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        task_type="CAUSAL_LM",
    )

    # GRPO training arguments
    training_args = GRPOConfig(
        output_dir=config["output_dir"],

        # GRPO-specific
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],

        # Training
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],

        # Logging
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        log_completions=config["log_completions"],

        # GRPO algorithm
        beta=config["beta"],
        scale_rewards=config["scale_rewards"],

        # Reward weights (in GRPOConfig, not GRPOTrainer)
        reward_weights=[
            config["grader_reward_weight"],
            config["efficiency_reward_weight"],
        ],

        # Misc
        report_to="none",
    )

    # Reward functions
    reward_funcs = [grader_reward, efficiency_reward]

    # Create trainer
    print(f"\nInitializing GRPOTrainer...")
    print(f"  Model: {config['model_name']}")
    print(f"  LoRA: r={config['lora_r']}, alpha={config['lora_alpha']}")
    print(f"  Num generations: {config['num_generations']}")
    print(f"  Max completion length: {config['max_completion_length']}")
    print(f"  Beta (KL penalty): {config['beta']}")
    print(f"  Reward weights: grader={config['grader_reward_weight']}, "
          f"efficiency={config['efficiency_reward_weight']}")
    print(f"  LR: {config['learning_rate']}")

    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                print(f"Step {state.global_step}: {logs}", flush=True)
                sys.stdout.flush()

    trainer = GRPOTrainer(
        model=config["model_name"],
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        peft_config=peft_config,
        environment_factory=FlightRebookingGRPOEnv,
        callbacks=[LogCallback()],
    )

    # Train
    print(f"\nStarting GRPO training...")
    trainer.train()

    # Save final checkpoint
    final_dir = os.path.join(config["output_dir"], "final")
    trainer.save_model(final_dir)
    print(f"\nSaved final model to {final_dir}")

    if hasattr(trainer, "tokenizer") and trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(final_dir)
        print(f"Saved tokenizer to {final_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GRPO training for Flight Rebooking Agent"
    )
    parser.add_argument(
        "--config", type=str, default="training/configs/grpo_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)

    args = parser.parse_args()

    config = load_config(args.config)
    for key in ["model_name", "dataset_dir", "output_dir", "num_generations",
                 "learning_rate", "lora_r", "beta"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    train(config)


if __name__ == "__main__":
    main()
