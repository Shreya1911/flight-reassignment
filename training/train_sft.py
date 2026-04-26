"""
SFT (Supervised Fine-Tuning) training script for the Flight Rebooking Agent.

Phase 1 of the training pipeline: teaches the model the tool-calling format,
basic booking strategy, and constraint awareness using expert trajectories.

Usage:
    python -m training.train_sft
    python -m training.train_sft --config training/configs/sft_config.yaml

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
    "model_name": "Qwen/Qwen2.5-3B",
    "dataset_dir": "training/sft_dataset",
    "dataset_repo": "Shreya1911/flight-rebooking-sft-data_v2",
    "output_dir": "/app/checkpoints/sft",

    # LoRA — reduced capacity to prevent memorization
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.10,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", "down_proj",
    ],

    # Training — 1 epoch only, lower LR, regularization
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.10,
    "weight_decay": 0.05,
    "bf16": True,
    "gradient_checkpointing": True,
    "max_length": 8192,
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 100,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "assistant_only_loss": True,
    "packing": False,
}


def load_config(config_path: str | None) -> dict:
    """Load config from YAML file, falling back to defaults."""
    config = dict(DEFAULTS)
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    """Run SFT training."""
    from datasets import load_from_disk, load_dataset, DatasetDict
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # Fix cache permission error - point to writable directory
    os.environ["HF_HOME"] = "/app/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/app/hf_cache"

    # Login without saving token to disk
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    # Load dataset — try HF Hub first, then local disk
    train_dataset = None
    eval_dataset = None

    if config.get("dataset_repo"):
        print(f"Loading dataset from HF Hub: {config['dataset_repo']}")
        loaded = load_dataset(config["dataset_repo"])
        if isinstance(loaded, DatasetDict):
            train_dataset = loaded.get("train", loaded[list(loaded.keys())[0]])
            eval_dataset = loaded.get("eval") or loaded.get("test")
        else:
            train_dataset = loaded
    else:
        dataset_dir = config["dataset_dir"]
        print(f"Loading dataset from disk: {dataset_dir}")
        loaded = load_from_disk(dataset_dir)
        if isinstance(loaded, DatasetDict):
            train_dataset = loaded.get("train", loaded[list(loaded.keys())[0]])
            eval_dataset = loaded.get("eval") or loaded.get("test")
        else:
            train_dataset = loaded

    print(f"  Train records: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"  Eval records: {len(eval_dataset)}")
    else:
        print(f"  Eval: none (no eval split found)")

    # LoRA config
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        task_type="CAUSAL_LM",
    )

    # Build SFT training arguments
    sft_kwargs = dict(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config.get("weight_decay", 0.0),
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],

        # SFT-specific
        max_length=config["max_length"],
        assistant_only_loss=config["assistant_only_loss"],
        packing=config["packing"],

        # Misc
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        push_to_hub=True,
        hub_model_id="Shreya1911/flight-rebooking-sft",
        hub_strategy="every_save",
    )

    # Add save_steps if strategy is "steps"
    if config.get("save_strategy") == "steps" and config.get("save_steps"):
        sft_kwargs["save_steps"] = config["save_steps"]

    # Add eval config if we have an eval split
    if eval_dataset is not None:
        sft_kwargs["eval_strategy"] = config.get("eval_strategy", "steps")
        eval_steps = config.get("eval_steps", 100)
        sft_kwargs["eval_steps"] = eval_steps
        sft_kwargs["per_device_eval_batch_size"] = config.get(
            "per_device_eval_batch_size",
            config["per_device_train_batch_size"],
        )

    training_args = SFTConfig(**sft_kwargs)

    # Create trainer
    print(f"\nInitializing SFTTrainer...")
    print(f"  Model: {config['model_name']}")
    print(f"  LoRA: r={config['lora_r']}, alpha={config['lora_alpha']}, "
          f"dropout={config['lora_dropout']}")
    print(f"  Targets: {config['lora_target_modules']}")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Batch size (effective): "
          f"{config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  LR: {config['learning_rate']}, weight_decay: {config.get('weight_decay', 0)}")
    print(f"  Max length: {config['max_length']}")
    print(f"  Assistant-only loss: {config['assistant_only_loss']}")

    from transformers import TrainerCallback

    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                print(f"Step {state.global_step}: {logs}", flush=True)
                sys.stdout.flush()

    trainer_kwargs = dict(
        model=config["model_name"],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        callbacks=[LogCallback()],
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = SFTTrainer(**trainer_kwargs)

    # Train
    print(f"\nStarting training...")
    trainer.train()

    # Save final checkpoint
    final_dir = os.path.join(config["output_dir"], "final")
    trainer.save_model(final_dir)
    print(f"\nSaved final model to {final_dir}")

    # Save tokenizer too for easy loading
    if hasattr(trainer, "tokenizer") and trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(final_dir)
        print(f"Saved tokenizer to {final_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT training for Flight Rebooking Agent")
    parser.add_argument(
        "--config", type=str, default="training/configs/sft_config.yaml",
        help="Path to YAML config file",
    )
    # Allow CLI overrides for common params
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--dataset_repo", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)

    args = parser.parse_args()

    # Load config, then apply CLI overrides
    config = load_config(args.config)
    for key in ["model_name", "dataset_dir", "output_dir", "num_train_epochs",
                 "learning_rate", "max_length", "lora_r","dataset_repo"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    train(config)


if __name__ == "__main__":
    main()
