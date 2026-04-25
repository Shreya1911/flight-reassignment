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
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset_dir": "training/sft_dataset",
    "dataset_repo": "Shreya1911/flight-rebooking-sft-data",
    "output_dir": "checkpoints/sft",

    # LoRA
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Training
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.05,
    "bf16": True,
    "gradient_checkpointing": True,
    "max_length": 4096,
    "logging_steps": 10,
    "save_strategy": "epoch",
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
    from datasets import load_from_disk,load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # Load dataset
    dataset_dir = config["dataset_dir"]
    print(f"Loading dataset from {dataset_dir}...")
    from huggingface_hub import login
    import os

    # Login to HF (uses HF_TOKEN env variable set in Space secrets)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load dataset from HF Hub if dataset_repo is set, else fall back to local
    if config.get("dataset_repo"):
        print(f"Loading dataset from HF Hub: {config['dataset_repo']}")
        dataset = load_dataset(config["dataset_repo"], split="train")
    else:
        print(f"Loading dataset from disk: {config['dataset_dir']}")
        dataset = load_from_disk(config["dataset_dir"])

    print(f"  Dataset: {dataset}")
    print(f"  Records: {len(dataset)}")
    print(f"  Dataset: {dataset}")

    # LoRA config
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        task_type="CAUSAL_LM",
    )

    # SFT training arguments
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],

        # SFT-specific
        max_length=config["max_length"],
        assistant_only_loss=config["assistant_only_loss"],
        packing=config["packing"],

        # Misc
        report_to="none",  # Change to "wandb" if you use W&B
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        # ADD THESE THREE LINES:
        push_to_hub=True,
        hub_model_id="Shreya1911/flight-rebooking-sft",
        hub_strategy="every_save",
    )

    # Create trainer
    print(f"\nInitializing SFTTrainer...")
    print(f"  Model: {config['model_name']}")
    print(f"  LoRA: r={config['lora_r']}, alpha={config['lora_alpha']}")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Batch size (effective): "
          f"{config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Max length: {config['max_length']}")
    print(f"  Assistant-only loss: {config['assistant_only_loss']}")

    trainer = SFTTrainer(
        model=config["model_name"],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

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
