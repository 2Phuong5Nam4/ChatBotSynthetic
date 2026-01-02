#!/usr/bin/env python3
"""
Training script for SFT (Supervised Fine-Tuning) with LoRA.

This script loads configuration from YAML, sets up the model, dataset, and trainer,
then executes training with monitoring and model saving.

Usage:
    python scripts/train_sft.py --config configs/sft.yaml
    python scripts/train_sft.py --config configs/sft.yaml --override training.max_steps=100
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.sft import (
    ModelLoader,
    DatasetLoader,
    SFTTrainerWrapper,
    TrainingMonitor,
    ConfigLoader
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train language model with SFT and LoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Override config values (e.g., training.max_steps=100)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving model after training"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print configuration and exit"
    )

    return parser.parse_args()


def apply_overrides(config, overrides):
    """
    Apply command-line overrides to configuration.

    Args:
        config: Configuration dictionary
        overrides: List of override strings (e.g., ["training.max_steps=100"])

    Returns:
        Updated configuration
    """
    if not overrides:
        return config

    for override in overrides:
        if "=" not in override:
            print(f"Warning: Invalid override format: {override}")
            continue

        key_path, value = override.split("=", 1)
        keys = key_path.split(".")

        # Navigate to the nested key
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value (try to infer type)
        final_key = keys[-1]
        try:
            # Try to evaluate as Python literal
            import ast
            current[final_key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string if evaluation fails
            current[final_key] = value

        print(f"Override: {key_path} = {current[final_key]}")

    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    print("=" * 60)
    print("SFT Training with LoRA - Heineken Vietnam Chatbot")
    print("=" * 60)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = ConfigLoader.load_config(args.config)

    # Apply overrides
    if args.override:
        print("\nApplying configuration overrides:")
        config = apply_overrides(config, args.override)

    # Validate configuration
    print("\nValidating configuration...")
    ConfigLoader.validate_config(config)
    print("✓ Configuration is valid")

    # Show config and exit if requested
    if args.show_config:
        print("\nConfiguration:")
        import yaml
        print(yaml.dump(config, default_flow_style=False))
        return

    # Initialize memory tracking
    start_memory = None
    if config.get("monitoring", {}).get("show_memory_stats", True) and torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Initial GPU Memory Stats")
        print("=" * 60)
        stats = ModelLoader.show_memory_stats()
        ModelLoader.print_memory_stats(stats)
        start_memory = stats["current_memory_gb"]

    # Step 1: Load model and tokenizer
    print("\n" + "=" * 60)
    print("Step 1: Loading Model and Tokenizer")
    print("=" * 60)
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_and_configure()
    print(f"✓ Model loaded: {config['model']['name']}")
    print(f"✓ Max sequence length: {config['model']['max_seq_length']}")
    print(f"✓ Chat template: {config.get('chat_template', 'llama-3.1')}")

    # Show memory after model loading
    if config.get("monitoring", {}).get("show_memory_stats", True) and torch.cuda.is_available():
        print("\nGPU Memory after model loading:")
        stats = ModelLoader.show_memory_stats(start_memory)
        ModelLoader.print_memory_stats(stats)

    # Step 2: Load and prepare dataset
    print("\n" + "=" * 60)
    print("Step 2: Loading and Preparing Dataset")
    print("=" * 60)
    dataset_loader = DatasetLoader(config, tokenizer)
    dataset = dataset_loader.prepare_dataset()
    print(f"✓ Dataset loaded: {config['dataset']['data_path']}")
    print(f"✓ Number of examples: {len(dataset)}")

    # Step 3: Create trainer
    print("\n" + "=" * 60)
    print("Step 3: Creating Trainer")
    print("=" * 60)
    trainer_wrapper = SFTTrainerWrapper(config, model, tokenizer, dataset)
    trainer_wrapper.create_trainer()
    print("✓ SFT Trainer created")

    # Enable response-only training if configured
    if config.get("response_only_training", {}).get("enabled", False):
        trainer_wrapper.enable_response_only_training()
        print("✓ Response-only training enabled")

    # Step 4: Train
    print("\n" + "=" * 60)
    print("Step 4: Training")
    print("=" * 60)
    print(f"Output directory: {config['training']['output_dir']}")
    print(f"Max steps: {config['training'].get('max_steps', 'N/A')}")
    print(f"Num epochs: {config['training'].get('num_train_epochs', 'N/A')}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Batch size: {config['training']['per_device_train_batch_size']}")
    print(f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
    print()

    trainer_stats = trainer_wrapper.train()

    # Show training statistics
    if config.get("monitoring", {}).get("log_training_stats", True):
        TrainingMonitor.print_training_stats(trainer_stats, start_memory)

    # Step 5: Save model
    if not args.no_save:
        print("\n" + "=" * 60)
        print("Step 5: Saving Model")
        print("=" * 60)
        trainer_wrapper.save_model()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
