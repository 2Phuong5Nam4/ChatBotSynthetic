"""
Trainer module for fine-tuning language models with SFT.
Includes training configuration, response-only training, and model saving utilities.
"""

from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq, TrainerCallback
from unsloth.chat_templates import train_on_responses_only
from unsloth import FastLanguageModel
from typing import Dict, Any, Optional
import torch


class SFTTrainerWrapper:
    """Wrapper for SFT (Supervised Fine-Tuning) training."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Any,
        tokenizer: Any,
        dataset: Any
    ):
        """
        Initialize SFT Trainer.

        Args:
            config: Configuration dictionary
            model: The model to train
            tokenizer: The tokenizer
            dataset: Prepared training dataset
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.trainer = None
        self.training_config = config.get("training", {})
        self.response_only_config = config.get("response_only_training", {})

    def create_trainer(self) -> SFTTrainer:
        """
        Create SFT trainer with configuration.

        Returns:
            Configured SFTTrainer
        """
        # Get dataset text field
        dataset_text_field = self.config.get("dataset", {}).get("text_field", "text")

        # Get max sequence length
        max_seq_length = self.config.get("model", {}).get("max_seq_length", 2048)

        # Create SFT config
        sft_config = SFTConfig(
            per_device_train_batch_size=self.training_config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 4),
            warmup_steps=self.training_config.get("warmup_steps", 5),
            max_steps=self.training_config.get("max_steps", 30),
            num_train_epochs=self.training_config.get("num_train_epochs"),
            learning_rate=self.training_config.get("learning_rate", 2e-4),
            logging_steps=self.training_config.get("logging_steps", 1),
            optim=self.training_config.get("optim", "adamw_8bit"),
            weight_decay=self.training_config.get("weight_decay", 0.001),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "linear"),
            seed=self.training_config.get("seed", 3407),
            output_dir=self.training_config.get("output_dir", "outputs"),
            report_to=self.training_config.get("report_to", "none"),
            save_steps=self.training_config.get("save_steps", 10),
            save_total_limit=self.training_config.get("save_total_limit", 3),
        )

        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            packing=self.training_config.get("packing", False),
            args=sft_config,
        )

        self.trainer = trainer
        return trainer

    def enable_response_only_training(self) -> SFTTrainer:
        """
        Enable response-only training if configured.

        Returns:
            Trainer with response-only training enabled
        """
        if not self.trainer:
            raise ValueError("Trainer not created. Call create_trainer() first.")

        if not self.response_only_config.get("enabled", False):
            return self.trainer

        instruction_part = self.response_only_config.get(
            "instruction_part",
            "<|start_header_id|>user<|end_header_id|>\n\n"
        )
        response_part = self.response_only_config.get(
            "response_part",
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        self.trainer = train_on_responses_only(
            self.trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )

        return self.trainer

    def train(self):
        """Execute training."""
        if not self.trainer:
            raise ValueError("Trainer not created. Call create_trainer() first.")

        print("Starting training...")
        trainer_stats = self.trainer.train()
        print("Training completed!")

        return trainer_stats

    def setup_and_train(self):
        """Setup trainer and run training in one call."""
        self.create_trainer()
        self.enable_response_only_training()
        return self.train()

    def save_model(self, save_config: Optional[Dict[str, Any]] = None):
        """
        Save the trained model.

        Args:
            save_config: Optional save configuration. If None, uses config from self.config
        """
        if save_config is None:
            save_config = self.config.get("save", {})

        method = save_config.get("method", "merged_16bit")
        save_path = save_config.get("save_path", "outputs/final_model")

        print(f"Saving model using method: {method}")

        if method == "lora":
            # Save only LoRA adapters
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        elif method == "merged_16bit":
            # Save merged model in 16bit
            self.model.save_pretrained_merged(
                save_path,
                self.tokenizer,
                save_method="merged_16bit"
            )
        elif method == "merged_4bit":
            # Save merged model in 4bit
            self.model.save_pretrained_merged(
                save_path,
                self.tokenizer,
                save_method="merged_4bit"
            )
        elif method == "gguf":
            # Save in GGUF format
            quantization_method = save_config.get("gguf_quantization", "q4_k_m")
            self.model.save_pretrained_gguf(
                save_path,
                self.tokenizer,
                quantization_method=quantization_method
            )
        else:
            raise ValueError(f"Unknown save method: {method}")

        print(f"Model saved to: {save_path}")

        # Push to hub if configured
        if save_config.get("push_to_hub", False):
            self._push_to_hub(save_config)

    def _push_to_hub(self, save_config: Dict[str, Any]):
        """Push model to HuggingFace Hub."""
        hub_model_id = save_config.get("hub_model_id")
        if not hub_model_id:
            raise ValueError("hub_model_id must be specified to push to hub")

        hub_token = save_config.get("hub_token")

        print(f"Pushing model to hub: {hub_model_id}")
        self.model.push_to_hub(hub_model_id, token=hub_token)
        self.tokenizer.push_to_hub(hub_model_id, token=hub_token)
        print("Model pushed to hub successfully!")


class TrainingMonitor:
    """Utility class for monitoring training."""

    @staticmethod
    def print_training_stats(trainer_stats: Any, start_memory: Optional[float] = None):
        """
        Print training statistics.

        Args:
            trainer_stats: Training statistics from trainer.train()
            start_memory: Starting GPU memory for calculating usage
        """
        metrics = trainer_stats.metrics

        print("\n" + "=" * 50)
        print("Training Statistics")
        print("=" * 50)

        # Time stats
        train_runtime = metrics.get('train_runtime', 0)
        print(f"Training time: {train_runtime:.2f} seconds ({train_runtime/60:.2f} minutes)")

        # Loss stats
        if 'train_loss' in metrics:
            print(f"Final training loss: {metrics['train_loss']:.4f}")

        # Samples per second
        if 'train_samples_per_second' in metrics:
            print(f"Samples/second: {metrics['train_samples_per_second']:.2f}")

        # Steps per second
        if 'train_steps_per_second' in metrics:
            print(f"Steps/second: {metrics['train_steps_per_second']:.2f}")

        # Memory stats
        if start_memory is not None and torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            used_memory_for_training = round(current_memory - start_memory, 3)
            used_percentage = round(current_memory / max_memory * 100, 3)
            training_percentage = round(used_memory_for_training / max_memory * 100, 3)

            print(f"\nPeak reserved memory: {current_memory} GB ({used_percentage}%)")
            print(f"Memory used for training: {used_memory_for_training} GB ({training_percentage}%)")

        print("=" * 50 + "\n")
