"""
Model loader module for loading and configuring language models with LoRA.
Supports Unsloth FastLanguageModel for efficient training.
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from typing import Optional, Dict, Any, Tuple, List


class ModelLoader:
    """Loads and configures language models for fine-tuning."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelLoader with configuration.

        Args:
            config: Dictionary containing model and LoRA configuration
        """
        self.model_config = config.get("model", {})
        self.lora_config = config.get("lora", {})
        self.chat_template = config.get("chat_template", "llama-3.1")

    def load_model(self) -> Tuple[Any, Any]:
        """
        Load the base model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.get("name", "unsloth/Llama-3.2-3B-Instruct"),
            max_seq_length=self.model_config.get("max_seq_length", 2048),
            dtype=self.model_config.get("dtype"),
            load_in_4bit=self.model_config.get("load_in_4bit", True),
            token=self.model_config.get("token"),
        )

        return model, tokenizer

    def configure_lora(self, model: Any) -> Any:
        """
        Configure LoRA for the model.

        Args:
            model: The base model to apply LoRA to

        Returns:
            Model with LoRA configured
        """
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_config.get("r", 16),
            target_modules=self.lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_alpha=self.lora_config.get("lora_alpha", 16),
            lora_dropout=self.lora_config.get("lora_dropout", 0),
            bias=self.lora_config.get("bias", "none"),
            use_gradient_checkpointing=self.lora_config.get("use_gradient_checkpointing", "unsloth"),
            random_state=self.lora_config.get("random_state", 3407),
            use_rslora=self.lora_config.get("use_rslora", False),
            loftq_config=self.lora_config.get("loftq_config"),
        )

        return model

    def setup_chat_template(self, tokenizer: Any) -> Any:
        """
        Setup chat template for the tokenizer.

        Args:
            tokenizer: The tokenizer to configure

        Returns:
            Tokenizer with chat template configured
        """
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=self.chat_template,
        )

        return tokenizer

    def load_and_configure(self) -> Tuple[Any, Any]:
        """
        Load model, configure LoRA, and setup chat template.

        Returns:
            Tuple of (configured_model, configured_tokenizer)
        """
        # Load base model and tokenizer
        model, tokenizer = self.load_model()

        # Configure LoRA
        model = self.configure_lora(model)

        # Setup chat template
        tokenizer = self.setup_chat_template(tokenizer)

        return model, tokenizer

    @staticmethod
    def get_max_seq_length(config: Dict[str, Any]) -> int:
        """
        Get max sequence length from config.

        Args:
            config: Configuration dictionary

        Returns:
            Max sequence length
        """
        return config.get("model", {}).get("max_seq_length", 2048)

    @staticmethod
    def show_memory_stats(start_memory: Optional[float] = None) -> Dict[str, float]:
        """
        Show GPU memory statistics.

        Args:
            start_memory: Starting memory to calculate delta

        Returns:
            Dictionary with memory stats
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        gpu_stats = torch.cuda.get_device_properties(0)
        current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        stats = {
            "gpu_name": gpu_stats.name,
            "max_memory_gb": max_memory,
            "current_memory_gb": current_memory,
            "current_percentage": round(current_memory / max_memory * 100, 3),
        }

        if start_memory is not None:
            memory_used_for_training = round(current_memory - start_memory, 3)
            stats["memory_used_for_training_gb"] = memory_used_for_training
            stats["training_percentage"] = round(memory_used_for_training / max_memory * 100, 3)

        return stats

    @staticmethod
    def print_memory_stats(stats: Dict[str, float]):
        """Print memory statistics in a formatted way."""
        if "error" in stats:
            print(f"Error: {stats['error']}")
            return

        print(f"GPU = {stats['gpu_name']}. Max memory = {stats['max_memory_gb']} GB.")
        print(f"{stats['current_memory_gb']} GB of memory reserved ({stats['current_percentage']}%).")

        if "memory_used_for_training_gb" in stats:
            print(f"Memory used for training = {stats['memory_used_for_training_gb']} GB ({stats['training_percentage']}%).")
