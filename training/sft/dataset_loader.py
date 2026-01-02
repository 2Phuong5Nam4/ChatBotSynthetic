"""
Dataset loader module for preparing training datasets.
Handles formatting conversation data and applying chat templates.
"""

from datasets import load_dataset, Dataset
from typing import Dict, Any, Callable, Optional


class DatasetLoader:
    """Loads and prepares datasets for fine-tuning."""

    def __init__(self, config: Dict[str, Any], tokenizer: Any):
        """
        Initialize DatasetLoader with configuration and tokenizer.

        Args:
            config: Dictionary containing dataset configuration
            tokenizer: Tokenizer for formatting prompts
        """
        self.dataset_config = config.get("dataset", {})
        self.tokenizer = tokenizer

    def formatting_prompts_func(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format conversation examples using chat template.

        Args:
            examples: Batch of examples containing messages

        Returns:
            Dictionary with formatted text
        """
        message_field = self.dataset_config.get("message_field", "messages")
        convos = examples[message_field]

        texts = [
            self.tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            for convo in convos
        ]

        text_field = self.dataset_config.get("text_field", "text")
        return {text_field: texts}

    def load_dataset(self) -> Dataset:
        """
        Load dataset from file.

        Returns:
            Loaded dataset
        """
        data_path = self.dataset_config.get("data_path")
        if not data_path:
            raise ValueError("data_path must be specified in dataset config")

        dataset_format = self.dataset_config.get("format", "json")
        split = self.dataset_config.get("split", "train")

        dataset = load_dataset(
            dataset_format,
            data_files=data_path,
            split=split
        )

        return dataset

    def prepare_dataset(self) -> Dataset:
        """
        Load and prepare dataset with formatting.

        Returns:
            Prepared dataset with formatted text
        """
        dataset = self.load_dataset()

        # Apply formatting function
        dataset = dataset.map(
            self.formatting_prompts_func,
            batched=True
        )

        return dataset

    @staticmethod
    def create_custom_formatting_func(
        tokenizer: Any,
        message_field: str = "messages",
        text_field: str = "text",
        add_generation_prompt: bool = False
    ) -> Callable:
        """
        Create a custom formatting function.

        Args:
            tokenizer: Tokenizer to use
            message_field: Field name containing messages
            text_field: Output field name for formatted text
            add_generation_prompt: Whether to add generation prompt

        Returns:
            Formatting function
        """
        def formatting_func(examples: Dict[str, Any]) -> Dict[str, Any]:
            convos = examples[message_field]
            texts = [
                tokenizer.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt
                )
                for convo in convos
            ]
            return {text_field: texts}

        return formatting_func

    def get_text_field(self) -> str:
        """Get the text field name from config."""
        return self.dataset_config.get("text_field", "text")
