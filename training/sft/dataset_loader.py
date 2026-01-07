"""
Dataset loader module for preparing training datasets.
Handles formatting conversation data and applying chat templates.
"""

from datetime import date, datetime
from datasets import load_dataset, Dataset
from typing import Dict, Any, Callable, List, Optional


class DatasetLoader:
    """Loads and prepares datasets for fine-tuning."""

    def __init__(self, config: Dict[str, Any], tokenizer: Any):
        """
        Initialize DatasetLoader with configuration and tokenizer.

        Args:
            config: Dictionary containing dataset configuration
        """
        self.dataset_config = config.get("dataset", {})
        self.tokenizer = tokenizer

    def apply_chat_template(self, convo: List[Dict]) -> str:
        """
        Apply ChatML template to a conversation.

        Args:
            convo: List of message dictionaries with 'role' and 'content' keys.
                   Supported roles: system, user, assistant, tool

        Returns:
            Formatted conversation string in ChatML format with <think> tags
        """
        # Convert any datetime objects to strings before applying template
        def convert_datetimes(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetimes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetimes(item) for item in obj]
            else:
                return obj

        # Deep copy and convert convo to avoid mutating original
        convo_clean: List[Dict] = convert_datetimes(convo)  # type: ignore
        

        system_prompt = "Bạn là nhân viên CSKH Heineken Vietnam đang hỗ trợ trợ khách hàng theo những quy trình có sẵn."
        convo_clean.insert(0, {"role": "system", "content": system_prompt})
        formatted_text = self.tokenizer.apply_chat_template(convo_clean, tokenize=False, add_generation_prompt=False, enable_thinking=True)
        return formatted_text

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
            self.apply_chat_template(
                convo,
            )
            for convo in convos
        ]

        text_field = self.dataset_config.get("text_field", "text")
        return {text_field: texts}

    def load_dataset(self, split: str = "train") -> Dataset:
        """
        Load dataset from file.

        Args:
            split: Dataset split to load ("train" or "validation")

        Returns:
            Loaded dataset
        """
        # Support both old 'data_path' and new 'train_path'/'validation_path' configs
        train_path = self.dataset_config.get("train_path") or self.dataset_config.get("data_path")
        validation_path = self.dataset_config.get("validation_path")

        if not train_path:
            raise ValueError("train_path or data_path must be specified in dataset config")

        dataset_format = self.dataset_config.get("format", "jsonl")

        # Build data_files dictionary for train and validation splits
        data_files = {"train": train_path}
        if validation_path:
            data_files["validation"] = validation_path

        dataset = load_dataset(
            dataset_format,
            data_files=data_files,
            split=split,
        )
        
        # flatten messages in dataset
        flatten_convos = []
        for convo in dataset:
            current_turn = []
            for message in convo["messages"]:
                # Copy entire message dictionary to preserve all fields
                # (reasoning_content, tool_calls, etc.)
                current_turn.append(dict(message))

                if message["role"] == "assistant":
                    flatten_convo = dict(convo)
                    flatten_convo["messages"] = current_turn.copy()
                    flatten_convos.append(flatten_convo)
                    
        dataset = Dataset.from_list(flatten_convos)

        
        return dataset

    def prepare_dataset(self, split: str = "train") -> Dataset:
        """
        Load and prepare dataset with formatting.

        Args:
            split: Dataset split to load ("train" or "validation")

        Returns:
            Prepared dataset with formatted text
        """
        dataset = self.load_dataset(split=split)
        

        # Apply formatting function
        dataset = dataset.map(
            self.formatting_prompts_func,
            batched=True
        )

        return dataset

