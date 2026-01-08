"""
Dataset loader module for preparing training datasets.
Handles formatting conversation data and applying chat templates.
"""

import re
from datetime import date, datetime
import pandas as pd
from datasets import Dataset
from typing import Dict, Any, Callable, List, Optional
from transformers import TextStreamer


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

    def _strip_think_tags(self, content: str) -> str:
        """Remove <think>...</think> tags from content."""
        return re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()

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

        # Strip <think> tags from non-last assistant messages
        # The template will add <think> only for the last assistant turn
        for i, msg in enumerate(convo_clean):
            if msg["role"] == "assistant" and i < len(convo_clean) - 1:
                # Strip think tags from content
                if "content" in msg and msg["content"]:
                    msg["content"] = self._strip_think_tags(msg["content"])
                # Remove reasoning_content field for non-last turns
                if "reasoning_content" in msg:
                    del msg["reasoning_content"]

        system_prompt = "Bạn là nhân viên CSKH Heineken Vietnam đang hỗ trợ trợ khách hàng theo những quy trình có sẵn."
        convo_clean.insert(0, {"role": "system", "content": system_prompt})
        formatted_text = self.tokenizer.apply_chat_template(
            convo_clean, tokenize=False, add_generation_prompt=False, enable_thinking=True)
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
        return {text_field: texts, "messages": convos}

    def load_dataset(self, split: str = "train") -> List[Dict]:
        """
        Load dataset from file.

        Args:
            split: Dataset split to load ("train" or "validation")

        Returns:
            List of conversation dicts (not Dataset, to avoid PyArrow type issues)
        """
        # Support both old 'data_path' and new 'train_path'/'validation_path' configs
        train_path = self.dataset_config.get(
            "train_path") or self.dataset_config.get("data_path")
        validation_path = self.dataset_config.get("validation_path")

        if not train_path:
            raise ValueError(
                "train_path or data_path must be specified in dataset config")

        # Use pandas to load JSONL - it handles mixed types better than datasets
        file_path = train_path if split == "train" else (
            validation_path or train_path)
        df = pd.read_json(file_path, lines=True)
        records = df.to_dict(orient="records")

        # flatten messages in dataset
        flatten_convos = []
        for convo in records:
            current_turn = []
            for message in convo["messages"]:
                # Copy entire message dictionary to preserve all fields
                # (reasoning_content, tool_calls, etc.)
                current_turn.append(dict(message))

                if message["role"] == "assistant":
                    flatten_convo = dict(convo)
                    flatten_convo["messages"] = current_turn.copy()
                    flatten_convos.append(flatten_convo)

        return flatten_convos

    def prepare_dataset(self, split: str = "train") -> Dataset:
        """
        Load and prepare dataset with formatting.

        Args:
            split: Dataset split to load ("train" or "validation")

        Returns:
            Prepared dataset with formatted text
        """
        convos = self.load_dataset(split=split)

        # Apply chat template to each conversation
        message_field = self.dataset_config.get("message_field", "messages")
        text_field = self.dataset_config.get("text_field", "text")

        formatted_data = []
        for convo in convos:
            text = self.apply_chat_template(convo[message_field])
            formatted_data.append({text_field: text})

        # Now create Dataset with simple text-only structure (no mixed types)
        dataset = Dataset.from_list(formatted_data)

        return dataset
