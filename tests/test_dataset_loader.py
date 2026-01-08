"""
Tests for DatasetLoader class.
Only loads tokenizer (not full model) for efficiency.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

# Load dataset_loader module directly without triggering __init__.py imports
# This avoids loading unsloth which requires GPU
module_path = Path(__file__).parent.parent / "training" / "sft" / "dataset_loader.py"
spec = importlib.util.spec_from_file_location("dataset_loader", module_path)
dataset_loader_module = importlib.util.module_from_spec(spec)
sys.modules["dataset_loader"] = dataset_loader_module
spec.loader.exec_module(dataset_loader_module)

DatasetLoader = dataset_loader_module.DatasetLoader

from transformers import DataCollator
@pytest.fixture(scope="session")
def tokenizer():
    """
    Load tokenizer once for all tests.
    Uses TinyLlama tokenizer - small and fast to load.
    """
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-8B", use_fast=True)

    # Ensure chat template exists
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role']}}: {{message['content']}}\n{% endfor %}"

    return tokenizer


@pytest.fixture
def dataset_config():
    """Dataset configuration using actual data files."""
    return {
        "dataset": {
            "train_path": "data/train.jsonl",
            "validation_path": "data/validation.jsonl",
            "format": "json",  # Note: use "json" not "jsonl" for HF datasets
            "message_field": "messages",
            "text_field": "text"
        }
    }


class TestDatasetLoader:
    """Test suite for DatasetLoader class."""

    def test_initialization(self, dataset_config, tokenizer):
        """Test DatasetLoader initialization."""
        loader = DatasetLoader(dataset_config, tokenizer)

        assert loader.dataset_config == dataset_config["dataset"]
        assert loader.tokenizer == tokenizer

    def test_prepare_dataset(self, dataset_config, tokenizer):
        """Test preparing dataset with formatting function."""
        loader = DatasetLoader(dataset_config, tokenizer)
        dataset = loader.prepare_dataset(split="train")

        assert isinstance(dataset, Dataset)
        assert len(dataset) > 243

        # Check that text field is formatted
        first_item = dataset[0]
        assert "text" in first_item
        # check first 10 text field
        rows = [dataset[i] for i in range(3)]
        print("@@first 3 formatted texts:")
        for row in rows:
            print("==="*50)
            print(row["text"])
            print("---"*50)
            print(row["prompt"])
            print(row["answer"])
        print("@@"*50)
        print(dataset[0])
        
    def test_prepare_validation_dataset(self, dataset_config, tokenizer):
        """Test preparing validation dataset with formatting function."""
        loader = DatasetLoader(dataset_config, tokenizer)
        dataset = loader.prepare_dataset(split="validation")

        assert isinstance(dataset, Dataset)
        assert len(dataset) > 32

        # Check that text field is formatted
        first_item = dataset[0]
        assert "text" in first_item
        # check first 10 text field
        rows = [dataset[i] for i in range(3)]
        print("@@first 10 formatted texts in validation:")
        for row in rows:
            print("==="*50)
            print(row["text"])
            print("---"*50)
            print(row["prompt"])
            print(row["answer"])
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
