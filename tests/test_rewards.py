
import pytest
from training.grpo.rewards.format_think import format_thinking_reward
from training.grpo.rewards.answer import answer_reward
from training.sft.dataset_loader import DatasetLoader

from transformers import AutoTokenizer
@pytest.fixture(scope="session")
def tokenizer():
    """
    Load tokenizer once for all tests.
    Uses TinyLlama tokenizer - small and fast to load.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Qwen3-8B", use_fast=True)

    # Ensure chat template exists
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role']}}: {{message['content']}}\n{% endfor %}"

    return tokenizer


@pytest.fixture
def dataset_config():
    """Dataset configuration loaded from YAML config file."""
    from pathlib import Path
    from training.sft.config_loader import ConfigLoader
    config_path = Path(__file__).parent.parent / "configs" / "sft.yaml"
    config = ConfigLoader.load_config(str(config_path))
    return config

@pytest.fixture
def dataset_loader(dataset_config, tokenizer):
    """Fixture to create a DatasetLoader instance."""
    loader = DatasetLoader(dataset_config, tokenizer)
    return loader

def test_format_thinking_reward(dataset_loader: DatasetLoader  ):
    train = dataset_loader.prepare_dataset("train")
    train = train.map(lambda batch: {
    "thinking_reward": format_thinking_reward(
        prompts= batch["prompt"],
        completions=[[{"role": "assistant", "content": ans}] for ans in batch["answer"]],
        answer= batch["answer"],
        tokenizer=dataset_loader.tokenizer
    )
    }, batched=True, batch_size=8)
    # all rewards of dataset must == 1.0
    
    def is_all_one(row):
        if not abs(row["thinking_reward"] - 1.0) < 1e-6:
            return False
                # raise ValueError(f"Found reward != 1.0: {row['answer']} -> {row['thinking_reward']}")
        return True
    
    train = train.map(lambda row: {
        "check_all_one": is_all_one(row),
    }, batched=False)
    count_invalid = train.filter(lambda row: not row["check_all_one"])
    # print rows with thinking_reward != 1.0
    # for row in count_invalid:
    #     if row["thinking_reward"] != 1.0:
    #         print("==="*50)
    #         print(f"Thinking Reward: {row['thinking_reward']}")
    #         print(f"Answer: {row['answer']}")
    #         print("==="*50)
    assert len(count_invalid) == 0, f"Found {len(count_invalid)} rows with thinking_reward != 1.0"
    val = dataset_loader.prepare_dataset("validation")
    val = val.map(lambda batch: {
    "thinking_reward": format_thinking_reward(
        prompts= batch["prompt"],
        completions=[[{"role": "assistant", "content": ans}] for ans in batch["answer"]],
        answer= batch["answer"]
    )
    }, batched=True, batch_size=8)
    # all rewards of dataset must == 1.0    
    val = val.map(lambda row: {
        "check_all_one": is_all_one(row),
    }, batched=False)
    
    count_invalid_val = val.filter(lambda row: not row["check_all_one"])
    # print rows with thinking_reward != 1.0
    # for row in count_invalid_val:
    #     if row["thinking_reward"] != 1.0:
    #         print("==="*50)
    #         print(f"Thinking Reward: {row['thinking_reward']}")
    #         print(f"Answer: {row['answer']}")
    #         print("==="*50)   
    assert len(count_invalid_val) == 0, f"Found {len(count_invalid_val)} rows with thinking_reward != 1.0"
    
        
        
# def test_answer_reward(dataset_loader: DatasetLoader  ):
    
#     train = dataset_loader.prepare_dataset("train")
#     train = train.map(lambda batch: {
#     "answer_reward": answer_reward(
#         prompts = batch["prompt"],
#         completions = [[{"role": "assistant", "content": ans}] for ans in batch["answer"]],
#         answer= batch["answer"]
#     )
#     }, batched=True, batch_size=8)
#     # all rewards of dataset must == 1.0
#     def is_all_one(row):
#         if row["answer_reward"] != 1.0:
#                 raise ValueError(f"Found reward != 1.0: {row['answer']} -> {row['answer_reward']}")
#         return True 
#     train.map(lambda row: {
#         "check_all_one": is_all_one(row),
#     }, batched=False)
    