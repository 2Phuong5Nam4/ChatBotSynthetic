import os
import re
import json
from pathlib import Path
from typing import Any, List, Optional
from collections import Counter
from functools import partial
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import unsloth
from datasets import concatenate_datasets
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL

from training.sft.config_loader import ConfigLoader
from training.sft.dataset_loader import DatasetLoader
from training.grpo.rewards import answer_reward, format_thinking_reward

# ============================================================================
# Environment Configuration
# ============================================================================
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
# os.environ["PYTORCH_HIP_ALLOC_CONF"] = ""
# os.environ["PYTORCH_ALLOC_CONF"] = ""
# os.environ["VLLM_USE_CUDA_GRAPH"] = "0"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # Dùng Xformers thay vì FlashInfer
# os.environ["VLLM_USE_FLASHINFER"] = "0"

# ============================================================================
# Constants
# ============================================================================
MAX_PROMPT_LENGTH = 1501  # 1500 + 1 just in case!
MAX_SEQ_LENGTH = 2048


# ============================================================================
# Main Training Function
# ============================================================================
def run():
    # Load config
    config_path = Path('__file__').parent / "configs" / "sft.yaml"
    config = ConfigLoader.load_config(str(config_path))
    
    # Patch FastRL
    PatchFastRL("GRPO", FastLanguageModel)
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="2Phuong5Nam4/Heineken_qwen-3-8B_chatbot",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        full_finetuning=False,
        gpu_memory_utilization=0.6,
        max_lora_rank=32,
        max_num_seqs=2
    )
    
    # Setup tokenizer and dataset
    tokenizer.chat_template = config["chat_template"]
    loader = DatasetLoader(config, tokenizer)
    train_dataset = loader.prepare_dataset(split="train")
    dev_dataset = loader.prepare_dataset(split="validation")
    train_dataset = concatenate_datasets([train_dataset, dev_dataset])
    def filter_by_length(example):
        # Format thử ra chuỗi text cuối cùng
        # Lưu ý: tokenize=False để chỉ lấy string, kiểm tra nhanh hơn
        full_text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=False
        )

        # Đếm token
        # return True nếu độ dài <= 1845
        return len(tokenizer(full_text)["input_ids"]) <= MAX_PROMPT_LENGTH

    # 3. Áp dụng Filter (Dataset gốc vẫn nguyên vẹn, tạo ra dataset mới đã lọc)
    train_dataset = train_dataset.filter(
        filter_by_length,
        # num_proc=4 # Bật lên nếu dataset lớn (s>100k dòng) để chạy nhanh hơn
    )
    # Training configuration
    training_args = GRPOConfig(
        gradient_checkpointing=True,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=5,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
        # max_steps=300,
        num_train_epochs=2,
        save_steps=400,
        report_to="none",
        output_dir="outputs",
    )
    
    patched_format_thinking_reward = partial(
        format_thinking_reward,
        tokenizer=tokenizer,  # tokenizer đã load
    )
    patched_format_thinking_reward.__name__ = "format_thinking_reward"
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[answer_reward, patched_format_thinking_reward],
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train and save
    trainer.train()
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")


if __name__ == "__main__":
    run()