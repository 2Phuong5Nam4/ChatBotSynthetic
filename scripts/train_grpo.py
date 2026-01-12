from training.sft.config_loader import ConfigLoader
from training.sft.dataset_loader import DatasetLoader
from training.grpo.trainer import GrpoTrainer
from datasets import concatenate_datasets
from pathlib import Path
config_path = Path('__file__').parent / "configs" / "sft.yaml"
config = ConfigLoader.load_config(str(config_path))
max_prompt_length =1500 + 1 # + 1 just in case!
max_completion_length = 2048 - 1501

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "2Phuong5Nam4/Heineken_qwen-3-8B_chatbot",  # LoRA đã SFT
            max_seq_length = 1024*2,
            load_in_4bit = True,
            fast_inference = True,
            # CRITICAL cho T4
            gpu_memory_utilization = 0.8,
            max_num_seqs = 2,
            max_lora_rank = 32,
        )


tokenizer.chat_template = config["chat_template"]
loader = DatasetLoader(config, tokenizer)
train_dataset = loader.prepare_dataset(split="train")
dev_dataset = loader.prepare_dataset(split="validation")
train_dataset = concatenate_datasets([train_dataset, dev_dataset])
trainer = GrpoTrainer()
trainer.train(model, tokenizer, train_dataset)