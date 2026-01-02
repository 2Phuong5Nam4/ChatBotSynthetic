# Training Module

Refactored training pipeline for the Heineken Vietnam Chatbot with modular, reusable components.

## Structure

```
training/
├── __init__.py                 # Package initialization
└── sft/                        # Supervised Fine-Tuning module
    ├── __init__.py            # SFT module exports
    ├── config_loader.py       # YAML configuration loader with path resolution
    ├── dataset_loader.py      # Dataset loading and formatting
    ├── model_loader.py        # Model and LoRA configuration
    ├── trainer.py             # SFT trainer wrapper and monitoring
    └── infor.py              # Legacy info utilities

configs/
└── sft.yaml                   # SFT training configuration

scripts/
└── train_sft.py              # Main training script
```

## Quick Start

### 1. Configure Training

Edit `configs/sft.yaml` to customize your training:

```yaml
model:
  name: "unsloth/Llama-3.2-3B-Instruct"
  max_seq_length: 2048

dataset:
  data_path: "data/thinking_teacher_conversations.json"

training:
  max_steps: 30
  learning_rate: 2.0e-4
  output_dir: "checkpoints"
```

### 2. Run Training

```bash
# Basic training
python scripts/train_sft.py --config configs/sft.yaml

# Override config values
python scripts/train_sft.py --config configs/sft.yaml --override training.max_steps=100

# Show configuration without training
python scripts/train_sft.py --config configs/sft.yaml --show-config

# Train without saving model
python scripts/train_sft.py --config configs/sft.yaml --no-save
```

## Components

### ConfigLoader (`config_loader.py`)

Loads YAML configuration and resolves relative paths.

```python
from training.sft import ConfigLoader

config = ConfigLoader.load_config("configs/sft.yaml")
ConfigLoader.validate_config(config)
```

**Features:**
- Automatic path resolution (relative → absolute)
- Configuration validation
- Config merging for overrides
- YAML save/load utilities

### ModelLoader (`model_loader.py`)

Handles model loading and LoRA configuration.

```python
from training.sft import ModelLoader

model_loader = ModelLoader(config)
model, tokenizer = model_loader.load_and_configure()
```

**Features:**
- FastLanguageModel integration
- LoRA/QLoRA configuration
- Chat template setup
- GPU memory monitoring

### DatasetLoader (`dataset_loader.py`)

Prepares datasets with chat template formatting.

```python
from training.sft import DatasetLoader

dataset_loader = DatasetLoader(config, tokenizer)
dataset = dataset_loader.prepare_dataset()
```

**Features:**
- Automatic chat template application
- Batch processing
- Custom formatting functions
- Multiple dataset formats support

### SFTTrainerWrapper (`trainer.py`)

Wraps TRL's SFTTrainer with enhanced features.

```python
from training.sft import SFTTrainerWrapper

trainer = SFTTrainerWrapper(config, model, tokenizer, dataset)
trainer_stats = trainer.setup_and_train()
trainer.save_model()
```

**Features:**
- Response-only training
- Multiple save formats (LoRA, merged 16bit/4bit, GGUF)
- HuggingFace Hub integration
- Training statistics monitoring

### TrainingMonitor (`trainer.py`)

Utilities for monitoring training progress.

```python
from training.sft import TrainingMonitor

TrainingMonitor.print_training_stats(trainer_stats, start_memory)
```

## Configuration Options

### Model Configuration

```yaml
model:
  name: "unsloth/Llama-3.2-3B-Instruct"
  max_seq_length: 2048
  dtype: null  # null, float16, bfloat16
  load_in_4bit: true
  token: null  # HuggingFace token
```

### LoRA Configuration

```yaml
lora:
  r: 16  # Rank
  lora_alpha: 16
  lora_dropout: 0
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  use_gradient_checkpointing: "unsloth"
```

### Dataset Configuration

```yaml
dataset:
  data_path: "data/thinking_teacher_conversations.json"
  format: "json"
  split: "train"
  text_field: "text"
  message_field: "messages"
```

### Training Configuration

```yaml
training:
  output_dir: "checkpoints"
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_steps: 30
  learning_rate: 2.0e-4
  optim: "adamw_8bit"
  report_to: "none"  # wandb, tensorboard, none
```

### Saving Configuration

```yaml
save:
  method: "merged_16bit"  # lora, merged_16bit, merged_4bit, gguf
  save_path: "checkpoints/final_model"
  push_to_hub: false
  hub_model_id: "your-username/model-name"
```

## Advanced Usage

### Custom Training Loop

```python
from training.sft import ModelLoader, DatasetLoader, SFTTrainerWrapper, ConfigLoader

# Load config
config = ConfigLoader.load_config("configs/sft.yaml")

# Setup components
model_loader = ModelLoader(config)
model, tokenizer = model_loader.load_and_configure()

dataset_loader = DatasetLoader(config, tokenizer)
dataset = dataset_loader.prepare_dataset()

# Create and configure trainer
trainer = SFTTrainerWrapper(config, model, tokenizer, dataset)
trainer.create_trainer()
trainer.enable_response_only_training()

# Train
stats = trainer.train()

# Save
trainer.save_model()
```

### Using Different Save Methods

```python
# Save only LoRA adapters
trainer.save_model({"method": "lora", "save_path": "checkpoints/lora_only"})

# Save merged 4bit model
trainer.save_model({"method": "merged_4bit", "save_path": "checkpoints/merged_4bit"})

# Save as GGUF
trainer.save_model({
    "method": "gguf",
    "save_path": "checkpoints/model.gguf",
    "gguf_quantization": "q4_k_m"
})
```

### Memory Monitoring

```python
from training.sft import ModelLoader

# Get initial memory
stats = ModelLoader.show_memory_stats()
start_memory = stats["current_memory_gb"]

# ... training ...

# Get memory after training
stats = ModelLoader.show_memory_stats(start_memory)
ModelLoader.print_memory_stats(stats)
```

## Extending

### Adding New Training Methods

Create new modules under `training/`:

```
training/
├── sft/          # Supervised Fine-Tuning
├── dpo/          # Direct Preference Optimization (future)
└── rlhf/         # RLHF (future)
```

### Custom Components

All components accept configuration dictionaries and can be extended:

```python
class CustomDatasetLoader(DatasetLoader):
    def formatting_prompts_func(self, examples):
        # Custom formatting logic
        pass
```

## Troubleshooting

### Path Issues

All paths in config are automatically resolved relative to project root. Use relative paths:
- ✅ `data/training_data.json`
- ❌ `/home/user/project/data/training_data.json`

### Memory Issues

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use `load_in_4bit: true`
- Enable gradient checkpointing

### Import Errors

Ensure you're running from project root:
```bash
cd /home/namnp/ChatBotSynthetic
python scripts/train_sft.py --config configs/sft.yaml
```
