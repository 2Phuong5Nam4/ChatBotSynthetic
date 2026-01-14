from unsloth import FastLanguageModel
import torch

# 1. Load model gốc và adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="grpo_chat_model",  # Thư mục chứa adapter của bạn
    max_seq_length=2048,
    # dtype=None,  # Auto detect
    load_in_16bit=True,
    # dtype=torch.float16,  # hoặc torch.bfloat16
    # load_in_4bit=False,  # KHÔNG load 4-bit
)
# model = FastLanguageModel.get_peft_model(model)


# 3. Save merged model
output_dir = "./grpo_merged_model"
model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")

print(f"✅ Model đã được merge và lưu tại: {output_dir}")

# vllm serve grpo_merged_model \
#     --gpu-memory-utilization 0.8 \
#     --max-model-len 2048 