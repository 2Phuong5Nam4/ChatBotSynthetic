max_prompt_length =1500 + 1 # + 1 just in case!
max_completion_length = 2048 - 1501

from trl.trainer.grpo_config import GRPOConfig

from training.grpo.rewards import answer_reward, format_thinking_reward
from trl.trainer.grpo_trainer import GRPOTrainer


class GrpoTrainer:
    def __init__(self):
        

        self.training_args = GRPOConfig(
            learning_rate = 5e-6,
            adam_beta1 = 0.9,
            adam_beta2 = 0.99,
            weight_decay = 0.1,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "paged_adamw_8bit",
            logging_steps = 1,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1, # Increase to 4 for smoother training
            num_generations = 6, # Decrease if out of memory
            max_prompt_length = max_prompt_length,
            max_completion_length =max_completion_length,
            # num_train_epochs = 1, # Set to 1 for a full training run
            max_steps = 250,
            save_steps = 250,
            max_grad_norm = 0.1,
            report_to = "none", # Can use Weights & Biases
            output_dir = "outputs",
        )

    def train(self, model, tokenizer, train_dataset):
        
        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = [
                answer_reward,
                format_thinking_reward,
            ],
            args = self.training_args,
            train_dataset = train_dataset,
        )
        trainer.train()