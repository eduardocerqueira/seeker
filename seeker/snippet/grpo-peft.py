#date: 2025-02-11T16:48:54Z
#url: https://api.github.com/gists/daedf11ceb899d1f65c17b03e2cb9a01
#owner: https://api.github.com/users/aniket-ak

# Run this using this command
# accelerate launch --multi_gpu rl_grpo_peft.py

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. First, set up environment variables
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 2. Import necessary libraries
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

# 3. Initialize distributed environment
local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if local_rank != -1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    set_seed(42)


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir="outputs/Qwen-1.5B-GRPO"
    run_name="Qwen-1.5B-GRPO-gsm8k"

# 4. Setup FSDP and Accelerator
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# 5. Now create training arguments
training_args = GRPOConfig(
    output_dir="output",
    run_name="grpo-peft",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    # gradient_checkpointing=True,
    ddp_backend="nccl",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    local_rank=local_rank,  # Add this line
    torch_compile=False,
)

# 6. Create and prepare model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank if local_rank != -1 else "cuda"},
)

# Enable gradient checkpointing BEFORE creating trainer or wrapping with DDP
# model.gradient_checkpointing_enable()

tokenizer = "**********"
tokenizer.pad_token = "**********"

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    # lora_train_bias=False,
)

# Create the trainer
trainer = GRPOTrainer(
    model=model,  # Use unwrapped model here
    processing_class= "**********"
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

# Now wrap the trainer's model with DDP
if local_rank != -1:
    trainer.model = DDP(
        trainer.model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    # Add this to make gradient checkpointing accessible through DDP
    # trainer.model.module.gradient_checkpointing_enable = trainer.model.module.gradient_checkpointing_enable

# Add cleanup in try/finally
try:
    trainer.train()
finally:
    if local_rank != -1:
        dist.destroy_process_group()
    dist.destroy_process_group()
