#date: 2023-07-05T16:54:47Z
#url: https://api.github.com/gists/66b46667601c3ed803ddf06695778ab1
#owner: https://api.github.com/users/samching

import torch
from datasets import load_dataset
import argparse
import os
import math
from itertools import chain
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import (DummyOptim, DummyScheduler,
                              InitProcessGroupKwargs, set_seed)
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, set_seed, default_data_collator)

class CFG:
    BATCH_SIZE: int = 1
    GRADIENT_ACCUMULATE_EVERY: int = 4
    RESUME_FROM_CHECKPOINT: str = None 
    CHECKPOINTING_STEPS: int = 1000
    OUTPUT_DIR: str = "/mnt/nvme/home/save_model"
    ENTITY_NAME: str = ""

def main():

    set_seed(42)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=CFG.GRADIENT_ACCUMULATE_EVERY,
        mixed_precision="bf16",
        log_with="wandb", 
        kwargs_handlers=[timeout]
    )

    accelerator.init_trackers(
            project_name="falcon_big_law",
            init_kwargs={"wandb": {"entity": CFG.ENTITY_NAME}},
        )

    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b",
        use_cache=False,
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()

    accelerator.print(f"Training a {model.num_parameters():,} parameter model")

    device = accelerator.device
    model.to(device)


    # Dataloaders

    with accelerator.main_process_first():
        train_dataset = load_dataset('conceptofmind/biglaw-falcon-8k', split = 'train')

    train_loader = DataLoader(
        train_dataset, 
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=CFG.BATCH_SIZE
    )


    # Dummy Optimizer for DeepSpeed

    optim = DummyOptim(
        model.parameters(), 
        lr=3e-5
    )


    # Determine number of training steps

    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps: {max_train_steps}")


    # Dummy Scheduler for DeepSpeed

    scheduler = DummyScheduler(
        optim, 
        total_num_steps=max_train_steps, 
        warmup_num_steps=int(max_train_steps * 0.01)
    )


    # prepare

    model, optim, train_loader, scheduler = accelerator.prepare(
        model, optim, train_loader, scheduler
    )


    # checkpoint scheduler

    accelerator.register_for_checkpointing(scheduler)


    # Recalculate 

    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps recalculated: {max_train_steps}")

    # Total batch size for logging

    total_batch_size = (
        CFG.BATCH_SIZE * accelerator.num_processes * CFG.GRADIENT_ACCUMULATE_EVERY
    )
    accelerator.print(f"Total batch size: {total_batch_size}")

    # resume training

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if CFG.RESUME_FROM_CHECKPOINT:
        if CFG.RESUME_FROM_CHECKPOINT is not None or CFG.RESUME_FROM_CHECKPOINT != "":
            accelerator.print(f"Resuming from checkpoint {CFG.RESUME_FROM_CHECKPOINT}")
            accelerator.load_state(CFG.RESUME_FROM_CHECKPOINT)
            path = os.path.basename(CFG.RESUME_FROM_CHECKPOINT)
        training_difference = os.path.splitext(path)[0]

        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = (
            int(training_difference.replace("step_", ""))
            * CFG.GRADIENT_ACCUMULATE_EVERY
        ) 

    if CFG.RESUME_FROM_CHECKPOINT and resume_step is not None:
        # We need to skip steps until we reach the resumed step
        train_loader = accelerator.skip_first_batches(train_loader, resume_step)
        total_steps += resume_step
        progress_bar.update(resume_step)
        accelerator.print(f"Resuming training from step {resume_step}")

    # training

    model.train()
    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            inputs = batch["input_ids"]
            labels = batch["input_ids"]
            loss = model(inputs, labels=labels).loss
            accelerator.backward(loss)

            accelerator.log({"loss": loss.item()}, step=step)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()
            scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if isinstance(CFG.CHECKPOINTING_STEPS, int):
            if completed_steps % CFG.CHECKPOINTING_STEPS == 0:
                output_dir = f"step_{completed_steps}"
                if CFG.OUTPUT_DIR is not None:
                    output_dir = os.path.join(CFG.OUTPUT_DIR, output_dir)
                accelerator.save_state(output_dir)

        if completed_steps >= max_train_steps:
            break

    # end training

    accelerator.print(f"Training Finished")
    accelerator.end_training()

    # save final model

    accelerator.print(f"Saving model to {CFG.OUTPUT_DIR}")
    if CFG.OUTPUT_DIR is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        with accelerator.main_process_first():
            accelerator.save(
                unwrapped_model.state_dict(), f"{CFG.OUTPUT_DIR}/final/final_model.pt"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    main()