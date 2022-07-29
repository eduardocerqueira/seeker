#date: 2022-07-29T17:04:35Z
#url: https://api.github.com/gists/9b6df3a7b3e77a50475c6fe4bdadc0b7
#owner: https://api.github.com/users/miccio-dk

# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re

import numpy as np
import torch
import hydra
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from tqdm import tqdm

import PIL
from accelerate import Accelerator
from timm import create_model
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a ResNet50 on the Oxford-IIT Pet Dataset
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


# Function to get the label from the filename
def extract_label(fname):
    stem = fname.split(os.path.sep)[-1]
    return re.search(r"^(.*)_\d+\.jpg$", stem).groups()[0]


class PetsDataset(Dataset):
    def __init__(self, file_names, image_transform=None, label_to_id=None):
        self.file_names = file_names
        self.image_transform = image_transform
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        raw_image = PIL.Image.open(fname)
        image = raw_image.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        label = extract_label(fname)
        if self.label_to_id is not None:
            label = self.label_to_id[label]
        return {"image": image, "label": label}


def training_function(configs):
    # Initialize accelerator
    if configs.with_tracking:
        accelerator = Accelerator(
            cpu=configs.cpu, mixed_precision=configs.mixed_precision, log_with="all", logging_dir=configs.logging_dir
        )
    else:
        accelerator = Accelerator(cpu=configs.cpu, mixed_precision=configs.mixed_precision)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = configs.lr
    num_epochs = int(configs.num_epochs)
    seed = int(configs.seed)
    batch_size = int(configs.batch_size)
    image_size = configs.image_size
    if not isinstance(image_size, (list, tuple)):
        image_size = (image_size, image_size)

    # Parse out whether we are saving every epoch or after a certain number of batches
    if hasattr(configs.checkpointing_steps, "isdigit"):
        if configs.checkpointing_steps == "epoch":
            checkpointing_steps = configs.checkpointing_steps
        elif configs.checkpointing_steps.isdigit():
            checkpointing_steps = int(configs.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{configs.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configsuration
    if configs.with_tracking:
        if accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            if configs.logging_dir:
                run = os.path.join(configs.logging_dir, run)
            accelerator.init_trackers(run, OmegaConf.to_container(configs, resolve=True))

    # Grab all the image filenames
    file_names = [os.path.join(configs.data_dir, fname) for fname in os.listdir(configs.data_dir) if fname.endswith(".jpg")]

    # Build the label correspondences
    all_labels = [extract_label(fname) for fname in file_names]
    id_to_label = list(set(all_labels))
    id_to_label.sort()
    label_to_id = {lbl: i for i, lbl in enumerate(id_to_label)}

    # Set the seed before splitting the data.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Split our filenames between train and validation
    random_perm = np.random.permutation(len(file_names))
    cut = int(0.8 * len(file_names))
    train_split = random_perm[:cut]
    eval_split = random_perm[cut:]

    # For training we use a simple RandomResizedCrop
    train_tfm = Compose([RandomResizedCrop(image_size, scale=(0.5, 1.0)), ToTensor()])
    train_dataset = PetsDataset(
        [file_names[i] for i in train_split], image_transform=train_tfm, label_to_id=label_to_id
    )

    # For evaluation, we use a deterministic Resize
    eval_tfm = Compose([Resize(image_size), ToTensor()])
    eval_dataset = PetsDataset([file_names[i] for i in eval_split], image_transform=eval_tfm, label_to_id=label_to_id)

    # Instantiate dataloaders.
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=4)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Freezing the base model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    # We normalize the batches of images to be a bit faster.
    mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None].to(accelerator.device)
    std = torch.tensor(model.default_cfg["std"])[None, :, None, None].to(accelerator.device)

    # Instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr / 25)

    # Instantiate learning rate scheduler
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_dataloader))

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if configs.resume_from_checkpoint:
        if configs.resume_from_checkpoint is not None or configs.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {configs.resume_from_checkpoint}")
            accelerator.load_state(configs.resume_from_checkpoint)
            path = os.path.basename(configs.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # Now we train the model
    for epoch in range(starting_epoch, num_epochs):
        model.train()
        if configs.with_tracking:
            total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # We need to skip steps until we reach the resumed step
            if configs.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    overall_step += 1
                    continue
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            inputs = (batch["image"] - mean) / std
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
            # We keep track of the loss at each epoch
            if configs.with_tracking:
                total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            overall_step += 1
            if isinstance(checkpointing_steps, int):
                output_dir = f"step_{overall_step}"
                if overall_step % checkpointing_steps == 0:
                    if configs.output_dir is not None:
                        output_dir = os.path.join(configs.output_dir, output_dir)
                    accelerator.save_state(output_dir)
        model.eval()
        accurate = 0
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            inputs = (batch["image"] - mean) / std
            with torch.no_grad():
                outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["label"]))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader) - samples_seen]
                    references = references[: len(eval_dataloader) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            else:
                samples_seen += references.shape[0]
            accurate_preds = predictions == references
            accurate += accurate_preds.long().sum()

        eval_metric = accurate.item() / samples_seen
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")
        if configs.with_tracking:
            accelerator.log(
                {
                    "accuracy": 100 * eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                },
                step=overall_step,
            )
        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if configs.output_dir is not None:
                output_dir = os.path.join(configs.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if configs.with_tracking:
        accelerator.end_training()


@hydra.main(version_base=None, config_path=".", config_name="configs.yaml")
def main(configs):
    
    training_function(configs)


if __name__ == "__main__":
    main()