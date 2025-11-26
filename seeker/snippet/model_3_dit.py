#date: 2025-11-26T16:40:53Z
#url: https://api.github.com/gists/6cff268207f1a97c339553922ca03d37
#owner: https://api.github.com/users/arturshamsiev314

import os
import time

import numpy as np
import safetensors.torch
import torch.nn as nn
import torch.utils.data
from PIL import Image
from torchvision.datasets import EMNIST
from torchvision.transforms import v2 as T

torch.manual_seed(78)

SIZE = 24
transforms = T.Compose([
    T.RandomRotation(degrees=(-90, -90)),
    T.RandomHorizontalFlip(p=1),
    T.Resize(SIZE),
    T.ToTensor(),
    T.Normalize((0.175,), (0.35,)),
])

dataset = EMNIST(
    root="./emnst",
    split="balanced",
    download=True,
    transform=transforms,
)
# subset = Subset(dataset, torch.randperm(4096 * 2))
subset = dataset
num_classes = len(subset.classes)
classes_map = {c: i for (i, c) in enumerate(subset.classes)}
print("Classes:", num_classes)


class DenoiserBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio, condition_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        self.modulator_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 4),
            nn.SiLU(),
            nn.Linear(condition_dim * 4, hidden_dim * 6),
        )
        nn.init.zeros_(self.modulator_mlp[-1].weight)
        nn.init.zeros_(self.modulator_mlp[-1].bias)

    def forward(self, x, c):
        mod = self.modulator_mlp(c)  # (B, hidden_dim * 6)
        mod = mod.unsqueeze(1)  # (B, 1, hidden_dim * 6)
        scale_1, shift_1, gate_1, scale_2, shift_2, gate_2 = mod.chunk(6, dim=2)
        
        z = self.ln1(x)
        z = z * (1 - scale_1) + shift_1
        z, _ = self.attn(z, z, z)
        z = z * gate_1
        x_after_attn = x + z

        z = self.ln2(x_after_attn)
        z = z * (1 - scale_2) + shift_2
        z = self.ffn(z)
        z = z * gate_2
        return x_after_attn + z


class Denoiser(nn.Module):
    def __init__(self, hidden_dims, num_heads, num_blocks, condition_dim):
        super().__init__()

        self.input_patcher = nn.Unfold(kernel_size=2, stride=2)
        self.input_projector = nn.Linear(4, hidden_dims)
        self.class_embeddings = nn.Embedding(num_classes, condition_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 4),
            nn.SiLU(),
            nn.Linear(condition_dim * 4, condition_dim),
        )
        block_list = [DenoiserBlock(hidden_dims, num_heads, 4, condition_dim) for _ in range(num_blocks)]
        self.blocks = nn.ModuleList(block_list)
        self.output_decoder = nn.Sequential(
            nn.Fold(output_size=(SIZE, SIZE), kernel_size=2, stride=2),
            nn.Conv2d(in_channels=hidden_dims//4, out_channels=1, kernel_size=3, padding=1),
        )

        self.pos_embeddings = nn.Parameter(data=torch.randn(144, hidden_dims))
        self.pos_embeddings_scale = nn.Parameter(torch.zeros(1))
        self.time_linear = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.LayerNorm(condition_dim)
        )

    def forward(self, x, t, c):
        input_patches = self.input_patcher(x)
        input_seq = input_patches.permute(0, 2, 1)
        hidden = self.input_projector(input_seq)
        hidden = hidden + self.pos_embeddings * self.pos_embeddings_scale

        time_embedding = self.time_linear(t)
        class_embedding = self.class_embeddings(c)
        class_condition = self.class_mlp(class_embedding)
        condition = time_embedding + class_condition
        for block in self.blocks:
            hidden = block(hidden, condition)

        output_seq = hidden.permute(0, 2, 1)
        return self.output_decoder(output_seq)


log_img_path = f"./final_dit_{int(time.time())}_log_2.png"
classes = torch.round(torch.rand([5]) * num_classes - 1).to(torch.int32)


def inference(noise: torch.Tensor, model, class_idx, num_steps=50) -> torch.Tensor:
    for s in range(num_steps, 0, -1):
        time = torch.tensor(s / num_steps, device=noise.device)  # (1, 1)
        time = time.expand(noise.size(0), 1)
        pred = model(noise, time, class_idx).detach()
        noise = noise + pred * (1 / num_steps)
    return noise


def log_sample(noise, d_model):
    classes = torch.tensor([1, 2, 13, 14, 15], device=noise.device)
    sample = inference(noise, d_model, classes, num_steps=40).detach()
    sample = torch.cat(torch.unbind(sample.squeeze(1)), dim=1)
    sample = (sample - sample.min()) / (sample.max() - sample.min()).clamp_min(0)
    output = (sample * 255).cpu().numpy().astype(np.uint8)

    if os.path.isfile(log_img_path):
        log_img = Image.open(log_img_path).convert("L")
        log_img = np.array(log_img)
        output = np.concatenate([log_img, output])

    Image.fromarray(output).save(log_img_path)


BATCH_SIZE = 128
LR = 6e-4
DEVICE = 'cuda'
EPOCHS = 10

data_loader = torch.utils.data.DataLoader(
    dataset=subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
model = Denoiser(hidden_dims=16*8, num_heads=8, num_blocks=8, condition_dim=32)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

num_samples = 5
sample_noise = torch.randn(5, 1, SIZE, SIZE, device=DEVICE)

if __name__ == "__main__":
    print(f"Update sample log every {(EPOCHS - 1) // num_samples} epoch")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for x, clazz in data_loader:
            x = x.to(DEVICE)

            clazz = clazz.to(DEVICE)

            time = torch.rand((x.size(0), 1), device=DEVICE)

            noise = torch.randn_like(x, device=DEVICE)

            true_velocity = x - noise

            # Помните, чем больше time, тем больше шума
            ext_time = time.reshape(time.size(0), 1, 1, 1)
            # xt = x * (1 - ext_time) + noise * ext_time
            # Можно было написать вот так:
            xt = noise + true_velocity * (1 - ext_time)

            pred_velocity = model(xt, time, clazz)  # передаём в модель сэмплы, time и класс

            loss = torch.mean((true_velocity - pred_velocity) ** 2)  # ошибка между ожидаемым и предсказанным значением
            epoch_loss += loss.item()  # накапливаем ошибку для логирования

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 2 == 0:
            with torch.no_grad():
                log_sample(sample_noise, model)
        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1} completed.")
            print(f"Loss: {epoch_loss / len(subset) * 1000:.2f}")

    safetensors.torch.save_model(model, "./emnist_dit.sft")
