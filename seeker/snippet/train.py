#date: 2022-10-10T17:14:10Z
#url: https://api.github.com/gists/d56455177fd9f504270108dbbcbe2ef9
#owner: https://api.github.com/users/chavinlo

import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    num_frames = 20,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    'gifs',                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = 1,
    train_lr = 1e-4,
    save_and_sample_every = 1000,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()
