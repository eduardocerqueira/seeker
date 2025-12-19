#date: 2025-12-19T16:53:04Z
#url: https://api.github.com/gists/6e674f6bb9cb423edece8742c47aa4fd
#owner: https://api.github.com/users/johannes-spies

# train a diffusion model with score matching for a 2D checkerboard dataset in 10mins on a laptop cpu
# $ pip install jax flax numpy matplotlib optax wandb
# $ wget https://gist.githubusercontent.com/johannes-spies/e94f60351bd1673eb7f5b735e46ec191/raw/9929c9284d138a6ef9c092546318e1dc59941a3c/dataset.csv
# $ python -m diffusion

from typing import Callable, Sequence
from itertools import batched
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, jax.random as jr
from flax import nnx
import optax
from dsm import Dsm
import wandb

hypers = {
  "seed": 0,

  # diffusion hyperparameters
  "diffusion_steps": 100,
  "beta_start": 1e-4,
  "beta_end": 5e-3,

  # model hyperparameters
  "hidden_dims": [512] * 4,
  "time_embedding_dim": 16,

  # training hyperparameters
  "num_epochs": 500,
  "batch_size": 128,
  "num_eval_samples": 10000,
  "learning_rate": 3e-4,
  "eval_interval": 10,
}
run = wandb.init(project="diffusion-checkerboard", config=hypers)

rngs = nnx.Rngs(hypers["seed"])
print(f"{jax.devices()=}")

# load data
dataset = np.loadtxt("dataset.csv", delimiter=",", skiprows=1)
np.random.seed(0)
np.random.shuffle(dataset)
print("shuffled dataset")

# split into train and test
thres = int(len(dataset) * 0.95)
train, test = dataset[:thres], dataset[thres:]
print(f"split dataset in {train.shape=}, {test.shape=}")

# set up a diffusion trainer with a linear schedule
dsm = Dsm.linear(hypers["diffusion_steps"], hypers["beta_start"], hypers["beta_end"])
print(f"{dsm.alpha_bar[-1]=:.04f}")

# define an MLP to learn the mapping
class MlpWithTime(nnx.Module):
  def __init__(
    self,
    sample_dim: int,
    hidden_dims: Sequence[int],
    max_time: int,
    time_embedding_dim: int,
    activation_fn: Callable[[jax.Array], jax.Array] = nnx.silu,
    *,
    rngs: nnx.Rngs,
  ):
    layers = []
    dims = [sample_dim + time_embedding_dim] + list(hidden_dims) + [sample_dim]
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
      layers.append(nnx.Linear(in_dim, out_dim, rngs=rngs))
      layers.append(activation_fn)
    self.model = nnx.Sequential(*layers[:-1])
    self.time_embedding = nnx.Embed(max_time, time_embedding_dim, rngs=rngs)

  def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
    # attach the time information to
    time_embedding = self.time_embedding(t)
    x = jnp.concatenate([x, time_embedding], axis=-1)
    return self.model(x)

# create a model and optimizer
noise_model = MlpWithTime(
  2,
  hypers["hidden_dims"],
  dsm.diffusion_steps,
  hypers["time_embedding_dim"],
  rngs=rngs,
)
tx = optax.adamw(hypers["learning_rate"])
optimizer = nnx.Optimizer(noise_model, tx, wrt=nnx.Param)
num_params = sum([p.size for _, p in nnx.flatten(nnx.split(noise_model)[1])[1]])
print(f"created model with {num_params} parameters")

# train model
train_step = nnx.jit(dsm.train_step)
batch_loss = nnx.jit(dsm.batch_loss)
for epoch in (pbar := trange(hypers["num_epochs"])):
  log = {}

  # run updates on training set
  train_loss = 0.
  num_batches = 0
  for batch in batched(tqdm(train, desc="train", leave=False), hypers["batch_size"]):
    batch = jnp.stack(batch)
    loss = train_step(noise_model, optimizer, batch, rngs.train()).item()
    train_loss += loss 
    num_batches += 1
  train_loss /= num_batches
  log["train/loss"] = train_loss

  # compute validation error on test set
  test_loss = 0.
  num_batches = 0
  for batch in batched(tqdm(train, desc="validation", leave=False), hypers["batch_size"]):
    # create validation data
    batch = jnp.stack(batch)
    rng_noise, rng_time = jr.split(rngs.train())
    batch_noises = jr.normal(rng_noise, shape=batch.shape)
    batch_times = jr.randint(rng_time, (len(batch),), 0, dsm.diffusion_steps)
    batch_noised_samples = jax.vmap(dsm.get_noised_sample)(batch, batch_times, batch_noises)

    # compute loss for validation set
    loss = batch_loss(noise_model, batch_noised_samples, batch_times, batch_noises).item()
    test_loss += loss
    num_batches += 1
  test_loss /= num_batches
  log["test/loss"] = test_loss

  if epoch % hypers["eval_interval"] == 0 or epoch == hypers["num_epochs"] - 1 or epoch == 0:
    # generate samples using the learned model
    sample_rngs = rngs.fork(split=hypers["num_eval_samples"])
    batch_sample_fn = nnx.vmap(
      lambda model, rngs: dsm.sample(model, sample_shape=(2,), rngs=rngs),
      in_axes=(None, 0),
    )
    samples = batch_sample_fn(noise_model, sample_rngs)
    
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.4, c="black", linewidths=0)
    plt.axis("off")
    plt.tight_layout()
    log["samples"] = wandb.Image(fig)
    plt.close(fig)

  pbar.set_description(f"{train_loss=:.04f} {test_loss=:0.4f}")
  run.log(log, step=epoch)
