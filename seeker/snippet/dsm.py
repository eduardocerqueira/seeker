#date: 2025-12-19T16:53:04Z
#url: https://api.github.com/gists/6e674f6bb9cb423edece8742c47aa4fd
#owner: https://api.github.com/users/johannes-spies

from typing import Callable
from functools import partial
import optax
from flax import nnx
import jax, jax.numpy as jnp, jax.random as jr

class Dsm(nnx.Module):
  "Denoising score-matching."
  def __init__(self, beta: jax.Array):
    self.beta = beta
  
  diffusion_steps = property(lambda self: len(self.beta))
  alpha = property(lambda self: 1 - self.beta)
  alpha_bar = property(lambda self: jnp.cumprod(self.alpha))
  sigma = property(lambda self: jnp.sqrt(self.beta))

  def loss(
    self,
    noise_model: Callable[[jax.Array, jax.Array], jax.Array],
    xt: jax.Array,
    t: jax.Array,
    eps: jax.Array,
  ) -> jax.Array:
    predictions = noise_model(xt, t)
    return optax.losses.l2_loss(predictions, eps) # type: ignore

  def get_noised_sample(self, x0: jax.Array, t: jax.Array, eps: jax.Array) -> jax.Array:
    """
    This is the cool thing about diffusion: we don't need to simulate the forward
    process step by step. For a gaussian transition kernel, we can directly sample
    $x_t$ given $x_0$ at any time step $t$.
    """
    return jnp.sqrt(self.alpha_bar[t]) * x0 + jnp.sqrt(1 - self.alpha_bar[t]) * eps
  
  def batch_loss(
    self,
    noise_model: Callable[[jax.Array, jax.Array], jax.Array],
    xts: jax.Array,
    ts: jax.Array,
    noises: jax.Array,
  ) -> jax.Array:
    batch_loss = jax.vmap(self.loss, (None, 0, 0, 0))
    losses = batch_loss(noise_model, xts, ts, noises)
    return jnp.mean(losses)

  def train_step(
    self,
    noise_model: Callable[[jax.Array, jax.Array], jax.Array],
    optimizer: nnx.Optimizer,
    batch: jax.Array,
    rng: jax.Array,
  ):
    # compute batch input
    rng_noise, rng_time = jr.split(rng)
    batch_size, *_ = batch.shape
    batch_noises = jr.normal(rng_noise, shape=batch.shape)
    batch_times = jr.randint(rng_time, (batch_size,), 0, self.diffusion_steps)
    batch_noised_samples = jax.vmap(self.get_noised_sample)(batch, batch_times, batch_noises)

    # compute loss and update
    value_and_grad = nnx.value_and_grad(self.batch_loss)
    loss, grads = value_and_grad(noise_model, batch_noised_samples, batch_times, batch_noises)
    optimizer.update(noise_model, grads)

    return loss
  
  def _sample_step(
    self,
    noise_model: Callable[[jax.Array, jax.Array], jax.Array],
    xt: jax.Array,
    rng: jax.Array,
    t: jax.Array,
  ) -> jax.Array:
    alpha_t = self.alpha[t]
    alpha_bar_t = self.alpha_bar[t]
    sigma_t = self.sigma[t]
    noise = noise_model(xt, t)
    z = jnp.where(t != 0, jr.normal(rng, shape=xt.shape), jnp.zeros_like(xt))
    return 1 / jnp.sqrt(alpha_t) * (xt - (1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t) * noise) + sigma_t * z
  
  @partial(nnx.jit, static_argnames=["sample_shape", "return_trajectory"])
  def sample(
    self,
    noise_model: Callable[[jax.Array, jax.Array], jax.Array],
    sample_shape: tuple[int, ...] = (),
    return_trajectory: bool = False,
    *,
    rngs: nnx.Rngs,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    def body(xt, rng_and_t):
      rng, t = rng_and_t
      new = self._sample_step(noise_model, xt, rng, t)
      return new, new
    
    timesteps = jnp.arange(self.diffusion_steps)[::-1]
    sample_rngs = jr.split(rngs.sample(), len(timesteps))
    # the last sample_rng will not be used in the loop because z is substituted with zeros
    # -> we can steal the rng for the initial sample
    xt_init = jr.normal(sample_rngs[-1], shape=sample_shape)
    x0, trajectory = jax.lax.scan(body, init=xt_init, xs=(sample_rngs, timesteps))
    if return_trajectory:
      return x0, trajectory
    return x0
  
  @classmethod
  def linear(cls, diffusion_steps: int, beta_start: float, beta_end: float):
    beta = jnp.linspace(beta_start, beta_end, diffusion_steps)
    return cls(beta)