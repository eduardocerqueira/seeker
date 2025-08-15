#date: 2025-08-15T17:06:44Z
#url: https://api.github.com/gists/6f309643e29050220c1ae5e9e2c6c3a2
#owner: https://api.github.com/users/andrewsykim

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import linen as nn
from typing import Sequence
import os

import ray
import ray.train
from ray.train.v2.api.config import ScalingConfig, RunConfig
from ray.train.v2.jax import JaxTrainer

from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from flax.core.frozen_dict import FrozenDict


class TrainState(train_state.TrainState):
    pass


class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(features=feat, name=f'dense_{i}')(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x


def get_synthetic_batch(key: jax.random.PRNGKey, batch_size: int, input_dim: int, output_dim: int):
    """Generates a batch of synthetic data."""
    input_key, label_key, noise_key = jax.random.split(key, 3)
    inputs = jax.random.normal(input_key, (batch_size, input_dim))
    true_w = jnp.arange(input_dim, dtype=jnp.float32).reshape((input_dim, 1))
    true_b = 0.5
    labels = inputs @ true_w + true_b + \
        jax.random.normal(noise_key, (batch_size, output_dim)) * 0.1
    return {'inputs': inputs, 'labels': labels}


def create_train_state(rng: jax.random.PRNGKey, learning_rate: float, model: nn.Module, input_shape: tuple) -> TrainState:
    """Creates an initial TrainState on the host CPU."""
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state: TrainState, batch: dict):
    """A single training step on sharded data."""
    def loss_fn(params: FrozenDict):
        predictions = state.apply_fn({'params': params}, batch['inputs'])
        loss = jnp.mean((predictions - batch['labels']) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_loop_per_worker(config):
    num_devices = len(jax.devices())
    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(device_mesh, axis_names=('dp',))

    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    global_batch_size = config["global_batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]

    assert global_batch_size % num_devices == 0, "Batch size must be divisible by num devices."
    per_device_batch_size = global_batch_size // num_devices

    param_sharding = NamedSharding(mesh, PartitionSpec())
    data_sharding = NamedSharding(mesh, PartitionSpec('dp', None))
    batch_sharding = {'inputs': data_sharding, 'labels': data_sharding}

    key = jax.random.PRNGKey(0)
    model = SimpleMLP(features=[64, 32, output_dim])
    key, init_key = jax.random.split(key)

    unsharded_state = create_train_state(
        init_key, learning_rate, model,
        input_shape=(per_device_batch_size, input_dim)
    )

    state_sharding = jax.tree_util.tree_map(
        lambda x: param_sharding, unsharded_state,
        is_leaf=lambda x: isinstance(x, jax.Array)
    )
    state = jax.device_put(unsharded_state, state_sharding)

    if jax.process_index() == 0:
        print(f"Starting training on {num_devices} devices...")

    for epoch in range(num_epochs):
        key, batch_key = jax.random.split(key)
        host_batch = get_synthetic_batch(
            batch_key, global_batch_size, input_dim, output_dim
        )
        sharded_batch = jax.device_put(host_batch, batch_sharding)

        state, loss = train_step(state, sharded_batch)

        # TODO: this raises an exception, fix it later
        ray.train.report({"loss": loss.item()})

        if (epoch + 1) % 20 == 0 and jax.process_index() == 0:
            print(f"Epoch {epoch+1:3d}, Loss: {loss:.4f}")

    if jax.process_index() == 0:
        print("Training loop finished.")


def main():
    print("Starting JAX training with Ray Train...")

    trainer = JaxTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "input_dim": 10,
            "output_dim": 1,
            "global_batch_size": 1024,
            "num_epochs": 200,
            "learning_rate": 0.01,

        },
        scaling_config=ScalingConfig(
            use_tpu=True,
            num_workers=4,
            topology="4x4",
            accelerator_type="TPU-V6E",
            resources_per_worker={"TPU": 4},
            placement_strategy="SPREAD",
        ),
        run_config=RunConfig(
            name="fixed-jaxtrainer-example",
        ),
    )

    result = trainer.fit()
    print("Training complete!")

    # Fetching final loss causes errors, fix later.
    print(f"Final loss (reported by Ray Train): {result.metrics['loss']}")


if __name__ == "__main__":
    main()
