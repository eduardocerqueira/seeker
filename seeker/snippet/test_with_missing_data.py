#date: 2026-03-16T17:50:25Z
#url: https://api.github.com/gists/63b71e4637dd57a5b23d02c2b8025aca
#owner: https://api.github.com/users/DanWaxman

import arviz as az
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import MCMC, NUTS, Predictive

from dynestyx.simulators import DiscreteTimeSimulator
from tests.models import discrete_time_lti_simplified_model
from tests.test_utils import get_output_dir

SAVE_FIG = True


def _apply_missingness_pattern(
    obs_values: jnp.ndarray, missingness_pattern: str, missing_key
) -> jnp.ndarray:
    if missingness_pattern == "none":
        return obs_values

    n_obs = obs_values.shape[0]
    missing_values = jnp.full_like(obs_values, jnp.nan)
    keep_mask = jnp.ones((n_obs,), dtype=bool)

    if missingness_pattern == "random":
        # Randomly drop roughly 20% of observations.
        keep_mask = jr.bernoulli(missing_key, p=0.8, shape=(n_obs,))
    elif missingness_pattern == "sequential":
        # Regularly drop every 5th observation.
        keep_mask = (jnp.arange(n_obs) % 5) != 0
    elif missingness_pattern == "block":
        # Drop one contiguous middle block.
        block_len = max(1, n_obs // 5)
        block_start = (n_obs - block_len) // 2
        block_mask = (jnp.arange(n_obs) >= block_start) & (
            jnp.arange(n_obs) < block_start + block_len
        )
        keep_mask = ~block_mask

    return jnp.where(keep_mask[:, None], obs_values, missing_values)


@pytest.mark.parametrize("use_controls", [False, True])
@pytest.mark.parametrize("missingness_pattern", ["none", "random", "sequential", "block"])
@pytest.mark.parametrize("num_samples", [250])
def test_lti_system_missing_data_science(
    use_controls: bool,
    missingness_pattern: str,
    num_samples: int,
):
    """Discrete-time LTI using LTI_discrete factory with missing observations."""
    rng_key = jr.PRNGKey(0)

    data_init_key, mcmc_key, ctrl_key, missing_key = jr.split(rng_key, 4)

    true_alpha = 0.4
    # Longer timeseries (~200 obs) so data inform alpha more, like continuous LTI
    obs_times = jnp.arange(start=0.0, stop=200.0, step=1.0)

    ctrl_times = None
    ctrl_values = None
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        ctrl_times = obs_times

    true_params = {"alpha": jnp.array(true_alpha)}
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    with DiscreteTimeSimulator():
        synthetic = predictive(
            data_init_key,
            obs_times=obs_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )

    obs_values = synthetic["observations"].squeeze(0)
    obs_values = _apply_missingness_pattern(obs_values, missingness_pattern, missing_key)

    def data_conditioned_model():
        with DiscreteTimeSimulator():
            return discrete_time_lti_simplified_model(
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

    output_dir_name = (
        "test_lti_discrete_simplified"
        + ("_controlled" if use_controls else "")
        + f"_missing_{missingness_pattern}"
    )
    OUTPUT_DIR = get_output_dir(output_dir_name)

    plot_times = synthetic["times"]

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        plt.plot(
            plot_times.squeeze(0),
            synthetic["states"].squeeze(0)[:, 0],
            label="x[0]",
        )
        plt.plot(
            plot_times.squeeze(0),
            synthetic["states"].squeeze(0)[:, 1],
            label="x[1]",
        )
        plt.plot(
            plot_times.squeeze(0),
            synthetic["observations"].squeeze(0)[:, 0],
            label="observations",
            linestyle="--",
        )
        plt.legend()
        plt.savefig(OUTPUT_DIR / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    mcmc_key = jr.PRNGKey(0)
    mcmc = MCMC(
        NUTS(data_conditioned_model),
        num_samples=num_samples,
        num_warmup=num_samples,
    )

    mcmc.run(mcmc_key)

    posterior_samples = mcmc.get_samples()

    assert "alpha" in posterior_samples
    posterior_alpha = posterior_samples["alpha"]
    assert len(posterior_alpha) == num_samples
    assert not jnp.isnan(posterior_alpha).any()
    assert not jnp.isinf(posterior_alpha).any()

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        az.plot_posterior(
            posterior_alpha, hdi_prob=0.95, ref_val=true_params["alpha"].item()
        )
        plt.savefig(OUTPUT_DIR / "posterior_alpha.png", dpi=150, bbox_inches="tight")
        plt.close()

    true_alpha = true_params["alpha"]
    tol = 0.2
    assert jnp.abs(posterior_alpha.mean() - true_alpha) < tol

    hdi_data = az.hdi(posterior_alpha, hdi_prob=0.95)
    hdi_min = hdi_data["x"].sel(hdi="lower").item()
    hdi_max = hdi_data["x"].sel(hdi="higher").item()
    assert hdi_min <= true_alpha <= hdi_max, (
        f"True alpha {true_alpha} not in HDI {hdi_min}, {hdi_max}"
    )
