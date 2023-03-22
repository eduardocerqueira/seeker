#date: 2023-03-22T16:53:49Z
#url: https://api.github.com/gists/604269e379f70548b61a9adea9ee3d30
#owner: https://api.github.com/users/vkurenkov

# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
from typing import Callable, Dict, Optional, Tuple
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random

import d4rl
import gym
import numpy as np
import pyrallis
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class Hparams:
    # Experiment
    device: str = "cuda"  # PyTorch device
    policy: str = "IQL"  # Policy name
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    save_model: bool = False  # Save model and optimizer parameters
    save_path: str = "./models/iql"  # Checkpoints location
    load_model: str = ""  # Model load file name, "" doesn't load
    n_seeds: int = 10  # Number of seeds to run
    # IQL
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_determenistic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    # Wandb logging
    use_wandb: bool = True  # Use wandb logging
    # Checkpoints
    file_name: str = f"{policy}_{env}"  # Logging prefix


# Gives name to the wandb run
def get_run_name(args_cmd: Hparams) -> str:
    """
    Gives name to the wandb run
    :param args_cmd: run parameters
    :return: name based on run parameters
    """
    return "_".join([args_cmd.policy, args_cmd.env])


# Initializes wandb run
def wandb_init(args: Hparams):
    """
    Initializes wandb logging
    :param args: run parameters
    """
    name = get_run_name(args)
    wandb.init(project="CORL", config=asdict(args))
    wandb.run.name = name
    wandb.run.save()


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    """
    Sets random seeds everywhere
    :param seed: seed to set
    :param env: environment to set seed in
    :param deterministic_torch: use deterministic torch or not
    :return:
    """
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(
    policy: Callable[[np.ndarray], np.ndarray],
    env_name: str,
    seed: int,
    mean: float,
    std: float,
    seed_offset: int = 100,
    eval_episodes: int = 10,
) -> float:
    """
    Evaluate policy online on multiple episodes
    :param policy: policy to run
    :param env_name: testing environment
    :param seed: random seed
    :param mean: state normalization mean
    :param std: state normalization std
    :param seed_offset: seed offset
    :param eval_episodes: number of episodes to run
    :return: average d4rl score
    """
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes: "
        f"{avg_reward:.3f}, D4RL score: {d4rl_score:.3f}"
    )
    print("---------------------------------------")
    return d4rl_score


class ReplayBuffer:
    """Standard replay buffer"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = 1000000,
        device: str = "cpu",
    ):
        """
        Constructs replay buffer
        :param state_dim: environment state dimensions
        :param action_dim: environment actions dimensions
        :param max_size: maximal buffer size
        :param device: device for training
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    # Adds transaction to replay buffer
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ):
        """
        Adds transaction to replay buffer
        :param state: new state
        :param action: new action
        :param next_state: new next state
        :param reward: new reward
        :param done: is episode done or not
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # Samples batch from replay buffer
    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples batch from buffer
        :param batch_size: number of transactions to sample
        :return: Transactions information.
        batch of states [batch_size, state_dim]
        batch of actions [batch_size, action_dim]
        batch of next states [batch_size, state_dim]
        batch of rewards [batch_size, 1]
        batch of not_dones [batch_size, 1]
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

    # Stores d4rl dataset to replay buffer
    def convert_d4rl(self, dataset: Dict[str, np.ndarray]):
        """
        Extracts data from d4rl dataset to buffer
        :param dataset: d4rl dataset
        """
        self.state = dataset["observations"]
        self.action = dataset["actions"]
        self.next_state = dataset["next_observations"]
        self.reward = dataset["rewards"].reshape(-1, 1)
        self.not_done = 1.0 - dataset["terminals"].reshape(-1, 1)
        self.size = self.state.shape[0]

    # Normalizes states in replay buffer
    def normalize_states(self, eps: float = 1e-3):
        """
        Normalizes states
        :param eps: numerical stability constant
        :return: states mean and std
        """
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std


def soft_update(target: nn.Module, source: nn.Module, alpha: float):
    """
    Performs soft update
    :param target: network to update
    :param source: source of new weights
    :param alpha: how much to add
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - alpha).add_(source_param.data, alpha=alpha)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Calculates asymmetric L2 loss
    :param u: differences
    :param tau: offset
    :return: loss value
    """
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    """Squeeze wrapper"""

    def __init__(self, dim=-1):
        """
        Initializes squeeze wrapper
        :param dim: squeeze dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs squeeze
        :param x: input data
        :return: squeezed data
        """
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    """Vanilla MLP"""

    def __init__(
        self,
        dims,
        activation: nn.Module = nn.ReLU,
        output_activation: nn.Module = None,
        squeeze_output: bool = False,
    ):
        """
        Initializes MLP
        :param dims: network dimensions
        :param activation: activation between layers
        :param output_activation: output activation
        :param squeeze_output: squeeze output

        Raises:
            ValueError: if not enough dims.
        """
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation is not None:
            layers.append(output_activation())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass
        :param x: input data
        :return: output data
        """
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Stochastic policy"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        """
        Initializes policy
        :param obs_dim: environment state space dimensions
        :param act_dim: environment action space dimensions
        :param max_action: environment max actions values
        :param hidden_dim: network hidden dims
        :param n_hidden: number of network layers
        """
        super().__init__()
        self.net = MLP([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> MultivariateNormal:
        """
        Policy forward pass
        :param obs: environment state [batch_size, obs_dim]
        :return: new action distribution [batch_size, act_dim]
        """
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False, enable_grad: bool = False
    ):
        """
        Performs action
        :param obs: environment state [batch_size, obs_dim]
        :param deterministic: deterministic or stochastic
        :param enable_grad: enable grad flow
        :return: new action
        """
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            action = dist.mean if deterministic else dist.sample()
            action = torch.clamp(
                self.max_action * action, -self.max_action, self.max_action
            )
            return action


class DeterministicPolicy(nn.Module):
    """Deterministic policy"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        """
        Initializes policy
        :param obs_dim: environment state space dimensions
        :param act_dim: environment action space dimensions
        :param max_action: environment max actions values
        :param hidden_dim: network hidden dims
        :param n_hidden: number of network layers
        """
        super().__init__()
        self.net = MLP(
            [obs_dim, *([hidden_dim] * n_hidden), act_dim], output_activation=nn.Tanh
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Policy forward pass
        :param obs: environment state [batch_size, obs_dim]
        :return: new action [batch_size, act_dim]
        """
        return self.net(obs)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False, enable_grad: bool = False
    ):
        """
        Performs action
        :param obs: environment state [batch_size, obs_dim]
        :param deterministic: deterministic or stochastic
        :param enable_grad: enable grad flow
        :return: new action
        """
        with torch.set_grad_enabled(enable_grad):
            return torch.clamp(
                self(obs) * self.max_action, -self.max_action, self.max_action
            )


class TwinQ(nn.Module):
    """Twin Q-value network"""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        """
        Initializes Q networks
        :param state_dim: environment state space dimensions
        :param action_dim: environment action space dimensions
        :param hidden_dim: network hidden dims
        :param n_hidden: number of network layers
        """
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets values of both Q functions
        :param state: environment state [batch_size, state_dim]
        :param action: action in environment [batch_size, action_dim]
        :return: Q-values [2, batch_size, action_dim]
        """
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Gets minimum of Q-functions
        :param state: environment state [batch_size, state_dim]
        :param action: action in environment [batch_size, action_dim]
        :return: Q-values [batch_size, action_dim]
        """
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    """Value function network"""

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        """
        Initializes V network
        :param state_dim: environment state space dimensions
        :param hidden_dim: network hidden dims
        :param n_hidden: number of network layers
        """
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Gets Value function value
        :param state: environment state [batch_size, state_dim]
        :return: Q-values [batch_size, action_dim]
        """
        return self.v(state)


class ImplicitQLearning(Callable):
    """IQL algorithm"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        iql_determenistic: bool = False,
        device: str = "cpu",
    ):
        """
        Initializes IQL
        :param state_dim: environment state space dimensions
        :param action_dim: environment action space dimensions
        :param max_action: environment max actions values
        :param iql_tau: coefficient for asymmetric loss
        :param beta: inverse temperature
        :param max_steps: max time steps to run environment
        :param lr: networks learning rate
        :param discount: discount factor
        :param tau: target networks update rate
        :param iql_determenistic: use deterministic actor
        :param device: device for training
        """
        self.max_action = max_action
        self.qf = TwinQ(state_dim, action_dim).to(device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = ValueFunction(state_dim).to(device)
        self.policy = (
            DeterministicPolicy(state_dim, action_dim, max_action).to(device)
            if iql_determenistic
            else GaussianPolicy(state_dim, action_dim, max_action).to(device)
        )
        self.v_optimizer = Adam(self.vf.parameters(), lr=lr)
        self.q_optimizer = Adam(self.qf.parameters(), lr=lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Get action based on state
        :param state: environment state [state_dim]
        :return: new action [action_dim]
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return (
            self.policy.act(state, deterministic=True, enable_grad=False)
            .cpu()
            .data.numpy()
            .flatten()
        )

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        """
        Performs V-network update
        :param observations: batch of observations [batch_size, state_dim]
        :param actions: batch of actions [batch_size, action_dim]
        :param log_dict: logging dictionary
        :return: advantage used for other updates [batch_size, 1]
        """
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v,
        observations,
        actions,
        rewards,
        terminals,
        log_dict,
    ):
        """
        Performs Q-network update
        :param next_v: V-function value for next state
        :param observations: batch of observations [batch_size, state_dim]
        :param actions: batch of actions [batch_size, action_dim]
        :param rewards: batch of rewards [batch_size, 1]
        :param terminals: batch of terminal flags [batch_size, 1]
        :param log_dict: logging dictionary
        """
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(self, adv, observations, actions, log_dict):
        """
        Performs policy update
        :param adv: state advantage
        :param observations: batch of observations [batch_size, state_dim]
        :param actions: batch of actions [batch_size, action_dim]
        :param log_dict: logging dictionary

        Raises:
            RuntimeError: if dimensions missmatch.
            NotImplementedError: if unsupported output
        """
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

    def train(
        self, replay_buffer: ReplayBuffer, batch_size: int = 256
    ) -> Dict[str, float]:
        """
        Performs single training step
        :param replay_buffer: replay buffer to sample from
        :param batch_size: batch size
        :return: dictionary with logging information [metric -> value]
        """
        self.total_it += 1
        (
            observations,
            actions,
            next_observations,
            rewards,
            not_done,
        ) = replay_buffer.sample(batch_size)
        terminals = 1.0 - not_done
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        terminals = terminals.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, terminals, log_dict)
        # Update policy
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def save(self, filename: Path):
        """
        Saves model into files
        :param filename: path for saving
        """
        str_filename = str(filename)

        torch.save(self.qf.state_dict(), Path(str_filename + "_q"))
        torch.save(self.q_optimizer.state_dict(), Path(str_filename + "_q_optimizer"))

        torch.save(self.vf.state_dict(), Path(str_filename + "_value"))
        torch.save(
            self.v_optimizer.state_dict(), Path(str_filename + "_value_optimizer")
        )

        torch.save(self.policy.state_dict(), Path(str_filename + "_actor"))
        torch.save(
            self.policy_optimizer.state_dict(), Path(str_filename + "_actor_optimizer")
        )
        torch.save(
            self.policy_lr_schedule.state_dict(), Path(str_filename + "_actor_scheduler")
        )

    def load(self, filename: Path):
        """
        Loads model from files
        :param filename: path for loading
        """
        str_filename = str(filename)

        self.qf.load_state_dict(torch.load(Path(str_filename + "_q")))
        self.q_optimizer.load_state_dict(torch.load(Path(str_filename + "_q_optimizer")))
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(torch.load(Path(str_filename + "_value")))
        self.v_optimizer.load_state_dict(
            torch.load(Path(str_filename + "_value_optimizer"))
        )

        self.policy.load_state_dict(torch.load(Path(str_filename + "_actor")))
        self.policy_optimizer.load_state_dict(
            torch.load(Path(str_filename + "_actor_optimizer"))
        )
        self.policy_lr_schedule.load_state_dict(
            torch.load(Path(str_filename + "_actor_scheduler"))
        )


# Parse arguments and run training
@pyrallis.wrap()
def main(args: Hparams):
    """Runs training process"""

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "device": args.device,
        # IQL
        "beta": args.beta,
        "iql_tau": args.iql_tau,
        "iql_determenistic": args.iql_determenistic,
        "max_steps": args.max_timesteps,
    }

    for seed in range(args.seed, args.seed + args.n_seeds):
        print("---------------------------------------")
        print(f"Policy: {args.policy}, Env: {args.env}, Seed: {seed}")
        print("---------------------------------------")

        file_name = args.file_name + f"_{seed}"

        # Set seeds
        set_seed(seed, env)
        # Initialize policy
        policy = ImplicitQLearning(**kwargs)

        if args.load_model != "":
            policy.load(Path(args.load_model))

        replay_buffer = ReplayBuffer(state_dim, action_dim, device=args.device)
        replay_buffer.convert_d4rl(d4rl.qlearning_dataset(env))
        if args.normalize:
            mean, std = replay_buffer.normalize_states()
        else:
            mean, std = 0, 1

        if args.use_wandb:
            wandb_init(args)

        evaluations = []
        for t in range(int(args.max_timesteps)):
            log_dict = policy.train(replay_buffer, args.batch_size)
            if args.use_wandb:
                wandb.log(log_dict, step=policy.total_it)
            # Evaluate episode
            if (t + 1) % args.eval_freq == 0:
                print(f"Time steps: {t + 1}")
                evaluations.append(eval_policy(policy, args.env, seed, mean, std))
                checkpoint_path = Path(save_path, file_name)
                if args.save_model:
                    policy.save(checkpoint_path)
                if args.use_wandb:
                    wandb.log(
                        {"d4rl_normalized_score": evaluations[-1]}, step=policy.total_it
                    )
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()