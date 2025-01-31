#date: 2025-01-31T16:41:34Z
#url: https://api.github.com/gists/4c71364f30331c2403c9aa65c789913e
#owner: https://api.github.com/users/kvr06-ai

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_spread_v3

################################################################################
# 1. Environment Setup and Policies
################################################################################

def make_env(seed=None):
    env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=50, continuous_actions=False)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        env.seed(seed)
    env.reset()
    return env

def random_policy(obs, agent_idx, env):
    return np.random.randint(0, 5)

def expert_policy(obs, agent_idx, env):
    ax, ay = obs[0], obs[1]
    lx = obs[2 + 2*agent_idx]
    ly = obs[3 + 2*agent_idx]
    dx = lx - ax
    dy = ly - ay
    move_h = 3 if dx > 0 else 1  # right vs left
    move_v = 4 if dy > 0 else 2  # up vs down
    return move_h if abs(dx) > abs(dy) else move_v

def semi_expert_policy(obs, agent_idx, env):
    if np.random.rand() < 0.8:
        return expert_policy(obs, agent_idx, env)
    else:
        return random_policy(obs, agent_idx, env)

################################################################################
# 2. Offline Data Collection
################################################################################

def run_episode(env, policy_dict, max_cycles=50):
    env.reset()
    transitions = []
    done = False
    step_count = 0

    while not done and step_count < max_cycles:
        agent = env.agent_selection
        obs, rew, term, trunc, info = env.last(observe=True)
        if term or trunc:
            done = True
            break

        action = policy_dict[agent](obs, int(agent[-1]), env)
        env.step(action)

        next_obs, next_rew, next_term, next_trunc, next_info = env.last(observe=True)
        transitions.append((agent, obs, action, next_obs))

        step_count += 1
        if next_term or next_trunc:
            done = True

    return transitions

def collect_offline_data(env_maker, policy_combos, episodes_per_policy=5):
    dataset = []
    for combo_name, policy_fns in policy_combos.items():
        for ep in range(episodes_per_policy):
            env = env_maker()
            policy_dict = {}
            for i in range(env.num_agents):
                policy_dict[f"agent_{i}"] = policy_fns[i]
            transitions = run_episode(env, policy_dict, max_cycles=env.max_cycles)
            dataset.append({
                "policy_name": combo_name,
                "transitions": transitions
            })
    return dataset

def add_agent0_random_data(env_maker, dataset, episodes=5):
    """
    Adds data for a scenario where agent_0 = random, agents 1 & 2 = expert.
    This ensures we have explicit coverage of unilateral deviation for agent_0.
    """
    combo_name = "agent0_random_others_expert"
    for _ in range(episodes):
        env = env_maker()
        policy_dict = {
            "agent_0": lambda obs, idx, e: random_policy(obs, idx, e),
            "agent_1": lambda obs, idx, e: expert_policy(obs, idx, e),
            "agent_2": lambda obs, idx, e: expert_policy(obs, idx, e)
        }
        transitions = run_episode(env, policy_dict, max_cycles=env.max_cycles)
        dataset.append({
            "policy_name": combo_name,
            "transitions": transitions
        })

################################################################################
# 3. Preference Labeling
################################################################################

def default_trajectory_score(transitions):
    """
    A simple function that prefers shorter episodes (fewer steps).
    """
    return -len(transitions)

def collisions_and_distance_score(transitions):
    """
    Example alternative scoring function. In a real scenario, 
    you might parse environment data for collisions, distances, etc.
    Here, we just randomly perturb a base score for demonstration.
    """
    base = -len(transitions)
    return base + np.random.randint(-5, 5)

def generate_preferences(dataset, score_fn, num_pairs=50):
    preferences = []
    n = len(dataset)
    all_indices = list(range(n))
    for _ in range(num_pairs):
        i, j = random.sample(all_indices, 2)
        traj_i = dataset[i]["transitions"]
        traj_j = dataset[j]["transitions"]
        score_i = score_fn(traj_i)
        score_j = score_fn(traj_j)
        preferred = 0 if score_i > score_j else 1
        preferences.append({
            "i_idx": i,
            "j_idx": j,
            "preferred": preferred
        })
    return preferences

################################################################################
# 4. Reward Model Variants
################################################################################

class RewardNetwork(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def extract_features(transitions):
    if not transitions:
        return np.zeros(16, dtype=np.float32)
    sum_feat = np.zeros(16, dtype=np.float32)
    count = 0
    for (agent, obs, act, nxt) in transitions:
        obs_arr = np.array(obs, dtype=np.float32)
        obs_arr = obs_arr[:16] if len(obs_arr) > 16 else np.pad(obs_arr, (0,16-len(obs_arr)))
        sum_feat += obs_arr
        count += 1
    if count > 0:
        sum_feat /= count
    return sum_feat

def build_dataset_for_reward_model(offline_data, preferences):
    training_pairs = []
    for pref in preferences:
        i_idx = pref["i_idx"]
        j_idx = pref["j_idx"]
        feat_i = extract_features(offline_data[i_idx]["transitions"])
        feat_j = extract_features(offline_data[j_idx]["transitions"])
        label = 1 if pref["preferred"] == 0 else 0
        training_pairs.append((feat_i, feat_j, label))
    return training_pairs

# A. Logistic (BCEWithLogitsLoss)
def train_reward_model_logistic(training_pairs, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardNetwork(input_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        random.shuffle(training_pairs)
        total_loss = 0.0
        for (featA, featB, lbl) in training_pairs:
            tA = torch.tensor(featA, dtype=torch.float, device=device).unsqueeze(0)
            tB = torch.tensor(featB, dtype=torch.float, device=device).unsqueeze(0)
            lbl_t = torch.tensor([lbl], dtype=torch.float, device=device)

            scoreA = model(tA)
            scoreB = model(tB)
            diff = scoreA - scoreB
            loss = criterion(diff, lbl_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(training_pairs)
        print(f"[Reward Logistic] Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")
    return model

# B. Hinge Loss
def train_reward_model_hinge(training_pairs, epochs=10, lr=1e-3, margin=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardNetwork(input_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        random.shuffle(training_pairs)
        total_loss = 0.0
        for (featA, featB, lbl) in training_pairs:
            tA = torch.tensor(featA, dtype=torch.float, device=device).unsqueeze(0)
            tB = torch.tensor(featB, dtype=torch.float, device=device).unsqueeze(0)
            scoreA = model(tA)
            scoreB = model(tB)
            if lbl == 1:  # A is better
                diff = scoreA - scoreB
            else:         # B is better
                diff = scoreB - scoreA
            hinge_loss = torch.clamp(margin - diff, min=0.0).mean()
            optimizer.zero_grad()
            hinge_loss.backward()
            optimizer.step()
            total_loss += hinge_loss.item()

        avg_loss = total_loss / len(training_pairs)
        print(f"[Reward Hinge] Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")
    return model

################################################################################
# 5. Offline Multi-Agent RL Methods
################################################################################

class SimpleVDN(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    def forward(self, x):
        return self.net(x)

def transition_to_feature(transition):
    (agent, obs, act, nxt) = transition
    obs_arr = np.array(obs, dtype=np.float32)[:8]
    nxt_arr = np.array(nxt, dtype=np.float32)[:8]
    if len(obs_arr) < 8:
        obs_arr = np.pad(obs_arr, (0,8-len(obs_arr)))
    if len(nxt_arr) < 8:
        nxt_arr = np.pad(nxt_arr, (0,8-len(nxt_arr)))
    return np.concatenate([obs_arr, nxt_arr], axis=0)

def compute_learned_reward(reward_model, transition):
    feats = extract_features([transition])
    with torch.no_grad():
        val = reward_model(torch.tensor(feats, dtype=torch.float).unsqueeze(0))
    return val.item()

# A. Single-step = reward
def offline_value_decomposition(dataset, reward_model, epochs=10, gamma=0.95, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qnet = SimpleVDN(input_dim=16, hidden_dim=64, n_actions=5).to(device)
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Flatten transitions
    offline_buffer = []
    for traj in dataset:
        offline_buffer.extend(traj["transitions"])

    for ep in range(epochs):
        random.shuffle(offline_buffer)
        losses = []
        for transition in offline_buffer:
            (agent, obs, act, nxt) = transition
            s_feat = transition_to_feature(transition)
            r_pred = compute_learned_reward(reward_model, transition)

            s_tensor = torch.tensor(s_feat, dtype=torch.float, device=device).unsqueeze(0)
            q_vals = qnet(s_tensor)
            q_val_a = q_vals[0, act]

            target = torch.tensor(r_pred, dtype=torch.float, device=device)
            loss = mse(q_val_a, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f"[Offline RL Orig] Epoch {ep+1}/{epochs}, Q-loss={avg_loss:.4f}")

    return qnet

# B. 1-step bootstrapped Q
def offline_value_decomposition_bootstrap(dataset, reward_model, epochs=10, gamma=0.95, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qnet = SimpleVDN(input_dim=16, hidden_dim=64, n_actions=5).to(device)
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Build a buffer of (s_feat, a, r, s_feat_next)
    offline_buffer = []
    for traj in dataset:
        transitions = traj["transitions"]
        for i in range(len(transitions)):
            t = transitions[i]
            (agent, obs, act, nxt) = t
            r_pred = compute_learned_reward(reward_model, t)
            s_feat = transition_to_feature(t)
            # next_feat
            if i+1 < len(transitions):
                s_feat_next = transition_to_feature(transitions[i+1])
            else:
                s_feat_next = None
            offline_buffer.append((s_feat, act, r_pred, s_feat_next))

    for ep in range(epochs):
        random.shuffle(offline_buffer)
        losses = []
        for (s_feat, a, r, s_feat_next) in offline_buffer:
            s_tensor = torch.tensor(s_feat, dtype=torch.float, device=device).unsqueeze(0)
            q_vals = qnet(s_tensor)
            q_val_a = q_vals[0, a]

            if s_feat_next is not None:
                ns_tensor = torch.tensor(s_feat_next, dtype=torch.float, device=device).unsqueeze(0)
                with torch.no_grad():
                    ns_q_vals = qnet(ns_tensor)
                    next_q = torch.max(ns_q_vals)
                target_val = torch.tensor(r, dtype=torch.float, device=device) + gamma*next_q
            else:
                target_val = torch.tensor(r, dtype=torch.float, device=device)

            loss = mse(q_val_a, target_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f"[Offline RL Boot] Epoch {ep+1}/{epochs}, Q-loss={avg_loss:.4f}")

    return qnet

################################################################################
# 6. Evaluation
################################################################################

def evaluate_learned_policy(env_maker, qnet, reward_model, episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_pred_rewards = []
    total_real_rewards = []

    for ep in range(episodes):
        env = env_maker()
        env.reset()
        done = False
        step_count = 0
        ep_pred = 0.0
        ep_real = 0.0

        while not done and step_count < env.max_cycles:
            agent = env.agent_selection
            obs, rew, term, trunc, info = env.last(observe=True)
            if term or trunc:
                done = True
                break

            # Argmax Q
            obs_arr = np.array(obs, dtype=np.float32)[:8]
            if len(obs_arr) < 8:
                obs_arr = np.pad(obs_arr, (0,8-len(obs_arr)))
            feat = np.concatenate([obs_arr, np.zeros(8, dtype=np.float32)], axis=0)

            with torch.no_grad():
                s_tensor = torch.tensor(feat, dtype=torch.float).unsqueeze(0).to(device)
                q_vals = qnet(s_tensor)
                action = torch.argmax(q_vals, dim=1).item()

            env.step(action)
            next_obs, next_rew, next_term, next_trunc, next_info = env.last(observe=True)

            transition = (agent, obs, action, next_obs)
            pred_r = compute_learned_reward(reward_model, transition)
            ep_pred += pred_r
            ep_real += rew

            step_count += 1
            if next_term or next_trunc:
                done = True

        total_pred_rewards.append(ep_pred)
        total_real_rewards.append(ep_real)

    avg_pred = np.mean(total_pred_rewards)
    avg_real = np.mean(total_real_rewards)
    print(f"[Evaluation] Over {episodes} episodes:")
    print(f"  Avg Learned Reward = {avg_pred:.3f}")
    print(f"  Avg Real Env Reward= {avg_real:.3f}\n")
    return avg_pred, avg_real

def check_unilateral_deviation(env_maker, qnet, reward_model, agent_to_deviate, deviation_policy, episodes=5):
    print("Baseline (all Qnet):")
    base_pred, base_real = evaluate_learned_policy(env_maker, qnet, reward_model, episodes=episodes)

    def qnet_policy(obs, idx, env):
        obs_arr = np.array(obs, dtype=np.float32)[:8]
        if len(obs_arr) < 8:
            obs_arr = np.pad(obs_arr, (0,8-len(obs_arr)))
        feat = np.concatenate([obs_arr, np.zeros(8, dtype=np.float32)], axis=0)
        with torch.no_grad():
            s_tensor = torch.tensor(feat, dtype=torch.float).unsqueeze(0)
            q_vals = qnet(s_tensor)
            return torch.argmax(q_vals, dim=1).item()

    def policy_selector(agent_name):
        idx = int(agent_name[-1])
        if idx == agent_to_deviate:
            return lambda o, i, e: deviation_policy(o, i, e)
        else:
            return lambda o, i, e: qnet_policy(o, i, e)

    total_pred_rewards = []
    total_real_rewards = []

    print(f"Deviation (agent_{agent_to_deviate} => {deviation_policy.__name__}):")
    for ep in range(episodes):
        env = env_maker()
        env.reset()
        done = False
        step_count = 0
        ep_pred = 0.0
        ep_real = 0.0

        while not done and step_count < env.max_cycles:
            agent = env.agent_selection
            obs, rew, term, trunc, info = env.last(observe=True)
            if term or trunc:
                done = True
                break

            pol_fn = policy_selector(agent)
            action = pol_fn(obs, int(agent[-1]), env)

            env.step(action)
            next_obs, next_rew, next_term, next_trunc, next_info = env.last(observe=True)

            transition = (agent, obs, action, next_obs)
            pred_r = compute_learned_reward(reward_model, transition)
            ep_pred += pred_r
            ep_real += rew

            step_count += 1
            if next_term or next_trunc:
                done = True

        total_pred_rewards.append(ep_pred)
        total_real_rewards.append(ep_real)

    dev_pred = np.mean(total_pred_rewards)
    dev_real = np.mean(total_real_rewards)

    print(f"  Dev. avg learned reward={dev_pred:.3f}")
    print(f"  Dev. avg real env reward={dev_real:.3f}\n")

    return {
        "baseline_pred": base_pred,
        "baseline_real": base_real,
        "deviation_pred": dev_pred,
        "deviation_real": dev_real
    }

################################################################################
# 7. Main Pipeline
################################################################################

def policy_for_agent_i(agent_idx, policy_type):
    def _wrapped(obs, idx, env):
        if policy_type == "random":
            return random_policy(obs, idx, env)
        elif policy_type == "expert":
            return expert_policy(obs, idx, env)
        elif policy_type == "semi_expert":
            return semi_expert_policy(obs, idx, env)
        return random_policy(obs, idx, env)
    return _wrapped

def run_experiment(
    reward_model_variant="hinge",
    rl_variant="bootstrap",
    extra_random_data=True,
    episodes_per_policy=5
):
    random.seed(42)
    np.random.seed(42)

    # Define policy combos
    policy_combos = {
        "expert": [
            policy_for_agent_i(i, "expert") for i in range(3)
        ],
        "semi_expert": [
            policy_for_agent_i(i, "semi_expert") for i in range(3)
        ],
        "random": [
            policy_for_agent_i(i, "random") for i in range(3)
        ],
        "mixed": [
            policy_for_agent_i(0, "expert"),
            policy_for_agent_i(1, "random"),
            policy_for_agent_i(2, "semi_expert")
        ]
    }

    # 1) Collect data
    dataset = collect_offline_data(make_env, policy_combos, episodes_per_policy=episodes_per_policy)

    # Optionally add explicit coverage scenario
    if extra_random_data:
        add_agent0_random_data(make_env, dataset, episodes=episodes_per_policy)

    print(f"\nFinal offline dataset size: {len(dataset)} trajectories.\n")

    # 2) Generate preferences
    # We can pick between default_trajectory_score or collisions_and_distance_score
    preferences = generate_preferences(dataset, default_trajectory_score, num_pairs=50)
    # or: preferences = generate_preferences(dataset, collisions_and_distance_score, num_pairs=50)

    # 3) Build training pairs
    training_pairs = build_dataset_for_reward_model(dataset, preferences)

    # 4) Train reward model
    if reward_model_variant == "hinge":
        reward_model = train_reward_model_hinge(training_pairs, epochs=10, lr=1e-3, margin=1.0)
    else:
        reward_model = train_reward_model_logistic(training_pairs, epochs=10, lr=1e-3)

    # 5) Offline multi-agent RL
    if rl_variant == "bootstrap":
        qnet = offline_value_decomposition_bootstrap(dataset, reward_model, epochs=10, gamma=0.95, lr=1e-3)
    else:
        qnet = offline_value_decomposition(dataset, reward_model, epochs=10, gamma=0.95, lr=1e-3)

    # 6) Evaluate
    print("Evaluating learned Q-net policy:\n")
    evaluate_learned_policy(make_env, qnet, reward_model, episodes=5)

    # 7) Check unilateral deviation
    print("Unilateral Deviation (agent_0 => random):\n")
    check_unilateral_deviation(make_env, qnet, reward_model, agent_to_deviate=0, deviation_policy=random_policy)

# Run everything if this is the main cell
if __name__ == "__main__":
    # Example usage: run with hinge reward + bootstrap Q, adding extra coverage data
    run_experiment(
        reward_model_variant="hinge",
        rl_variant="bootstrap",
        extra_random_data=True,
        episodes_per_policy=5
    )
