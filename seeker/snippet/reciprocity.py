#date: 2022-05-20T17:15:20Z
#url: https://api.github.com/gists/517ae16ce40765a1a1ec55970511834b
#owner: https://api.github.com/users/cool-RR

# This is a small experiment in learning reciprocity. This is part of my research, details here:
# https://r.rachum.com
#
# Feel free to run this experiment, and also play with the constants defined below.
#
# First install the dependencies:
#
#    pip install gym numpy stable-baselines3
#
# Run this script, and you'll see this output:
#
#    Starting reciprocity experiment.
#    Evaluating score... Done.
#    Score before training: -14.05
#    Sample game before training: CCCCDCDDDC
#
#    Training for 3000 steps... Done.
#
#    Evaluating score... Done.
#    Score after training: -12.24
#    Sample game after training: CCCCCCCCCC
#
# What's happening here? We have a learning agent playing 10 rounds of Prisoner's Dilemma against a
# hardcoded Tit-For-Tat opponent. The magic property that we're interested in here is reciprocity,
# and that's demonstrated by the hardcoded Tit-For-Tat opponent. That opponent plays Cooperate on
# the first round, and then on any subsequent round it plays the same move that the learning player
# played on the previous round.
#
# This experiment shows how reciprocity can be taught. When our learning player first plays, it
# plays an arbitrary sequence of Cooperate and Defect actions. When it's trained, it learns by
# trial-and-error that the only way to win here is to cooperate back. When it's finished training,
# it plays only Cooperate. It increases its score from around -14 to around -12.
#
# The next step is to get agents to learn reciprocity without using a hardcoded agent: http://r.rachum.com/emergent-reciprocity
#
# Sign up to get updates about my research here: http://r.rachum.com/announce

import os
import logging
from typing import Tuple

# Avoid TensorFlow spam:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import gym.spaces
import stable_baselines3.common.evaluation
import stable_baselines3.common.monitor

N_TRAINING_STEPS = 3_000
N_ROUNDS = 10


class TitForTatEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.MultiDiscrete([N_ROUNDS + 1, 3])

    def reset(self) -> np.ndarray:
        self.i_round = 0
        self.last_opponent_move = 2
        self.last_move = None
        self.is_end = False
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        return np.array([self.i_round, self.last_opponent_move])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.last_opponent_move = self.last_move if self.i_round >= 1 else 1
        self.last_move = action
        self.i_round += 1
        self.is_end = (self.i_round == N_ROUNDS)

        if self.last_move == 0:
            reward = 0 if (self.last_opponent_move == 1) else -2
        else:
            reward = -1 if (self.last_opponent_move == 1) else -3

        return self.get_observation(), reward, self.is_end, {}


def get_sample_game(policy: stable_baselines3.common.base_class.BaseAlgorithm,
                    env: gym.Env) -> str:
    observation = env.reset()
    result = ''
    while not env.is_end:
        action, _state = policy.predict(observation, deterministic=True)
        result += 'DC'[action]
        observation, reward, done, info = env.step(action)
    return result


def reciprocity():
    print('Starting reciprocity experiment.')
    gym.envs.register('TitForTat-v0', entry_point='reciprocity:TitForTatEnv')
    env = stable_baselines3.common.monitor.Monitor(gym.make('TitForTat-v0'))
    policy = stable_baselines3.PPO('MlpPolicy', env, verbose=False)
    get_score = lambda: stable_baselines3.common.evaluation.evaluate_policy(
        policy, env, n_eval_episodes=1_000, deterministic=False)[0]

    print('Evaluating score... ', end='')
    score_before_training = get_score()
    print('Done.')


    print(f'Score before training: {score_before_training:.2f}')
    print(f'Sample game before training: {get_sample_game(policy, env)}')

    print(f'\nTraining for {N_TRAINING_STEPS} steps... ', end='')
    policy.learn(N_TRAINING_STEPS)
    print('Done.\n')

    print('Evaluating score... ', end='')
    score_after_training = get_score()
    print('Done.')

    print(f'Score after training: {score_after_training:.2f}')
    print(f'Sample game after training: {get_sample_game(policy, env)}')


if __name__ == '__main__':
    reciprocity()