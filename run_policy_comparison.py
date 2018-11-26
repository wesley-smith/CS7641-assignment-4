import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

from dennybritz_policy_iteration import policy_iteration
from dennybritz_value_iteration import value_iteration

from constants import FL4x4, FL8x8, TERM_STATE_MAP, GOAL_STATE_MAP
from helpers import visualize_policy, visualize_value, visualize_env
from my_frozen_lake import FrozenLakeEnv

# Change this to work on a different environment
ENV_NAME = FL8x8

def count_different_entries(a, b):
    assert a.size == b.size, 'Arrays need to be the same size'
    return a.size - np.sum(np.isclose(a, b))

if __name__ == '__main__':
    gamma = 0.9
    theta = 0.0001
    env_kwargs = {
        'map_name': ENV_NAME,
        'slip_rate': .2,
        'rewards': (-0.04, -1, 1)
    }
    vi_env = FrozenLakeEnv(**env_kwargs)
    vi_env = vi_env.unwrapped
    vi_policy, vi_V, _, _ = value_iteration(vi_env, discount_factor=gamma, theta=theta)
    pi_env = FrozenLakeEnv(**env_kwargs)
    pi_env = pi_env.unwrapped
    pi_policy, pi_V, _, _ = policy_iteration(pi_env, discount_factor=gamma, theta=theta)

    assert np.all(np.isclose(pi_policy, vi_policy, atol=0.05)), "Policies don't match"
    assert np.all(np.isclose(pi_V, vi_V, atol=0.05)), "Values don't match"
    visualize_policy(vi_policy, ENV_NAME, vi_env.desc.shape, 'Optimal policy - Modified transition model')
    visualize_value(vi_V, ENV_NAME, vi_env.desc.shape, 'Value estimates - Modified transition model')
