from pprint import pprint

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

from constants import FL4x4, FL8x8, TERM_STATE_MAP, GOAL_STATE_MAP
from helpers import visualize_policy, visualize_value

from dennybritz_q_learning import q_learning
import dennybritz_plotting as plotting
from my_frozen_lake import FrozenLakeEnv


# Change this to work on a different environment
ENV_NAME = FL8x8

def get_state_action_value(final_policy):
    return np.max(final_policy, axis=1)

def plot_epsilon_decay(epsilon, decay, n_episodes, stats):
    e_prime = epsilon * np.ones(n_episodes)
    for ix in range(n_episodes):
        e_prime[ix] *= decay ** ix
    smoothing_window=10
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed, label='Episode reward')
    plt.plot(e_prime, label='Decayed epsilon value', linestyle='--')
    plt.title("Epsilon-greedy with decay (epsilon=1.0, decay=0.999)")
    plt.xlabel('Episode')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    env = FrozenLakeEnv(
        map_name=ENV_NAME,
        rewards=(-0.04, -1, +1), # living, hole, goal
        slip_rate=0.2
    )
    env = env.unwrapped
    # Tunables
    method='explore'
    n_episodes = 10000
    gamma = 0.95
    alpha = 0.8
    epsilon = 1.0
    decay = 0.999
    Ne = 10
    q, stats, Nsa, policy = q_learning(
        env=env,
        method=method,
        num_episodes=n_episodes,
        discount_factor=gamma,
        alpha=alpha,
        epsilon=epsilon,
        decay=decay,
        Ne=Ne
    )
    pprint(q)
    pprint(Nsa)
    visualize_policy(policy, ENV_NAME, env.desc.shape, 'Q-learning: Exploration Function')
    value = get_state_action_value(policy)
    visualize_value(value, ENV_NAME, env.desc.shape)
    env.render()
    plotting.plot_episode_stats(stats)
    plot_epsilon_decay(epsilon, decay, n_episodes, stats)


