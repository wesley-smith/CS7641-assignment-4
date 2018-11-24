import itertools
import numpy as np
import sys

from collections import defaultdict

import dennybritz_plotting as plotting
from aima_utils import argmax


def make_epsilon_greedy_policy(Q, epsilon, decay, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation, episode):
        e_prime = epsilon * decay ** episode
        A = np.ones(nA, dtype=float) * e_prime / nA
        if np.all(np.isclose(Q[observation], np.zeros(nA))):
            best_action = np.random.randint(nA)
        else:
            best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - e_prime)
        return A

    return policy_fn

def make_exploration_function(Rplus, Ne):
    """
    Creates an "exploratory" policy (Exploration Function from AIMA chapter 21) based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        Nsa: A dictionary that maps state -> number of times an action has been taken
        Rplus: Large reward value to assign before iteration limit
        Ne: Minimum number of times that each action will be taken at each state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def exploration_fn(u, n):
        if n < Ne:
            return Rplus
        else:
            return u
    return np.vectorize(exploration_fn)


def q_learning(env, method, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, decay=1.0, Rplus=None, Ne=None):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        method: ['greedy', 'explore'] whether to use a greedy or an explorative policy
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        decay: exponential decay rate for epsilon
        Rplus: Optimistic reward given to unexplored states
        Ne: Minimum number of times that each action will be taken at each state

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    # Keeps track of how many times we've taken action a in state s
    Nsa = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    if method == 'greedy':
        policy = make_epsilon_greedy_policy(Q, epsilon, decay, env.action_space.n)
        def get_next_action(state_, episode):
            action_probs = policy(state_, episode)
            return np.random.choice(np.arange(len(action_probs)), p=action_probs)
    elif method == 'explore':
        if not Rplus:
            Rplus = max(env.reward_range)
        if not Ne:
            Ne = 100
        exploration_fn = make_exploration_function(Rplus, Ne)
        done_exploring = False
        def get_next_action(state_, episode):
            exploration_values = exploration_fn(Q[state_], Nsa[state_])
            if np.allclose(exploration_values, exploration_values[0]):
                return np.random.randint(env.nA)
            else:
                return np.argmax(exploration_values)
    else:
        raise ValueError('Unsupported method type')

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Get an action based on the exploration function
            action = get_next_action(state, i_episode)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            Nsa[state][action] += 1

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if method == 'explore' and not done_exploring:
                for arr in Nsa.values():
                    if not np.all(arr >= Ne):
                        break
                else:
                    done_exploring = True
                    print('All done with exploration at episode %i, step %i' % (i_episode, t))
            if done:
                break

            state = next_state

    final_policy = np.zeros((env.nS, env.nA))
    for state in range(env.nS):
        final_policy[state] = Q[state]
    return Q, stats, Nsa, final_policy