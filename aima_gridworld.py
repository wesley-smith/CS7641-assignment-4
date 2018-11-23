from aima_mdp import value_iteration, best_policy, GridMDP, policy_iteration
from aima_utils import print_table

# Row, col indices start at 0 starting from lower left
# Cols are specified first (e.g. (col, row))

# Define some environments from Open AI Gym
frozen_lake_v0 = GridMDP(
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, +1]],
    terminals=[
        (0, 0), (1, 2), (3, 2), (3, 3), # Holes
        (3, 0) # Goal
    ]) 
# Adjust for negative rewards
frozen_lake_v0_adj_rew = GridMDP(
    [[-0.04, -0.04, -0.04, -0.04],
     [-0.04, -1, -0.04, -1],
     [-0.04, -0.04, -0.04, -1],
     [-1, -0.04, -0.04, +1]],
    terminals=[
        (0, 0), (1, 2), (3, 2), (3, 3), # Holes
        (3, 0) # Goal
    ])
env = frozen_lake_v0_adj_rew
# pi = best_policy(env, value_iteration(env, .01))
# print_table(env.to_arrows(pi))
pi = best_policy(env, policy_iteration(env))
print_table(env.to_arrows(pi))