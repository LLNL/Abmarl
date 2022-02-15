# 5.1 Monte Carlo prediction p92.
from gym.spaces import Discrete
import numpy as np

from abmarl.managers import SimulationManager
from abmarl.external import GymWrapper
from abmarl.pols.policy import _GreedyPolicy, _EpsilonSoftPolicy

from .generate_episode import generate_episode


def off_policy(sim, iteration=10_000, gamma=0.9, horizon=200):
    """
    Off-policy Monte Carlo control estimates an optimal policy in a simulation. Trains a greedy
    policy be generating trajectories from an epsilon-soft behavior policy.

    Args:
        sim: The simulation, obviously.
        iteration: The number of times to iterate the learning algorithm.
        gamme: The discount factor

    Returns:
        sim: The simulation. Algorithms may wrap simulations before training in them, so this
            simulation may be wrapped.
        q_table: The Q values
        policy: The policy that is learned.
    """
    assert isinstance(sim, SimulationManager)
    sim = GymWrapper(sim)
    assert isinstance(sim.observation_space, Discrete)
    assert isinstance(sim.action_space, Discrete)
    q_table = np.random.normal(0, 1, size=(sim.observation_space.n, sim.action_space.n))
    c_table = 0 * q_table
    policy = _GreedyPolicy(q_table)
    for i in range(iteration):
        behavior_policy = _EpsilonSoftPolicy(q_table)
        states, actions, rewards, = generate_episode(sim, behavior_policy, horizon)
        G = 0
        W = 1
        for i in reversed(range(len(states))):
            state, action, reward = states[i], actions[i], rewards[i]
            G = gamma * G + reward
            c_table[state, action] += W
            q_table[state, action] = q_table[state, action] + W/(c_table[state, action]) * \
                (G - q_table[state, action])
            if action != policy.act(state): # Nonoptimal action
                break
            W /= behavior_policy.probability(state, action)

    return sim, q_table, policy
