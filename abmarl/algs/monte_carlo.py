# 5.1 Monte Carlo prediction p92.
from gym.spaces import Discrete
import numpy as np

from abmarl.managers import SimulationManager
from abmarl.external import GymWrapper
from abmarl.pols import GreedyPolicy, EpsilonSoftPolicy, RandomFirstActionPolicy
from abmarl.tools import numpy_utils as npu

from .generate_episode import generate_episode


def exploring_starts(sim, iteration=10_000, gamma=0.9, horizon=200):
    """
    Estimate an optimal policy over an simulation using monte carlo policy estimation.

    Args:
        sim: The simulation, obviously.
        iteration: The number of times to iterate the learning algorithm.
        gamma: The discount factor
        horizon: the time horizon for the trajectory.

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
    policy = RandomFirstActionPolicy(q_table)
    state_action_returns = {}

    for i in range(iteration):
        states, actions, rewards = generate_episode(sim, policy, horizon)
        states = np.stack(states)
        actions = np.stack(actions)
        G = 0
        for i in reversed(range(len(states))):
            state, action, reward = states[i], actions[i], rewards[i]
            G = gamma * G + reward
            if not (npu.array_in_array(state, states[:i]) and
                    npu.array_in_array(action, actions[:i])):
                if (state, action) not in state_action_returns:
                    state_action_returns[(state, action)] = [G]
                else:
                    state_action_returns[(state, action)].append(G)
                q_table[state, action] = np.mean(state_action_returns[(state, action)])

    return sim, q_table, policy


def epsilon_soft(sim, iteration=10_000, gamma=0.9, epsilon=0.1, horizon=200):
    """
    Estimate an optimal policy over a simulation using monte carlo policy estimation. The policy
    is technically non-optimal because it is epsilon-soft.

    Args:
        sim: The simulation, obviously.
        iteration: The number of times to iterate the learning algorithm.
        gamme: The discount factor
        epsilon: The exploration probability.

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
    policy = EpsilonSoftPolicy(q_table, epsilon=epsilon)
    state_action_returns = {}

    for i in range(iteration):
        states, actions, rewards = generate_episode(sim, policy, horizon)
        states = np.stack(states)
        actions = np.stack(actions)
        G = 0
        for i in reversed(range(len(states))):
            state, action, reward = states[i], actions[i], rewards[i]
            G = gamma * G + reward
            if not (npu.array_in_array(state, states[:i]) and
                    npu.array_in_array(action, actions[:i])):
                if (state, action) not in state_action_returns:
                    state_action_returns[(state, action)] = [G]
                else:
                    state_action_returns[(state, action)].append(G)
                q_table[state, action] = np.mean(state_action_returns[(state, action)])

    return sim, q_table, policy


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
    policy = GreedyPolicy(q_table)
    for i in range(iteration):
        behavior_policy = EpsilonSoftPolicy(q_table)
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
