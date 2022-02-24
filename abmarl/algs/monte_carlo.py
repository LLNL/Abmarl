# 5.1 Monte Carlo prediction p92.
from gym.spaces import Discrete
import numpy as np

from abmarl.managers import SimulationManager
from abmarl.external import GymWrapper
from abmarl.policies.q_table_policy import EpsilonSoftPolicy


def generate_episode(sim, policy, horizon=200):
    """
    Generate an episode from a policy acting on an simulation.

    Returns: sequence of state, action, reward.
    """
    obs = sim.reset()
    policy.reset() # Reset the policy too so that it knows its the beginning of the episode.
    states, actions, rewards = [], [], []
    states.append(obs)
    for _ in range(horizon):
        action = policy.compute_action(obs)
        obs, reward, done, _ = sim.step(action)
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        if done:
            break

    states.pop() # Pop off the terminating state
    return states, actions, rewards


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
    policy = EpsilonSoftPolicy(
        observation_space=sim.observation_space,
        action_space=sim.action_space,
        q_table=q_table
    )
    for i in range(iteration):
        behavior_policy = EpsilonSoftPolicy(
            observation_space=policy.observation_space,
            action_space=policy.action_space,
            q_table=policy.q_table
        )
        states, actions, rewards, = generate_episode(sim, behavior_policy, horizon)
        G = 0
        W = 1
        for i in reversed(range(len(states))):
            state, action, reward = states[i], actions[i], rewards[i]
            G = gamma * G + reward
            c_table[state, action] += W
            q_table[state, action] = q_table[state, action] + W/(c_table[state, action]) * \
                (G - q_table[state, action])
            if action != policy.compute_action(state): # Nonoptimal action
                break
            W /= behavior_policy.probability(state, action)

    return sim, q_table, policy
