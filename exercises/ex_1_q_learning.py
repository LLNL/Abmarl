# ------------------------------------- #
# --- DSSI 2021 RL Class Exercise 1 --- #
# ------------------------------------- #

# In this exercise, we will write our own Q-learning method using the algorithm
# defined in our class. The skeleton of the function is already in place for us;
# we just need to fill out the details where we see TODO.

from gym.spaces import Discrete
import numpy as np

from abmarl.external import GymWrapper
from abmarl.managers import SimulationManager
from abmarl.pols import EpsilonSoftPolicy

def q_learning(sim, iterations=10_000, gamma=0.95, alpha=0.1, epsilon=0.1, horizon=200):
    """
    Implementation of the Q-Learning algorithm as found in the 2021 DSSI RL class.
    The update is like so
        Q(s,u) <- Q(s,u) + alpha * (reward + gamma * max(Q(s',*)) - Q(s,u))
    where s' is the next state and the max is over all possible actions. The policy
    is an epsilon-greedy policy.

    Args:
        iterations: The number of training episodes.
        gamma: The reward discount factor.
        alpha: The learning rate.
        epsilon: The exploration rate.
        horizon: The maximum number of steps per episode.
    
    Returns:
        sim: The simulation.
        q_table: The Q values that we train.
        policy: The policy that is learned.
    """
    # Setup the simulation
    assert isinstance(sim, SimulationManager), "The simulation must be a SimulationManager"
    sim = GymWrapper(sim)
    assert isinstance(sim.observation_space, Discrete), \
        "The Simulation must be wrapped with RavelDiscreteWrapper"
    assert isinstance(sim.action_space, Discrete), \
        "The Simulation must be wrapped with RavelDiscreteWrapper"

    # Setup the policy
    # TODO: Create the q_table
    policy = EpsilonSoftPolicy(q_table, epsilon=epsilon)

    # Begin simulations
    for i in range(iterations):
        print(f"Episode {i}")
        # TODO: Reset the simulation and grab the observation
        for _ in range(horizon):
            action = policy.act(obs)
            # TODO: Step the simulation and grab the output
            # TODO: Update the q_table according to the update rule
            obs = next_obs
            if done:
                break

    return sim, q_table, policy
