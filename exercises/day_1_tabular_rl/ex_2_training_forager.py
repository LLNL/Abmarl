# ------------------------------------- #
# --- DSSI 2021 RL Class Exercise 2 --- #
# ------------------------------------- #

# In this exercise, we will use our Q-learning method to train an RL agent to
# forage a grid. We will initialize the simulation, specify the training parameters,
# and connect the simulation with the trainer. We will run a training experiment
# and then visualize the results and output some training statistics.

import pickle

from abmarl.sim.examples import ForagingSim
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.managers import AllStepManager

# Setup the simulation
sim = AllStepManager(RavelDiscreteWrapper(ForagingSim()))

# Setup the trainer
from ex_1_q_learning import q_learning
# NOTE: You can use Abmarl's Q-learner if you did not finish exercise 1
# from abmarl.algs.q_learning import q_learning

# Trainig parameters:
iterations = 1000
gamma = 0.95
alpha = 0.1
epsilon = 0.1
horizon = 200

# Train the agent
sim, q_table, policy = q_learning(sim, iterations, gamma, alpha, epsilon, horizon)

# Dump the trained policy and simulation
with open('trained_forager.pkl', 'wb') as fp:
    pickle.dump({'sim': sim, 'policy': policy}, fp)
