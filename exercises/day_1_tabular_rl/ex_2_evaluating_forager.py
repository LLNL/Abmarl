# ------------------------------------- #
# --- DSSI 2021 RL Class Exercise 2 --- #
# ------------------------------------- #

# We now visualize and evaluate the trained policy. We'll simply load the simulation
# and policy from the pickle file and run it through a number of episodes. Here,
# we turn off exploration in the policy.

import pickle

from matplotlib import pyplot as plt
import numpy as np

# Load the simulation and trained policy
with open('trained_forager.pkl', 'rb') as fp:
    data = pickle.load(fp)
sim = data['sim']
policy = data['policy']
policy.epsilon = 0

# Evaluate the simulation
steps_before_done = [200] * 10
fig = plt.figure()
for i in range(10):
    print(f"Episode {i}")
    obs = sim.reset()
    sim.render(fig=fig)
    for j in range(200):
        obs, _, done, _ = sim.step(policy.act(obs))
        sim.render(fig=fig)
        if done:
            steps_before_done[i] = j
            break

print(f"Average number of steps to forage all resources: {np.mean(steps_before_done)}")
