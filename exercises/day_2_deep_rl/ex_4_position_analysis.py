# ------------------------------------- #
# --- DSSI 2021 RL Class Exercise 4 --- #
# ------------------------------------- #

# We now visualize and analyze the trained policy.

# Policies can be visualized with
# abmarl visualize ~/abmarl_results/Forager-DATE-TIME/

# Policies can be analyzed with
# abmarl anaylze ~/abmarl_results/Forager-DATE-TIME/ ex_4_position_analysis.py

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from abmarl.external import GymWrapper

def run(sim, trainer):
    region = sim.sim.position_state.region
    grid = np.zeros((region, region))
    forager = sim.agents['forager']
    sim = GymWrapper(sim)
    for episode in range(100):
        print('Episode: {}'.format(episode))
        obs = sim.reset()
        pox, poy = forager.position
        grid[pox, poy] += 1
        for j in range(200): # Run until the episode ends or max 200 steps
            # Get actions from policies
            action = trainer.compute_action(obs)
            # Step the simulation
            obs, reward, done, info = sim.step(action)
            pox, poy = forager.position
            grid[pox, poy] += 1
            if done:
                break

    plt.figure(1)
    plt.title("Position concentration")
    sns.heatmap(np.flipud(np.transpose(grid)), linewidth=0.5)
    plt.show()
