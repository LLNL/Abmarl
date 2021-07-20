# ------------------------------------- #
# --- DSSI 2021 RL Class Exercise 3 --- #
# ------------------------------------- #

# In this exercise, we will examine the observation and action space used in our
# Foraging environment. We will play with the number of food resources, the
# foragers view range, the forager's attack range, and the size of the grid.

# Import the simulation environment and the agents
from abmarl.sim.examples import DeepForagingSim, Forager
from abmarl.managers import AllStepManager

# Instatiate the forager
AGENT_VIEW = 5 # TODO: Play around with this value
ATTACK_RANGE = 1 # TODO: Play around with this value
agents = {'forager': Forager(
    id='forager',
    agent_view=AGENT_VIEW,
    attack_range=ATTACK_RANGE,
)}

# Instatiate the simulation
REGION = 20 # TODO: Play around with this value
NUM_FOOD = 12 # TODO: PLay around with this value
sim = DeepForagingSim(
    NUM_FOOD,
    region=REGION,
    agents=agents
)
sim = AllStepManager(sim)

# Reset the simulation and examine the observation
# Notice that the food spawns at random locations every time
from matplotlib import pyplot as plt

fig = plt.figure()
for i in range(5): # 5 Resets
    obs = sim.reset()
    sim.render(fig=fig)
    plt.pause(1)
print(obs['forager']['position'])
plt.show()
# Before closing the window, notice how the agent's observation coorelates to
# nearby food in the grid.

# Run the simulation
# The agent will not move and will only attack. As you increaes with the forager's
# attack_range, notice that it can harvest food further from itself. If the attack
# range is less than agent_view, then the forager may be able to see food, but
# not harvest it.
fig = plt.figure()
for i in range(5):
    print(f"Episode {i}")
    sim.reset()
    sim.render(fig=fig)
    for _ in range(ATTACK_RANGE):
        action = {'forager': {'attack': True}}
        obs, _, done, _ = sim.step(action)
        print(); print(obs['forager']['position'])
        sim.render(fig=fig)
        plt.pause(2)
        if done['__all__']:
            break

# TODO: Consider this
# When the forager chooses to harvest, the simulation determines if there is nearby
# food, and if so the foraging is successful. So the attack range does not change
# the forager's action space; it only changes the impact on the simulation.
# Consider the other parameters we have played with. Which of these paraemters
# would require us to redo the policy training if we changed it:
# agent_view?
# num_food?
# region?
