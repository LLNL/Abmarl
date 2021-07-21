# ------------------------------------- #
# --- DSSI 2021 RL Class Exercise 5 --- #
# ------------------------------------- #

# In this exercise, we will conifgure homogeneous agents in the Hunting and Forager
# simuation.

# Import the simulation environment and the agents
from abmarl.sim.examples.forager_hunter import HuntingForagingSim, Forager, Hunter
from abmarl.managers import AllStepManager

# Instatiate the foragers and hunters
NUM_FORAGERS = 5 # TODO: Play with the number of foragers
foragers = {f'forager{i}': Forager(
    id=f'forager{i}',
    agent_view=3,
    attack_range=1,
) for i in range(NUM_FORAGERS)}

NUM_HUNTERS = 2 # TODO: Play with the number of hunters
hunters = {f'hunter{i}': Hunter(
    id=f'hunter{i}',
    agent_view=2,
    attack_range=1,
) for i in range(NUM_HUNTERS)}

agents = {**hunters, **foragers}

# Instatiate the simulation
REGION = 20 # TODO: Play with the grid size
NUM_FOOD = 12 # TODO: Play with the number of food
sim = HuntingForagingSim(
    NUM_FOOD,
    region=REGION,
    agents=agents
)
sim = AllStepManager(sim)

# Run the simulation
# Notice the interaction between the hunters, foragers, and food.
from matplotlib import pyplot as plt
fig = plt.figure()
for i in range(3):
    print(f"Episode {i}")
    sim.reset()
    done_agents = set()
    sim.render(fig=fig)
    for _ in range(50):
        action = {
            agent.id: agent.action_space.sample()
            for agent in agents.values() if agent.id not in done_agents
        }
        _, _, done, _ = sim.step(action)
        sim.render(fig=fig)
        for agent_id, value in done.items():
            if value:
                done_agents.add(agent_id)
        if done['__all__']:
            break
