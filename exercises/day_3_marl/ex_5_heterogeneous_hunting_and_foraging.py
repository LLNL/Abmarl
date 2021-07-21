# ------------------------------------- #
# --- DSSI 2021 RL Class Exercise 5 --- #
# ------------------------------------- #

# In this exercise, we will configure heterogeneous agents in the Hunting and Forager
# simuation.

from abmarl.sim.examples.forager_hunter import HuntingForagingSim, Forager, Hunter
from abmarl.managers import AllStepManager

# Intatiate custom foragers
foragers = {
    f'forager0': Forager(id=f'forager0', agent_view=0, attack_range=0),
    f'forager1': Forager(id=f'forager1', agent_view=1, attack_range=2),
    f'forager2': Forager(id=f'forager2', agent_view=1, attack_range=0),
    f'forager3': Forager(id=f'forager3', agent_view=2, attack_range=1),
    f'forager4': Forager(id=f'forager4', agent_view=3, attack_range=0),
    # ... Add as many as you'd like
}

# Intatiate custom hunters
hunters = {
    f'hunter0': Hunter(id=f'hunter0', agent_view=2, attack_range=1),
    f'hunter1': Hunter(id=f'hunter1', agent_view=3, attack_range=1),
    f'hunter2': Hunter(id=f'hunter2', agent_view=1, attack_range=3),
    # ... Add as many as you'd like
}

agents = {**hunters, **foragers}

# Instatiate the simulation
REGION = 20
NUM_FOOD = 12
sim = HuntingForagingSim(
    NUM_FOOD,
    region=REGION,
    agents=agents
)
sim = AllStepManager(sim)

# Run the simulation
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

# We cannot really see the impact of the heterogeneity on in the random simulation.
# We have to train the agents so that they can captialize on the hetereogeneity.
