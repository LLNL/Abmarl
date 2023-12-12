
from abmarl.sim.gridworld.agent import MovingAgent, OrientationAgent, GridWorldAgent, GridObservingAgent
from abmarl.sim.gridworld.smart import SmartGridWorldSimulation
from abmarl.sim.gridworld.actor import DriftMoveActor


class PacmanAgent(MovingAgent, OrientationAgent, GridObservingAgent):
    def __init__(self, **kwargs):
        super().__init__(move_range=1, view_range=100, **kwargs)


class WallAgent(GridWorldAgent): pass


class FoodAgent(GridWorldAgent): pass


class PacmanSim(SmartGridWorldSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman = self.agents['pacman']
        self.move_actor = DriftMoveActor(**kwargs)

        self.finalize()

    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            move_result = self.move_actor.process_action(self.agents[agent_id], action, **kwargs)
            if not move_result:
                self.rewards[agent_id] -= 0.1
            else:
                self.rewards[agent_id] += 0.01

        # Compute overlaps with pacman
        candidate_agents = self.grid[self.pacman.position[0], self.pacman.position[1]]
        for agent in candidate_agents.copy().values():
            if agent.id == self.pacman.id: continue
            if isinstance(agent, FoodAgent): # Pacman eats food
                self.rewards['pacman'] += 0.1
                self.grid.remove(agent, tuple(self.pacman.position))
                agent.active = False
