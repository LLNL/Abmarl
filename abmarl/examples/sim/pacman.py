
import numpy as np

from abmarl.sim.gridworld.agent import (
    MovingAgent, OrientationAgent, GridWorldAgent, GridObservingAgent, HealthAgent
)
from abmarl.sim.gridworld.smart import SmartGridWorldSimulation
from abmarl.sim.gridworld.actor import DriftMoveActor


class PacmanAgent(MovingAgent, OrientationAgent, GridObservingAgent, HealthAgent):
    def __init__(self, **kwargs):
        super().__init__(move_range=1, view_range=100, initial_health=1, **kwargs)


class WallAgent(GridWorldAgent): pass


class FoodAgent(HealthAgent):
    def __init__(self, **kwargs):
        super().__init__(render_size=50, initial_health=1, **kwargs)


class BaddieAgent(MovingAgent, OrientationAgent, GridObservingAgent):
    def __init__(self, **kwargs):
        super().__init__(move_range=1, view_range=100, **kwargs)


class PacmanSim(SmartGridWorldSimulation):
    """
    The Pacman Simulation is the familiar pacman arcade game.

    The simulation expects as single Pacman agent named "pacman". Food pieces should
    be "FoodAgents" and the baddies are "BaddieAgents", of which there can be multiple
    baddies. Pacman "consumes" the food as it overlaps them, and baddies "consume"
    pacman, ending the game, when it overlaps the pacman. All entities move according
    to the DriftMoveActor.

    We also assume that the grid is composed like the standard pacman grid; specifically,
    we hardcode the corridor-teleportation feature.
    """
    def __init__(self, reward_scheme=None, **kwargs):
        super().__init__(**kwargs)
        self.pacman = self.agents['pacman']
        self.move_actor = DriftMoveActor(**kwargs)
        self.reward_scheme = reward_scheme

        self.finalize()

    @property
    def reward_scheme(self):
        """
        Specify how the agents should be rewarded for different in-game events.

        Pacman support the following events: 'bad_move', 'entropy', 'eat_food',
        'kill', and 'die'. The reward scheme should be a dictionary that contains
        each of these events and maps them to a numerical value--positive for reward
        and negative for penalty.
        """
        return self._reward_scheme

    @reward_scheme.setter
    def reward_scheme(self, value):
        if value is not None:
            assert type(value) is dict, "Reward scheme must be a dictionary."
            for event, reward in value.items():
                assert event in ['bad_move', 'entropy', 'eat_food', 'kill', 'die'], \
                    "Supported events: 'bad_move', 'entropy', 'eat_food', 'kill', and 'die'."
                assert type(reward) in [int, float], f"Reward for {event} must be numerical."
            self._reward_scheme = value
        else:
            self._reward_scheme = {
                'bad_move': -0.1,
                'entropy': 0.01,
                'eat_food': 0.1,
                'kill': 1,
                'die': -1,
            }

    def step(self, action_dict, **kwargs):
        # First move the pacman and compute overlaps
        move_result = self.move_actor.process_action(self.pacman, action_dict['pacman'], **kwargs)
        if not move_result:
            self.rewards['pacman'] += self.reward_scheme['bad_move']
        else:
            self.rewards['pacman'] += self.reward_scheme['entropy']
        if np.array_equal(self.pacman.position, np.array([9, 0])):
            self.grid.remove(self.pacman, (9, 0))
            self.grid.place(self.pacman, (9, 20))
        elif np.array_equal(self.pacman.position, np.array([9, 20])):
            self.grid.remove(self.pacman, (9, 20))
            self.grid.place(self.pacman, (9, 0))

        # Compute overlaps with pacman
        candidate_agents = self.grid[self.pacman.position[0], self.pacman.position[1]]
        for agent in candidate_agents.copy().values():
            if agent.id == self.pacman.id: continue
            if isinstance(agent, FoodAgent): # Pacman eats food
                self.rewards['pacman'] += self.reward_scheme['eat_food']
                self.grid.remove(agent, tuple(self.pacman.position))
                agent.health = 0
            elif isinstance(agent, BaddieAgent): # Baddie eats pacman and game over
                self.rewards['pacman'] += self.reward_scheme['die']
                self.rewards[agent.id] += self.reward_scheme['kill']
                self.pacman.health = 0

        # Now move the baddies and compute overlaps with pacman
        for agent_id, action in action_dict.items():
            if agent_id == 'pacman': continue
            agent = self.agents[agent_id]
            move_result = self.move_actor.process_action(agent, action, **kwargs)
            if not move_result:
                self.rewards[agent_id] += self.reward_scheme['bad_move']
            else:
                self.rewards[agent_id] += self.reward_scheme['entropy']
            if np.array_equal(agent.position, np.array([9, 0])):
                self.grid.remove(agent, (9, 0))
                self.grid.place(agent, (9, 20))
            elif np.array_equal(agent.position, np.array([9, 20])):
                self.grid.remove(agent, (9, 20))
                self.grid.place(agent, (9, 0))

        # Compute overlaps with pacman
        candidate_agents = self.grid[self.pacman.position[0], self.pacman.position[1]]
        for agent in candidate_agents.copy().values():
            if agent.id == self.pacman.id: continue
            if isinstance(agent, BaddieAgent): # Baddie eats pacman and game over
                self.rewards['pacman'] += self.reward_scheme['die']
                self.rewards[agent.id] += self.reward_scheme['kill']
                self.pacman.health = 0

        # This is here because two baddies can overlap pacman at the same time,
        # and I want to reward both, so pacman is not removed until the end of the step.
        if not self.pacman.active:
            self.grid.remove(self.pacman, tuple(self.pacman.position))

    def get_done(self, agent_id, **kwargs):
        return self.get_all_done(**kwargs)

    def get_all_done(self, **kwargs):
        """
        Pacman is done if it dies or if all the food is gone.
        """
        if not self.pacman.active:
            return True
        else:
            for agent in self.agents.values():
                if isinstance(agent, FoodAgent):
                    return False # There is still food left
            # No food left
            return True

    def render(self, **kwargs):
        """
        Draw the state of the pacman game.
        """
        for agent in self.agents.values():
            if isinstance(agent, OrientationAgent):
                agent.render_shape = {
                    1: '<',
                    2: 'v',
                    3: '>',
                    4: '^'
                }[agent.orientation]
        super().render(
            gridlines=False,
            background_color='k',
            **kwargs
        )
