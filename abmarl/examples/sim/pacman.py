
import numpy as np

from abmarl.sim.gridworld.agent import MovingAgent, OrientationAgent, GridWorldAgent, GridObservingAgent, HealthAgent
from abmarl.sim.gridworld.smart import SmartGridWorldSimulation
from abmarl.sim.gridworld.actor import DriftMoveActor
from abmarl.tools.matplotlib_utils import mscatter


class PacmanAgent(MovingAgent, OrientationAgent, GridObservingAgent, HealthAgent):
    def __init__(self, **kwargs):
        super().__init__(move_range=1, view_range=100, initial_health=1, **kwargs)


class WallAgent(GridWorldAgent): pass


class FoodAgent(HealthAgent): pass


class BaddieAgent(MovingAgent, OrientationAgent, GridObservingAgent):
    def __init__(self, **kwargs):
        super().__init__(move_range=1, view_range=100, **kwargs)


class PacmanSim(SmartGridWorldSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman = self.agents['pacman']
        self.move_actor = DriftMoveActor(**kwargs)

        self.finalize()

    def step(self, action_dict, **kwargs):
        # First move the pacman and compute overlaps
        move_result = self.move_actor.process_action(self.pacman, action_dict['pacman'], **kwargs)
        if not move_result:
            self.rewards['pacman'] -= 0.1
        else:
            self.rewards['pacman'] += 0.01

        # Compute overlaps with pacman
        candidate_agents = self.grid[self.pacman.position[0], self.pacman.position[1]]
        for agent in candidate_agents.copy().values():
            if agent.id == self.pacman.id: continue
            # TODO: Actor/event handler that processes overlaps
            if isinstance(agent, FoodAgent): # Pacman eats food
                self.rewards['pacman'] += 0.1
                self.grid.remove(agent, tuple(self.pacman.position))
                agent.health = 0
            elif isinstance(agent, BaddieAgent): # Baddie eats pacman and game over
                self.rewards['pacman'] -= 1
                self.rewards[agent.id] += 1
                self.pacman.health = 0

        # Now move the baddies and compute overlaps with pacman
        for agent_id, action in action_dict.items():
            if agent_id == 'pacman': continue
            move_result = self.move_actor.process_action(self.agents[agent_id], action, **kwargs)
            if not move_result:
                self.rewards[agent_id] -= 0.1
            else:
                self.rewards[agent_id] += 0.01

        # Compute overlaps with pacman
        candidate_agents = self.grid[self.pacman.position[0], self.pacman.position[1]]
        for agent in candidate_agents.copy().values():
            if agent.id == self.pacman.id: continue
            # TODO: Actor/event handler that processes overlaps
            if isinstance(agent, BaddieAgent): # Baddie eats pacman and game over
                self.rewards['pacman'] -= 1
                self.rewards[agent.id] += 1
                self.pacman.health = 0

        # This is here because two baddies can overlap pacman at the same time,
        # and I want to reward both, so pacman is not removed until the end of the step.
        if not self.pacman.active:
            self.grid.remove(self.pacman, tuple(self.pacman.position))

    def get_done(self, agent_id, **kwargs):
        self.get_all_done(**kwargs)

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

    def render(self, fig=None, **kwargs):
        """
        Draw the state of the pacman game.

        Args:
            fig: The figure on which to draw the grid. It's important
                to provide this figure because the same figure must be used when drawing
                each state of the simulation. Otherwise, a ton of figures will pop up,
                which is very annoying.
        """
        draw_now = fig is None
        if draw_now:
            from matplotlib import pyplot as plt
            fig = plt.gcf()

        fig.clear()
        ax = fig.gca()
        ax.set_facecolor('k')

        # TODO: Update mscatter command to adjust size by agent.
        # TODO: Update sim render command to take args for background color and grid off
        # Draw the food
        food_x = [
            food.position[1] + 0.5 for food in self.agents.values()
            if isinstance(food, FoodAgent) and food.active
        ]
        food_y = [
            self.grid.rows - 0.5 - food.position[0] for food in self.agents.values()
            if isinstance(food, FoodAgent) and food.active
        ]
        food_shape = [food.render_shape for food in self.agents.values() if food.active and isinstance(food, FoodAgent)]
        food_color = [food.render_color for food in self.agents.values() if food.active and isinstance(food, FoodAgent)]
        mscatter(food_x, food_y, ax=ax, m=food_shape, s=50, facecolor=food_color)

        # Draw the rest
        agents_x = [
            agent.position[1] + 0.5 for agent in self.agents.values() if agent.active and not isinstance(agent, FoodAgent)
        ]
        agents_y = [
            self.grid.rows - 0.5 - agent.position[0]
            for agent in self.agents.values() if agent.active and not isinstance(agent, FoodAgent)
        ]
        shape = [agent.render_shape for agent in self.agents.values() if agent.active and not isinstance(agent, FoodAgent)]
        color = [agent.render_color for agent in self.agents.values() if agent.active and not isinstance(agent, FoodAgent)]
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, facecolor=color)

        if draw_now:
            plt.plot()
            plt.pause(1e-17)
