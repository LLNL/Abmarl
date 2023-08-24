
import numpy as np

from abmarl.sim import Agent
from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent
from abmarl.sim.gridworld.state import MazePlacementState
from abmarl.sim.gridworld.actor import MoveActor
from abmarl.sim.gridworld.observer import PositionCenteredEncodingObserver


class MultiMazeNavigationAgent(GridObservingAgent, MovingAgent):
    def __init__(self, **kwargs):
        super().__init__(move_range=1, **kwargs)


class MultiMazeNavigationSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # State Components
        self.position_state = MazePlacementState(**kwargs)

        # Action Components
        self.move_actor = MoveActor(**kwargs)

        # Observation Components
        self.grid_observer = PositionCenteredEncodingObserver(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)

        # Track the rewards
        self.reward = {
            agent.id: 0 for agent in self.agents.values() if isinstance(agent, Agent)
        }

    def step(self, action_dict, **kwargs):
        # Process moves
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            move_result = self.move_actor.process_action(agent, action, **kwargs)
            if not move_result:
                self.reward[agent_id] -= 0.1

            # Entropy penalty
            self.reward[agent_id] -= 0.01

    def get_obs(self, agent_id, **kwargs):
        return {
            **self.grid_observer.get_obs(self.agents[agent_id], **kwargs)
        }

    def get_reward(self, agent_id, **kwargs):
        reward = 1 if self.get_done(agent_id) else self.reward[agent_id]
        self.reward[agent_id] = 0
        return reward

    def get_done(self, agent_id, **kwargs):
        return np.array_equal(
            self.agents[agent_id].position, self.position_state.target_agent.position
        )

    def get_all_done(self, **kwargs):
        return all([
            self.get_done(agent.id)
            for agent in self.agents.values()
            if isinstance(agent, MultiMazeNavigationAgent)
        ])

    def get_info(self, agent_id, **kwargs):
        return {}
