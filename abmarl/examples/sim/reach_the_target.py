
import numpy as np

from abmarl.sim.agent_based_simulation import Agent
from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridWorldAgent, MovingAgent, AttackingAgent, \
    GridObservingAgent, HealthAgent
from abmarl.sim.gridworld.state import PositionState, HealthState
from abmarl.sim.gridworld.actor import SelectiveAttackActor, MoveActor
from abmarl.sim.gridworld.observer import SingleGridObserver
from abmarl.sim.gridworld.done import ActiveDone, DoneBaseComponent


class TargetDone(ActiveDone):
    """
    Agent is done if overlaps with a target agent.
    """
    def __init__(self, target=None, **kwargs):
        super().__init__(**kwargs)
        self.target = target

    @property
    def target(self):
        """
        The target agent.
        """
        return self._target

    @target.setter
    def target(self, value):
        assert value in self.agents.values(), "Target must be an agent."
        self._target = value

    def get_done(self, agent, **kwargs):
        """
        Return True if the agent overlaps with the target. Otherwise, return False.
        """
        if agent == self.target:
            return False
        else:
            return np.array_equal(agent.position, self.target.position)


class OnlyAgentLeftDone(DoneBaseComponent):
    """
    Agent and simulation is done when there is only one agent remaining.
    """
    def _agents_remaining(self):
        # Return the number of active Agents remaining in the game.
        return sum([
            1 for agent in self.agents.values() if agent.active and isinstance(agent, Agent)
        ])

    def get_done(self, agent_id, **kwargs):
        return self._agents_remaining() <= 1

    def get_all_done(self, **kwargs):
        return self._agents_remaining() <= 1


class BarrierAgent(GridWorldAgent):
    def __init__(self, **kwargs):
        super().__init__(
            encoding=1,
            blocking=True,
            render_shape='s',
            **kwargs
        )


class TargetAgent(AttackingAgent, GridObservingAgent):
    def __init__(self, **kwargs):
        super().__init__(
            id='target',
            encoding=2,
            render_color='g',
            **kwargs
        )


class RunningAgent(MovingAgent, GridObservingAgent, HealthAgent):
    def __init__(self, **kwargs):
        super().__init__(
            encoding=3,
            render_color='b',
            **kwargs
        )


class ReachTheTargetSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.grid = kwargs['grid']
        self.target = self.agents['target']

        # State Components
        self.position_state = PositionState(**kwargs)
        self.health_state = HealthState(**kwargs)

        # Action Components
        self.move_actor = MoveActor(**kwargs)
        self.attack_actor = SelectiveAttackActor(**kwargs)

        # Observer Components
        self.grid_observer = SingleGridObserver(**kwargs)

        # Done components
        self.active_done = ActiveDone(**kwargs)
        self.target_done = TargetDone(target=self.target, **kwargs)
        self.only_agent_done = OnlyAgentLeftDone(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.health_state.reset(**kwargs)
        self.position_state.reset(**kwargs)

        self.rewards = {agent.id: 0 for agent in self.agents.values() if isinstance(agent, Agent)}

    def step(self, action_dict, **kwargs):
        # Process the attacks
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.active:
                attack_status, attacked_agents = \
                    self.attack_actor.process_action(agent, action, **kwargs)
                if attack_status: # Attack was attempted
                    if not attacked_agents: # Attack failed
                        self.rewards[agent_id] -= 0.1
                    else:
                        for attacked_agent in attacked_agents:
                            if not attacked_agent.active: # Agent has died
                                self.rewards[attacked_agent.id] -= 1
                                self.rewards[agent_id] += 1

        # Process the moves
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if isinstance(agent, MovingAgent):
                if agent.active:
                    move_result = self.move_actor.process_action(agent, action, **kwargs)
                    if not move_result:
                        self.rewards[agent_id] -= 0.1
                if self.target_done.get_done(agent):
                    self.rewards[agent_id] += 1
                    self.grid.remove(agent, agent.position)
                    agent.active = False

        # Entropy penalty for the runners
        for agent_id in action_dict:
            if isinstance(self.agents[agent_id], RunningAgent):
                self.rewards[agent_id] -= 0.01

    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            **self.grid_observer.get_obs(agent, **kwargs)
        }

    def get_reward(self, agent_id, **kwargs):
        reward = self.rewards[agent_id]
        self.rewards[agent_id] = 0
        return reward

    def get_done(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        if isinstance(agent, RunningAgent):
            return self.active_done.get_done(agent, **kwargs) \
                or self.target_done.get_done(agent, **kwargs)
        elif isinstance(agent, TargetAgent):
            return self.only_agent_done.get_done(agent, **kwargs)

    def get_all_done(self, **kwargs):
        return self.only_agent_done.get_all_done(**kwargs)

    def get_info(self, agent_id, **kwargs):
        return {}
