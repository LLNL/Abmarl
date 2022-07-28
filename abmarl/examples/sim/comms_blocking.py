
from gym.spaces import Discrete, Dict, Box
import numpy as np
from matplotlib import pyplot as plt

from abmarl.sim import Agent
from abmarl.sim.gridworld.agent import MovingAgent, GridObservingAgent, GridWorldAgent
from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.state import PositionState, StateBaseComponent
from abmarl.sim.gridworld.actor import MoveActor, ActorBaseComponent
from abmarl.sim.gridworld.observer import SingleGridObserver, ObserverBaseComponent
from abmarl.sim.gridworld.done import DoneBaseComponent
import abmarl.sim.gridworld.utils as gu


class BroadcastingAgent(Agent, GridWorldAgent):
    def __init__(self, broadcast_range=None, initial_message=None, **kwargs):
        super().__init__(**kwargs)
        self.broadcast_range = broadcast_range
        self.initial_message = initial_message

    @property
    def broadcast_range(self):
        return self._broadcast_range

    @broadcast_range.setter
    def broadcast_range(self, value):
        assert type(value) is int and value >= 0, "Broadcast Range must be a nonnegative integer."
        self._broadcast_range = value

    @property
    def initial_message(self):
        return self._initial_message

    @initial_message.setter
    def initial_message(self, value):
        if value is not None:
            assert -1 <= value <= 1, "Initial message must be a number between -1 and 1."
        self._initial_message = value

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, value):
        self._message = min(max(value, -1), 1)

    @property
    def configured(self):
        return super().configured and self.broadcast_range is not None


class BroadcastingActor(ActorBaseComponent):
    """
    Process sending and receiving messages between agents.

    Broadcasting Agents can broadcast to agents within their range according to
    the broadcast mapping and if the agent is not blocked.
    """
    def __init__(self, broadcast_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.broadcast_mapping = broadcast_mapping
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.action_space[self.key] = Discrete(2)

    @property
    def key(self):
        return 'broadcast'

    @property
    def supported_agent_type(self):
        return BroadcastingAgent

    @property
    def broadcast_mapping(self):
        """
        Dict that dictates to which agents the broadcasting agent can broadcast.

        The dictionary maps the broadcasting agents' encodings to a list of encodings
        to which they can broadcast. For example, the folowing broadcast_mapping:
        {
            1: [3, 4, 5],
            3: [2, 3],
        }
        means that agents whose encoding is 1 can broadcast other agents whose encodings
        are 3, 4, or 5; and agents whose encoding is 3 can broadcast other agents whose
        encodings are 2 or 3.
        """
        return self._broadcast_mapping

    @broadcast_mapping.setter
    def broadcast_mapping(self, value):
        assert type(value) is dict, "Broadcast mapping must be dictionary."
        for k, v in value.items():
            assert type(k) is int, "All keys in broadcast mapping must be integer."
            assert type(v) is list, "All values in broadcast mapping must be list."
            for i in v:
                assert type(i) is int, \
                    "All elements in the broadcast mapping values must be integers."
        self._broadcast_mapping = value

    def process_action(self, broadcasting_agent, action_dict, **kwargs):
        """
        If the agent has chosen to broadcast, then we process their broadcast.

        The processing goes through a series of checks. The broadcast is successful
        if there is a receiving agent such that:
        1. The receiving agent is within range.
        2. The receiving agent is valid according to the broadcast_mapping.
        3. The receiving agent is observable by the broadcasting agent.

        If the broadcast is successful, then the receiving agent receives the message
        in its observation.
        """
        def determine_broadcast(agent):
            # Generate local grid and a broadcast mask.
            local_grid, mask = gu.create_grid_and_mask(
                agent, self.grid, agent.broadcast_range, self.agents
            )

            # Randomly scan the local grid for receiving agents.
            receiving_agents = []
            for r in range(2 * agent.broadcast_range + 1):
                for c in range(2 * agent.broadcast_range + 1):
                    if mask[r, c]: # We can see this cell
                        candidate_agents = local_grid[r, c]
                        if candidate_agents is not None:
                            for other in candidate_agents.values():
                                if other.id == agent.id: # Cannot broadcast to yourself
                                    continue
                                elif other.encoding not in self.broadcast_mapping[agent.encoding]:
                                    # Cannot broadcast to this type of agent
                                    continue
                                else:
                                    receiving_agents.append(other)
            return receiving_agents

        if isinstance(broadcasting_agent, self.supported_agent_type):
            action = action_dict[self.key]
            if action: # Agent has chosen to attack
                return determine_broadcast(broadcasting_agent)


class BroadcastingState(StateBaseComponent):
    def reset(self, **kwargs):
        for agent in self.agents.values():
            if isinstance(agent, BroadcastingAgent):
                if agent.initial_message is not None:
                    agent.message = agent.initial_message
                else:
                    agent.message = np.random.uniform(-1, 1)

        # Tracks agents receiving messages from other agents
        self.receiving_state = {
            agent.id: [] for agent in self.agents.values() if isinstance(agent, BroadcastingAgent)
        }

    def update_receipients(self, from_agent, to_agents):
        for agent in to_agents:
            self.receiving_state[agent.id].append((from_agent.id, from_agent.message))

    def update_message_and_reset_receiving(self, agent):
        receiving_from = self.receiving_state[agent.id]
        self.receiving_state[agent.id] = []

        messages = [message for _, message in receiving_from]
        messages.append(agent.message)
        agent.message = np.average(messages)

        return receiving_from


class BroadcastObserver(ObserverBaseComponent):
    def __init__(self, broadcasting_state=None, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(broadcasting_state, BroadcastingState), \
            "broadcasting_state must be an instance of BroadcastingState"
        self._broadcasting_state = broadcasting_state

        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.observation_space[self.key] = Dict({
                    other.id: Box(-1, 1, (1,))
                    for other in self.agents.values()
                    if isinstance(other, self.supported_agent_type)
                })

    @property
    def key(self):
        return 'message'

    @property
    def supported_agent_type(self):
        return BroadcastingAgent

    def get_obs(self, agent, **kwargs):
        if not isinstance(agent, self.supported_agent_type):
            return {}

        obs = {other: 0 for other in agent.observation_space[self.key]}
        receive_from = self._broadcasting_state.update_message_and_reset_receiving(agent)
        for agent_id, message in receive_from:
            obs[agent_id] = message
        obs[agent.id] = agent.message
        return obs


class AverageMessageDone(DoneBaseComponent):
    def __init__(self, done_tolerance=None, **kwargs):
        super().__init__(**kwargs)
        self.done_tolerance = done_tolerance

    @property
    def done_tolerance(self):
        return self._done_tolerance

    @done_tolerance.setter
    def done_tolerance(self, value):
        assert type(value) in [int, float], "Done tolerance must be a number."
        assert value > 0, "Done tolerance must be positive."
        self._done_tolerance = value

    def get_done(self, agent, **kwargs):
        if isinstance(agent, BroadcastingAgent):
            average = np.average([
                other.message for other in self.agents.values()
                if isinstance(other, BroadcastingAgent)
            ])
            return np.abs(agent.message - average) <= self.done_tolerance
        else:
            return False

    def get_all_done(self, **kwargs):
        for agent in self.agents.values():
            if isinstance(agent, BroadcastingAgent):
                if not self.get_done(agent):
                    return False
        return True


class BlockingAgent(MovingAgent, GridObservingAgent):
    def __init__(self, **kwargs):
        super().__init__(blocking=True, **kwargs)


class BroadcastSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        self.position_state = PositionState(**kwargs)
        self.broadcasting_state = BroadcastingState(**kwargs)

        self.move_actor = MoveActor(**kwargs)
        self.broadcast_actor = BroadcastingActor(**kwargs)

        self.grid_observer = SingleGridObserver(**kwargs)
        self.broadcast_observer = BroadcastObserver(
            broadcasting_state=self.broadcasting_state, **kwargs
        )

        self.done = AverageMessageDone(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.broadcasting_state.reset(**kwargs)

        self.rewards = {agent.id: 0 for agent in self.agents.values()}

    def step(self, action_dict, **kwargs):
        # process broadcasts
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            receiving_agents = self.broadcast_actor.process_action(agent, action, **kwargs)
            if receiving_agents is not None:
                self.broadcasting_state.update_receipients(agent, receiving_agents)

        # process moves
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            move_result = self.move_actor.process_action(agent, action, **kwargs)
            if not move_result:
                self.rewards[agent.id] -= 0.1

        # Entropy penalty
        for agent_id in action_dict:
            self.rewards[agent_id] -= 0.01

    def render(self, **kwargs):
        super().render(**kwargs)
        for agent in self.agents.values():
            if isinstance(agent, BroadcastingAgent):
                print(f"{agent.id}: {agent.message}")
        print()

    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            **self.grid_observer.get_obs(agent, **kwargs),
            **self.broadcast_observer.get_obs(agent, **kwargs)
        }

    def get_reward(self, agent_id, **kwargs):
        reward = self.rewards[agent_id]
        self.rewards[agent_id] = 0
        return reward

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(agent_id, **kwargs)

    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)

    def get_info(self, **kwargs):
        return {}


if __name__ == "__main__":
    agents = {
        'broadcaster0': BroadcastingAgent(
            id='broadcaster0', encoding=1, broadcast_range=6, render_color='green'
        ),
        'broadcaster1': BroadcastingAgent(
            id='broadcaster1', encoding=1, broadcast_range=6, render_color='green'
        ),
        'broadcaster2': BroadcastingAgent(
            id='broadcaster2', encoding=1, broadcast_range=6, render_color='green'
        ),
        'broadcaster3': BroadcastingAgent(
            id='broadcaster3', encoding=1, broadcast_range=6, render_color='green'
        ),
        'blocker0': BlockingAgent(
            id='blocker0', encoding=2, move_range=2, view_range=3, render_color='black'
        ),
        'blocker1': BlockingAgent(
            id='blocker1', encoding=2, move_range=1, view_range=3, render_color='black'
        ),
        'blocker2': BlockingAgent(
            id='blocker2', encoding=2, move_range=1, view_range=3, render_color='black'
        ),
    }
    sim = BroadcastSim.build_sim(
        7, 7, agents=agents, broadcast_mapping={1: [1]}, done_tolerance=5e-10
    )

    sim.reset()
    fig = plt.figure()
    sim.render(fig=fig)

    done_agents = set()
    for i in range(50):
        action = {
            agent.id: agent.action_space.sample()
            for agent in agents.values() if agent.id not in done_agents
        }
        sim.step(action)
        for agent in agents:
            if agent not in done_agents:
                obs = sim.get_obs(agent)
            if sim.get_done(agent):
                done_agents.add(agent)

        sim.render(fig=fig)
        if sim.get_all_done():
            break
