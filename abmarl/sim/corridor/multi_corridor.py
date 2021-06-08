from enum import IntEnum

from gym.spaces import Box, Discrete, MultiBinary
import numpy as np

from abmarl.sim import Agent, AgentBasedSimulation


class MultiCorridor(AgentBasedSimulation):
    """
    MultiCorridor simulation used for testing. Multiple agents spawn along
    a corridor and can choose to move left, right, or stay still. The agents
    must learn to move to the right until they reach the end position. The agent
    cannot move to spaces that area already occupied by other agents. An agent
    is done once it moves to the end position.

    The agent can observe its own position. It can also see if the two squares
    near it are occupied.
    """
    class Actions(IntEnum):
        LEFT = 0
        STAY = 1
        RIGHT = 2

    def __init__(self, end=10, num_agents=5):
        self.end = end
        agents = {}
        for i in range(num_agents):
            agents[f'agent{i}'] = Agent(
                id=f'agent{i}',
                action_space=Discrete(3),
                observation_space={
                    'position': Box(0, self.end-1, (1,), np.int),
                    'left': MultiBinary(1),
                    'right': MultiBinary(1)
                }
            )
        self.agents = agents

        self.finalize()

    def reset(self, **kwargs):
        """
        Randomly locate the agents on unique spaces within the Corridor.
        """
        location_sample = np.random.choice(self.end-1, len(self.agents), False)
        self.corridor = np.empty(self.end, dtype=object)
        for i, agent in enumerate(self.agents.values()):
            agent.position = location_sample[i]
            self.corridor[location_sample[i]] = agent

        # Track the agents' rewards over multiple steps.
        self.reward = {agent_id: 0 for agent_id in self.agents}

    def step(self, action_dict, **kwargs):
        """
        The agents can choose to move left, move right, or stay where they are. If
        an agent bumps into another agent, then both agents receive a penalty.
        The offending agent receives a larger penalty than the offended agent.
        The agent is done when it reaches the end of the corridor.
        """
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if action == self.Actions.LEFT:
                if agent.position != 0 and self.corridor[agent.position-1] is None:
                    # Good move, no extra penalty
                    self.corridor[agent.position] = None
                    agent.position -= 1
                    self.corridor[agent.position] = agent
                    self.reward[agent_id] -= 1 # Entropy penalty
                elif agent.position == 0: # Tried to move left from left-most square
                    # Bad move, only acting agent is involved and should be penalized.
                    self.reward[agent_id] -= 5 # Bad move
                else: # There was another agent to the left of me that I bumped into
                    # Bad move involving two agents. Both are penalized
                    self.reward[agent_id] -= 5 # Penalty for offending agent
                    self.reward[self.corridor[agent.position-1].id] -= 2 # Penalty for offended
            elif action == self.Actions.RIGHT:
                if self.corridor[agent.position + 1] is None:
                    # Good move, but is the agent done?
                    self.corridor[agent.position] = None
                    agent.position += 1
                    if agent.position == self.end-1:
                        # Agent has reached the end of the corridor!
                        self.reward[agent_id] += self.end ** 2
                    else:
                        # Good move, no extra penalty
                        self.corridor[agent.position] = agent
                        self.reward[agent_id] -= 1 # Entropy penalty
                else: # There was another agent to the right of me that I bumped into
                    # Bad move involving two agents. Both are penalized
                    self.reward[agent_id] -= 5 # Penalty for offending agent
                    self.reward[self.corridor[agent.position+1].id] -= 2 # Penalty for offended
            elif action == self.Actions.STAY:
                self.reward[agent_id] -= 1 # Entropy penalty

    def render(self, *args, fig=None, **kwargs):
        """
        Visualize the state of the simulation. If a figure is received, then we
        will draw but not actually plot because we assume the caller will do the
        work (e.g. with an Animation object). If there is no figure received, then
        we will draw and plot the simulation.
        """
        draw_now = fig is None
        if draw_now:
            from matplotlib import pyplot as plt
            fig = plt.gcf()

        fig.clear()
        ax = fig.gca()
        ax.set(xlim=(-0.5, self.end + 0.5), ylim=(-0.5, 0.5))
        ax.set_xticks(np.arange(-0.5, self.end + 0.5, 1.))
        ax.scatter(np.array(
            [agent.position for agent in self.agents.values()]),
            np.zeros(len(self.agents)),
            marker='s', s=200, c='g'
        )

        if draw_now:
            plt.plot()
            plt.pause(1e-17)

    def get_obs(self, agent_id, **kwargs):
        """
        Agents observe their own position and if the squares to the left and right
        are occupied by other agents.
        """
        agent_position = self.agents[agent_id].position
        if agent_position == 0 or self.corridor[agent_position-1] is None:
            left = False
        else:
            left = True
        if agent_position == self.end-1 or self.corridor[agent_position+1] is None:
            right = False
        else:
            right = True
        return {
            'position': [agent_position],
            'left': [left],
            'right': [right],
        }

    def get_done(self, agent_id, **kwargs):
        """
        Agents are done when they reach the end of the corridor.
        """
        return self.agents[agent_id].position == self.end - 1

    def get_all_done(self, **kwargs):
        """
        Simulation is done when all agents have reached the end of the corridor.
        """
        for agent in self.agents.values():
            if agent.position != self.end - 1:
                return False
        return True

    def get_reward(self, agent_id, **kwargs):
        """
        The agent's reward is tracked throughout the simulation and returned here.
        """
        agent_reward = self.reward[agent_id]
        self.reward[agent_id] = 0
        return agent_reward

    def get_info(self, agent_id, **kwargs):
        """
        Just return an empty dictionary becuase this simulation does not track
        any info.
        """
        return {}
