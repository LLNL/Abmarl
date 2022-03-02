
from gym.spaces import Dict

from abmarl.sim.agent_based_simulation import Agent
from abmarl.sim.wrappers import Wrapper

class SuperAgentWrapper(Wrapper):
    def __init__(self, sim, super_agent_mapping=None, **kwargs):
        self.sim = sim
        self.super_agent_mapping = super_agent_mapping
        self.agents = {}

        # Construct the agent dict with super agents
        for super_agent_id, sub_agent_list in self.super_agent_mapping.items():
            # Construct a mapping from the super agents to the sub agents' observation
            # and action spaces
            obs_mapping = {
                sub_agent_id: self.sim.agents[sub_agent_id].observation_space
                for sub_agent_id in sub_agent_list
            }
            action_mapping = {
                sub_agent_id: self.sim.agents[sub_agent_id].action_space
                for sub_agent_id in sub_agent_list
            }
            self.agents[super_agent_id] = Agent(
                id=super_agent_id,
                observation_space=Dict(obs_mapping),
                action_space=Dict(action_mapping)
            )

        # Add all uncovered agents to the dict of agetns
        for agent_id in self._uncovered_agents:
            self.agents[agent_id] = self.sim.agents[agent_id]

    @property
    def super_agent_mapping(self):
        return self._super_agent_mapping

    @super_agent_mapping.setter
    def super_agent_mapping(self, value):
        assert type(value) is dict, "super agent mapping must be a dictionary."
        self._covered_agents = set()
        for k, v in value.items():
            assert type(k) is str, "The keys super agent mapping must be the super agent's id."
            assert k not in self.sim.agents, "A super agent cannot have the same id as a sub_agent."
            assert type(v) is list, "The values in super agent mapping must be lists of agent ids."
            for sub_agent in v:
                assert type(sub_agent) is str, "The sub agents list must be agent ids."
                assert sub_agent in self.sim.agents, "The sub agent must be an agent in the underlying sim."
                assert sub_agent not in self._covered_agents, "The sub agent is already covered by another super agent."
                assert isinstance(self.sim.agents[sub_agent], Agent), "Covered agents must be learning Agents."
                self._covered_agents.add(sub_agent)
        self._uncovered_agents = self.sim.agents.keys() - self._covered_agents
        self._super_agent_mapping = value

    def step(self, action_dict, **kwargs):
        # "Unravel" the action dict so that super agent actions are decomposed
        # into the normal agent actions and then pass to the underlying sim.
        unravelled_action_dict = {}
        for agent_id, action in action_dict.items():
            assert agent_id not in self._covered_agents, \
                "We cannot receive actions from a sub agent that is covered by a super agent."
            if agent_id in self.super_agent_mapping: # A super agent action
                for sub_agent_id, sub_action in action.items():
                    # We can safely assume the format of the actions because we
                    # generated the action space
                    unravelled_action_dict[sub_agent_id] = sub_action
            else:
                unravelled_action_dict[agent_id] = action
        self.sim.step(unravelled_action_dict, **kwargs)

    def get_obs(self, agent_id, **kwargs):
        assert agent_id not in self._covered_agents, \
            "We cannot produce observations for a sub agent that is covered by a super agent."
        # We can safely assume the format of the observations because we generated
        # the observation space
        if agent_id in self.super_agent_mapping:
            return {
                sub_agent_id: self.sim.get_obs(sub_agent_id, **kwargs)
                for sub_agent_id in self.super_agent_mapping[agent_id]
            }
        else:
            return self.sim.get_obs(agent_id, **kwargs)

    def get_reward(self, agent_id, **kwargs):
        assert agent_id not in self._covered_agents, \
            "We cannot get rewards for a sub agent that is covered by a super agent."
        if agent_id in self.super_agent_mapping:
            return sum([
                self.sim.get_reward(sub_agent_id)
                for sub_agent_id in self.super_agent_mapping[agent_id]
            ])
        else:
            return self.sim.get_reward(agent_id, **kwargs)

    def get_done(self, agent_id, **kwargs):
        assert agent_id not in self._covered_agents, \
            "We cannot get done for a sub agent that is covered by a super agent."
        # TODO: explain why we choose all.
        # TODO: Do we need to add masking for active agents in the super agent?
        if agent_id in self.super_agent_mapping:
            return all([
                self.sim.get_done(sub_agent_id)
                for sub_agent_id in self.super_agent_mapping[agent_id]
            ])
        else:
            return self.sim.get_done(agent_id, **kwargs)

