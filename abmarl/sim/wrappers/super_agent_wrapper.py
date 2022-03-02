
from gym.spaces import Dict

from abmarl.sim.agent_based_simulation import Agent
from abmarl.sim.wrappers import Wrapper

class SuperAgentWrapper(Wrapper):
    def __init__(self, sim, super_agent_mapping, **kwargs):
        self.sim = sim
        self.super_agent_mapping = super_agent_mapping

        # Construct a mapping from the super agents to the sub agents observation
        # and action spaces
        obs_mapping = {}
        action_mapping = {}
        for super_agent_id, agent_list in super_agent_mapping.items():
            obs_mapping[super_agent_id] = {}
            action_mapping[super_agent_id] = {}
            for agent_id in agent_list:
                obs_mapping[super_agent_id][agent_id] = self.sim.agents[agent_id].observation_space
                action_mapping[super_agent_id][agent_id] = self.sim.agents[agent_id].action_space
            obs_mapping[super_agent_id] = Dict(obs_mapping[super_agent_id])
            action_mapping[super_agent_id] = Dict(action_mapping[super_agent_id])

        # Determine which sub agents are not combined into a super agent
        covered_agents = set(agent_id for agents in super_agent_mapping.values() for agent_id in agents)
        non_super_agents = set(self.sim.agents) - covered_agents

        # Store the super and non-covered sub agents together
        self.agents = {
            **{
                super_agent_id: Agent(
                    id=super_agent_id,
                    observation_space=obs_mapping[super_agent_id],
                    action_space=action_mapping[super_agent_id]
                ) for super_agent_id in super_agent_mapping
            },
            **{
                agent_id: self.sim.agents[agent_id] for agent_id in non_super_agents
            }
        }

    @property
    def super_agent_mapping(self):
        return self._super_agent_mapping

    @super_agent_mapping.setter
    def super_agent_mapping(self, value):
        assert type(value) is dict, "super agent mapping must be a dictionary."
        self._covered_agents = set()
        for k, v in value.items():
            assert type(k) is str, "The keys super agent mapping must be the super agent's id."
            assert type(v) is list, "The values in super agent mapping must be lists of agent ids."
            for sub_agent in v:
                assert type(sub_agent) is str, "The sub agents list must be agent ids."
                assert sub_agent in self.sim.agents, "The sub agent must be an agent in the underlying sim."
                assert sub_agent not in self._covered_agents, "The sub agent is already covered by another super agent."
                self._covered_agents.add(sub_agent)
        self._uncovered_agents = self.sim.agents.keys() - self._covered_agents
        self._super_agent_mapping = value

    def step(self, action_dict, **kwargs):
        # "Unravel" the action dict so that super agent actions are decomposed
        # into the normal agent actions and then pass to the underlying sim.
        unravelled_action_dict = {}
        for agent_id, action in action_dict.items():
            # TODO: Assert agent id is not in the covered agents since we want
            # the wrapper to raise an error if it receives an action from a sub_agent.
            if agent_id in self.super_agent_mapping: # A super agent action
                for sub_agent_id, sub_action in action.items():
                    unravelled_action_dict[sub_agent_id] = sub_action
            else:
                unravelled_action_dict[agent_id] = action
        self.sim.step(unravelled_action_dict, **kwargs)

    def get_obs(self, agent_id, **kwargs):
        # TODO: Assert that agent_id is not covered
        if agent_id in self.super_agent_mapping:
            return {
                sub_agent_id: self.sim.get_obs(sub_agent_id, **kwargs)
                for sub_agent_id in self.super_agent_mapping[agent_id]
            }
        else:
            return self.sim.get_obs(agent_id, **kwargs)

    def get_reward(self, agent_id, **kwargs):
        # TODO: Assert that agent_id is not covered
        if agent_id in self.super_agent_mapping:
            return sum([
                self.sim.get_reward(sub_agent_id)
                for sub_agent_id in self.super_agent_mapping[agent_id]
            ])
        else:
            return self.sim.get_reward(agent_id, **kwargs)

    def get_done(self, agent_id, **kwargs):
        # TODO: Assert that agent_id is not covered
        # TODO: explain why we choose all.
        if agent_id in self.super_agent_mapping:
            return all([
                self.sim.get_done(sub_agent_id)
                for sub_agent_id in self.super_agent_mapping[agent_id]
            ])
        else:
            return self.sim.get_done(agent_id, **kwargs)

