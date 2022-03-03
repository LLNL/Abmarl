
from gym.spaces import Dict

from abmarl.sim.agent_based_simulation import Agent
from abmarl.sim.wrappers import Wrapper


class SuperAgentWrapper(Wrapper):
    def __init__(self, sim, super_agent_mapping=None, **kwargs):
        self.sim = sim
        self.super_agent_mapping = super_agent_mapping

    @property
    def super_agent_mapping(self):
        return self._super_agent_mapping

    @super_agent_mapping.setter
    def super_agent_mapping(self, value):
        assert type(value) is dict, "super agent mapping must be a dictionary."
        self._covered_agents = set()
        for k, v in value.items():
            assert type(k) is str, "The keys super agent mapping must be the super agent's id."
            assert k not in self.sim.agents, \
                "A super agent cannot have the same id as an agent from the underlying sim."
            assert type(v) is list, "The values in super agent mapping must be lists of agent ids."
            for covered_agent in v:
                assert type(covered_agent) is str, "The covered agents list must be agent ids."
                assert covered_agent in self.sim.agents, "The covered agent must be an agent in the underlying sim."
                assert covered_agent not in self._covered_agents, "The agent is already covered by another super agent."
                assert isinstance(self.sim.agents[covered_agent], Agent), "Covered agents must be learning Agents."
                self._covered_agents.add(covered_agent)
        self._uncovered_agents = self.sim.agents.keys() - self._covered_agents
        self._super_agent_mapping = value
        # We need to reconstruct the agent dictionary if the super agent mapping
        # ever changes
        self._construct_agents_from_super_agent_mapping()

    def step(self, action_dict, **kwargs):
        # "Unravel" the action dict so that super agent actions are decomposed
        # into the normal agent actions and then pass to the underlying sim.
        unravelled_action_dict = {}
        for agent_id, action in action_dict.items():
            assert agent_id not in self._covered_agents, \
                "We cannot receive actions from an agent that is covered by a super agent."
            if agent_id in self.super_agent_mapping: # A super agent action
                for covered_agent_id, covered_action in action.items():
                    # We can safely assume the format of the actions because we
                    # generated the action space
                    if not self.sim.get_done(covered_agent_id):
                        # We don't want to send the simulation actions from covered
                        # agents that are done
                        unravelled_action_dict[covered_agent_id] = covered_action
            else:
                unravelled_action_dict[agent_id] = action
        self.sim.step(unravelled_action_dict, **kwargs)

    def get_obs(self, agent_id, **kwargs):
        assert agent_id not in self._covered_agents, \
            "We cannot produce observations for an agent that is covered by a super agent."
        # We can safely assume the format of the observations because we generated
        # the observation space
        if agent_id in self.super_agent_mapping:
            return {
                covered_agent_id: self.sim.get_obs(covered_agent_id, **kwargs)
                for covered_agent_id in self.super_agent_mapping[agent_id]
            }
        else:
            return self.sim.get_obs(agent_id, **kwargs)

    def get_reward(self, agent_id, **kwargs):
        assert agent_id not in self._covered_agents, \
            "We cannot get rewards for an agent that is covered by a super agent."
        if agent_id in self.super_agent_mapping:
            return sum([
                self.sim.get_reward(covered_agent_id)
                for covered_agent_id in self.super_agent_mapping[agent_id]
            ])
        else:
            return self.sim.get_reward(agent_id, **kwargs)

    def get_done(self, agent_id, **kwargs):
        assert agent_id not in self._covered_agents, \
            "We cannot get done for an agent that is covered by a super agent."
        # TODO: explain why we choose all.
        # TODO: Do we need to add masking for active agents in the super agent?
        if agent_id in self.super_agent_mapping:
            return all([
                self.sim.get_done(covered_agent_id)
                for covered_agent_id in self.super_agent_mapping[agent_id]
            ])
        else:
            return self.sim.get_done(agent_id, **kwargs)

    def get_info(self, agent_id, **kwargs):
        assert agent_id not in self._covered_agents, \
            "We cannot get info for an agent that is covered by a super agent."
        if agent_id in self.super_agent_mapping:
            return {
                covered_agent_id: self.sim.get_info(covered_agent_id, **kwargs)
                for covered_agent_id in self.super_agent_mapping[agent_id]
            }
        else:
            return self.sim.get_info(agent_id, **kwargs)

    def _construct_agents_from_super_agent_mapping(self):
        agents = {}

        # Construct the agent dict with super agents
        for super_agent_id, covered_agent_list in self.super_agent_mapping.items():
            # Construct a mapping from the super agents to the covered agents' observation
            # and action spaces
            obs_mapping = {
                covered_agent_id: self.sim.agents[covered_agent_id].observation_space
                for covered_agent_id in covered_agent_list
            }
            action_mapping = {
                covered_agent_id: self.sim.agents[covered_agent_id].action_space
                for covered_agent_id in covered_agent_list
            }
            agents[super_agent_id] = Agent(
                id=super_agent_id,
                observation_space=Dict(obs_mapping),
                action_space=Dict(action_mapping)
            )

        # Add all uncovered agents to the dict of agetns
        for agent_id in self._uncovered_agents:
            agents[agent_id] = self.sim.agents[agent_id]

        self.agents = agents


