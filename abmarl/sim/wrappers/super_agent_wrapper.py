
from gym.spaces import Dict

from abmarl.sim.agent_based_simulation import Agent
from abmarl.sim.wrappers import Wrapper
from abmarl.sim.wrappers.ravel_discrete_wrapper import unravel

class SuperAgentWrapper(Wrapper):
    def __init__(self, sim, super_agent_mapping, **kwargs):
        self.sim = sim
        self.super_agent_mapping = super_agent_mapping
        # TODO: It would be more effective to automatically add the non-included
        # agents in the super_agent_mapping so that they map to themselves. This
        # would avoid all the if statements and two-logical processing that we
        # have to do in the functions below.

        # TODO: Assert that two super agents don't control the same sub agent.

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

    def step(self, action_dict, **kwargs):
        # "Unravel" the action dict so that super agent actions are decomposed
        # into the normal agent actions and then pass to the underlying sim.
        unravelled_action_dict = {}
        for agent_id, action in action_dict.items():
            if agent_id in self.super_agent_mapping: # A super agent action
                for sub_agent_id, sub_action in action.items():
                    unravelled_action_dict[sub_agent_id] = sub_action
            else:
                unravelled_action_dict[agent_id] = action
        self.sim.step(unravelled_action_dict, **kwargs)

    def get_obs(self, agent_id, **kwargs):
        if agent_id in self.super_agent_mapping:
            return {
                sub_agent_id: self.sim.get_obs(sub_agent_id, **kwargs)
                for sub_agent_id in self.super_agent_mapping[agent_id]
            }
        else:
            return self.sim.get_obs(agent_id, **kwargs)

    def get_rewards(self, agent_id, **kwargs):
        if agent_id in self.super_agent_mapping:
            return sum([
                self.sim.get_reward(sub_agent_id)
                for sub_agent_id in self.super_agent_mapping[agent_id]
            ])
        else:
            return self.sim.get_reward(agent_id, **kwargs)
