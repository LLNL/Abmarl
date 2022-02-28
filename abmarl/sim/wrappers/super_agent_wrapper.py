
from gym.spaces import Dict

from abmarl.sim.agent_based_simulation import Agent
from abmarl.sim.wrappers import Wrapper

class SuperAgentWrapper(Wrapper):
    def __init__(self, sim, super_agent_mapping, **kwargs):
        self.sim = sim

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
