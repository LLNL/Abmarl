
import warnings

from gym.spaces import Dict, Discrete

from abmarl.sim.agent_based_simulation import Agent
from abmarl.sim.wrappers import Wrapper
from abmarl.tools import gym_utils as gu


class SuperAgentWrapper(Wrapper):
    """
    The SuperAgentWrapper creates "super" agents who cover and control multiple agents.

    The super agents take the observation and action spaces of all their covered
    agents. In addition, the observation space is given a "mask" channel to indicate
    which of their covered agents is done. This channel is important because
    the simulation dynamics change when a covered agent is done but the super agent
    may still be active (see comments on get_done). Without this mask, the super
    agent would experience completely different simulation dynamcis for some of
    its covered agents with no indication as to why.

    Unless handled carefully, the super agent will generate observations for done
    covered agents. This may contaminate the training data with an unfair advantage.
    For exmample, a dead covered agent should not be able to provide the super agent with
    useful information. In order to correct this, the user may supply the null
    observation for an ObservingAgent. When a covered agent is done, the SuperAgentWrapper
    will try to use its null observation going forward.

    Furthermore, super agents may still report actions for covered agents that
    are done. This wrapper filters out those actions before passing them to the
    underlying sim. See step for more details.
    """
    def __init__(self, sim, super_agent_mapping=None, **kwargs):
        self.sim = sim
        self.super_agent_mapping = super_agent_mapping
        self._warning_issued = False

    @property
    def super_agent_mapping(self):
        """
        A dictionary that maps from a super agent's id to a list of covered agent ids.

        Suppose our simulation has 5 agents and we use the following super agent mapping:
        {'super0': ['agent0', 'agent1'], 'super1': ['agent3', 'agent4']}
        The resulting agents dict would have keys 'super0', 'super1', and 'agent2';
        where 'agent0', 'agent1', 'agent3', and 'agent4' have been covered by the
        super agents and 'agent2' is left uncovered and therefore included in the
        dict of agents. If the super agent mapping is changed, then the dictionary
        of agents gets recreated immediately.

        Super agents cannot have the same id as any of the agents in the simulation.
        Two super agents cannot cover the same agent. All covered agents must be
        learning agents.
        """
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
                assert covered_agent in self.sim.agents, \
                    "The covered agent must be an agent in the underlying sim."
                assert covered_agent not in self._covered_agents, \
                    "The agent is already covered by another super agent."
                assert isinstance(self.sim.agents[covered_agent], Agent), \
                    "Covered agents must be learning Agents."
                self._covered_agents.add(covered_agent)
        self._uncovered_agents = self.sim.agents.keys() - self._covered_agents
        self._super_agent_mapping = value
        # We need to reconstruct the agent dictionary if the super agent mapping
        # ever changes
        self._construct_agents_from_super_agent_mapping()

    def reset(self, **kwargs):
        self._last_obs_reported = {
            agent: False
            for agent in self._covered_agents
        }
        self._last_reward_reported = {
            agent: False
            for agent in self._covered_agents
        }
        self.sim.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        """
        Give actions to the simulation.

        Super agent actions are decomposed into the covered agent actions and
        then passed to the underlying sim. Because of the nature of this wrapper,
        the super agents may provide actions for covered agents that are already
        done. We filter out these actions.

        Args:
            action_dict: Dictionary that maps agent ids to the actions. Covered
                agents should not be present.
        """
        unravelled_action_dict = {}
        for agent_id, action in action_dict.items():
            assert agent_id not in self._covered_agents, \
                "We cannot receive actions from an agent that is covered by a super agent."
            if agent_id in self.super_agent_mapping: # A super agent action
                # We can safely assume the format of the actions because we
                # generated the action space
                for covered_agent_id, covered_action in action.items():
                    # We don't want to send the simulation actions from covered
                    # agents that are done
                    if not self.sim.get_done(covered_agent_id):
                        unravelled_action_dict[covered_agent_id] = covered_action
            else:
                unravelled_action_dict[agent_id] = action
        self.sim.step(unravelled_action_dict, **kwargs)

    def get_obs(self, agent_id, **kwargs):
        """
        Report observations from the simulation.

        Super agent observations are collected from their covered agents. Super
        agents also have a "mask" channel that tells them which of their covered
        agent is done. This should assist the super agent in understanding the
        changing simulation dynamics for done agents (i.e. why actions from done
        agents don't do anything).

        The super agent will report an observation for done covered agents. This may
        result in an unfair advantage during training (e.g. dead agent should not
        produce useful information), and Abmarl will issue a warning. To properly
        handle this, the user can supply the null observation for each covered agent. In
        that case, the super agent will use the null observation for any done covered agents.

        Args:
            agent_id: The id of the agent for whom to produce an observation. Should
                not be a covered agent.

        Returns:
            The requested observation. Super agent observations are collected from
            the covered agents.
        """
        assert agent_id not in self._covered_agents, \
            "We cannot produce observations for an agent that is covered by a super agent."
        # We can safely assume the format of the observations because we generated
        # the observation space
        if agent_id in self.super_agent_mapping:
            obs = {'mask': {}}
            for covered_agent in self.super_agent_mapping[agent_id]:
                if self.sim.get_done(covered_agent, **kwargs):
                    if self._last_obs_reported[covered_agent]:
                        obs[covered_agent] = self._get_null_obs(covered_agent, **kwargs)
                        obs['mask'][covered_agent] = False
                    else:
                        obs[covered_agent] = self.sim.get_obs(covered_agent, **kwargs)
                        obs['mask'][covered_agent] = False
                        self._last_obs_reported[covered_agent] = True
                else:
                    obs[covered_agent] = self.sim.get_obs(covered_agent, **kwargs)
                    obs['mask'][covered_agent] = True
            return obs
        else:
            return self.sim.get_obs(agent_id, **kwargs)

    def get_reward(self, agent_id, **kwargs):
        """
        Report the agent's reward.

        A super agent's reward is the sum of all its active covered agents' rewards.

        Args:
            agent_id: The id of the agent for whom to report the reward. Should
                not be a covered agent.

        Returns:
            The requested reward. Super agent rewards are summed from the active covered
                agents.
        """
        assert agent_id not in self._covered_agents, \
            "We cannot get rewards for an agent that is covered by a super agent."
        if agent_id in self.super_agent_mapping:
            sum = 0
            for covered_agent in self.super_agent_mapping[agent_id]:
                if self.sim.get_done(covered_agent, **kwargs):
                    if not self._last_reward_reported[covered_agent]:
                        sum += self.sim.get_reward(covered_agent, **kwargs)
                        self._last_reward_reported[covered_agent] = True
                else:
                    sum += self.sim.get_reward(covered_agent, **kwargs)
            return sum
        else:
            return self.sim.get_reward(agent_id, **kwargs)

    def get_done(self, agent_id, **kwargs):
        """
        Report the agent's done condition.

        Because super agents are composed of multiple agents, it could be the case
        that some covered agents are done while other are not for the same super
        agent. Because we still want those non-done agents to interact with the
        simulation, the super agent only reports done when ALL of its covered agents
        report done.

        Args:
            agent_id: The id of the agent for whom to report the done condition.
                Should not be a covered agent.

        Returns:
            The requested done conndition. Super agents are done when all their
                covered agents are done.
        """
        assert agent_id not in self._covered_agents, \
            "We cannot get done for an agent that is covered by a super agent."
        if agent_id in self.super_agent_mapping:
            return all([
                self.sim.get_done(covered_agent_id)
                for covered_agent_id in self.super_agent_mapping[agent_id]
            ])
        else:
            return self.sim.get_done(agent_id, **kwargs)

    def get_info(self, agent_id, **kwargs):
        """
        Report the agent's additional info.

        Args:
            agent_id: The id of the agent for whom to get info. Should not be a
                covered agent.

        Returns:
            The requested info. Super agents info is collected from covered agents.
        """
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
            obs_mapping = {'mask': {}}
            for covered_agent_id in covered_agent_list:
                obs_mapping[covered_agent_id] = self.sim.agents[covered_agent_id].observation_space
                obs_mapping['mask'][covered_agent_id] = Discrete(2)
            action_mapping = {
                covered_agent_id: self.sim.agents[covered_agent_id].action_space
                for covered_agent_id in covered_agent_list
            }
            agents[super_agent_id] = Agent(
                id=super_agent_id,
                observation_space=gu.make_dict(obs_mapping),
                action_space=Dict(action_mapping)
            )

        # Add all uncovered agents to the dict of agents
        for agent_id in self._uncovered_agents:
            agents[agent_id] = self.sim.agents[agent_id]

        self.agents = agents

    def _get_null_obs(self, agent_id, **kwargs):
        assert agent_id in self._covered_agents, "Can only use null obs for covered agents."
        if self.sim.agents[agent_id].null_observation:
            return self.sim.agents[agent_id].null_observation
        # if agent_id in self.null_obs:
        #     return self.null_obs[agent_id]
        else:
            if not self._warning_issued:
                self._warning_issued = True
                warnings.warn(
                    "Some covered agents in the SuperAgentWrapper do not specify "
                    "a null observation. This may corrupt the learning data.",
                )
            return self.sim.get_obs(agent_id, **kwargs)
