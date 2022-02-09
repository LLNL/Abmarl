
from abc import ABC, abstractmethod

from abmarl.pols.policy import Policy
from abmarl.managers import SimulationManager
from abmarl.sim.agent_based_simulation import Agent

class MultiAgentTrainer(ABC):
    """
    Train policies with data generated by agents interacting in a simulation.
    """
    def __init__(self, sim=None, policies=None, policy_mapping_fn=None, **kwargs):
        self.sim = sim
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self._check_agent_policy_alignment()

    @property
    def sim(self):
        """
        The SimulationManager.
        """
        return self._sim

    @sim.setter
    def sim(self, value):
        assert isinstance(value, SimulationManager), "sim must be a Simulation Manager."
        self._sim = value

    @property
    def policies(self):
        """
        A dictionary that maps the policy id's to a policy object.
        """
        return self._policies

    @policies.setter
    def policies(self, value):
        assert type(value) is dict, "Policies must be a dict"
        for k, v in value.items():
            assert type(k) is str, \
                "The keys in the policies dictionary must be the policy names as strings."
            assert isinstance(v, Policy), \
                "The values in the policies dictionary must by Policy objects."
        self._policies = value

    @property
    def policy_mapping_fn(self):
        """
        A function that takes an agent's id as input and outputs its corresponding policy id.
        """
        return self._policy_mapping_fn

    @policy_mapping_fn.setter
    def policy_mapping_fn(self, value):
        assert callable(value), "Policy Mapping Function must be a function."
        self._policy_mapping_fn = value

    def compute_actions(self, obs):
        """
        Compute actions for agents in the observation.

        Forwards the observations to the respective policy for each agent.

        Args:
            obs: an observation dictionary, where the keys are the agents reporting
                from the sim and the values are the observations.

        Returns:
            An action dictionary where the keys are the agents from the observation
                and the values are the actions generated from each agent's policy.
        """
        return {
            agent_id: self.policies[self.policy_mapping_fn(agent_id)].compute_action(obs[agent_id])
            for agent_id in obs
        }

    # TODO: Upgrade to generate_batch
    def generate_episode(self, horizon=200):
        """
        Generate an episode of data.

        The fundamental data object is a SAR--a (state, action, reward) tuple.
        We restart the sim, generating initial observations (states) for agents
        reporting from the sim. Then we use the compute_action function to generate
        actions for those reporting agents. Those actions are given to the sim,
        which steps forward and generates rewards and new observations for reporting
        agents. This loop continues until the simulation is done or we hit the
        horizon.

        Args:
            horizon: The maximum number of steps per epsidoe. They episode may
                finish early, but it will not progress further than this number
                of steps.

        Returns:
            Three dictionaries, one for observations, another for actions, and
            another for rewards, thus making up the SAR sequences. The data is
            organized by agent_id, so you would call
                {observations, actions, rewards}[agent_id][i]
            in order to extract the ith SAR for an agent.
            NOTE: In multiagent simulations, the number of SARs may differ for
            each agent.
            
        """
        # Reset the simulation and policies
        obs = self.sim.reset()
        done = {agent: False for agent in obs}
        for policy in self.policies.values():
            policy.reset()

        # Data collection
        observations, actions, rewards = {}, {}, {}
        for agent_id, agent_obs in obs.items():
            observations[agent_id] = [agent_obs]

        # Generate episode of data
        for j in range(horizon):
            action = self.compute_actions(obs)
            obs, reward, done, _ = self.sim.step(action)

            # Store the data
            for agent_id, agent_obs in obs.items():
                try:
                    observations[agent_id].append(agent_obs)
                except KeyError:
                    observations[agent_id] = [agent_obs]
            for agent_id, agent_reward in reward.items():
                try:
                    rewards[agent_id].append(agent_reward)
                except KeyError:
                    rewards[agent_id] = [agent_reward]
            for agent_id, agent_action in action.items():
                try:
                    actions[agent_id].append(agent_action)
                except KeyError:
                    actions[agent_id] = [agent_action]

            # Exit if we're done
            if done['__all__']:
                break

            # We should not request actions for any more done agents.
            for agent_id, agent_done in done.items():
                if agent_done:
                    del obs[agent_id]

        return observations, actions, rewards

    @abstractmethod
    def train(self, iterations=10_000, **kwargs):
        """
        Train the policy objects using generated data.

        This function is abstract and should be implemented by the algorithm.
        The implementation should look something like this:
        for iter in range(iterations):
            observations, actions, rewards = self.generate_episode()
            # Implementation: update the policy with the generated data.

        Args:
            iterations: The number of training iterations.
            **kwargs: Any additional parameter your algorithm may need.
        """
        pass

    def _check_agent_policy_alignment(self):
        """
        Check agent-policy alignment.

        Check that every agent that acts and observes is assigned a policy and
        that the action and observation spaces between each agent and its assigned
        policy align.
        """
        # Quick assertion that all the spaces lines up
        for agent in self.sim.agents.values():
            if not isinstance(agent, Agent): continue
            policy_id = self.policy_mapping_fn(agent.id)
            policy = self.policies[policy_id]
            assert agent.action_space == policy.action_space, \
                f"agent{agent.id} has been assigned to policy {policy_id} but " + \
                "the action spaces are different."
            assert agent.observation_space == policy.observation_space, \
                f"agent{agent.id} has been assigned to policy {policy_id} but " + \
                "the observation spaces are different."


class SingleAgentTrainer(MultiAgentTrainer):
    """
    Train a policy with data generated by a single agent in a simulation.

    Since there is only a single agent, there is only a single policy to train.
    """
    def __init__(self, sim=None, policy=None, **kwargs):
        self.sim = sim
        self.policy = policy
        self._check_agent_policy_alignment()

    @property
    def policies(self):
        """
        The policy to train.
        """
        return {'policy': self._policy}

    @policies.setter
    def policies(self, value):
        assert isinstance(value, Policy), "policy must be a Policy object."
        self._policy = value

    @property
    def policy_mapping_fn(self):
        """
        Always returns "policy", which is the name we give the policy.
        """
        return 'policy'
