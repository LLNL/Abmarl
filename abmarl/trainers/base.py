
from abc import ABC, abstractmethod

class MulitAgentTrainer(ABC):
    def __init__(self, sim=None, policies=None, policy_mapping_fn=None, **kwargs):
        self.sim = sim
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn

    @property
    def sim(self):
        return self._sim

    @sim.setter
    def sim(self, value):
        self._sim = value

    @property
    def policies(self):
        """
        A dictionary that maps the policy id's to a policy object.
        """
        return self._policies

    @policies.setter
    def policies(self, value):
        self._policies = value

    @property
    def policy_mapping_fn(self):
        """
        A function that takes an agent's id as input and outputs its corresponding policy id.
        """
        return self._policy_mapping_fn

    @policy_mapping_fn.setter
    def policy_mapping_fn(self, value):
        self._policy_mapping_fn = value

    def compute_actions(self, obs):
        """
        Compute actions for agents in the observation.

        Forwards the observations to the respective policy for each agent.

        Args:
            obs: an observation dictionary, where the keys are the agents for which
                you want to generate actions and the values are the observations.

        Returns:
            An action dictionary where the keys are the agent ids from the obs
                and the values are the actions generated from that agent's policy.
        """
        return {
            agent_id: self.policies[self.policy_mapping_fn[agent_id]].compute_action(obs[agent_id])
            for agent_id in obs
        }

    @abstractmethod
    def train(self):
        pass
