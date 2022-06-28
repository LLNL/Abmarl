
import os
from pprint import pprint
from abmarl.policies.policy import RandomPolicy

from abmarl.trainers.base import MultiPolicyTrainer

class DebugTrainer(MultiPolicyTrainer):
    """
    Debug the training setup.

    The DebugTrainer generates episodes using the simulation and policies. Rather
    than training those policies, The DebugTrainer simply dumps the observations,
    actions, rewards, and dones to disk.

    The DebugTrainer can be run without policies. In this case, we generate random
    policies for each agent. This will allow you to debug the simulation without
    having to debug the policy setup too.
    """
    def __init__(self, sim=None, policies=None, output_dir=None, **kwargs):
        self.sim = sim # TODO: Does this use the super's property?
        if not policies:
            # Create random policies
            policies = {
                agent.id: RandomPolicy(
                    action_space=agent.action_space,
                    observation_space=agent.observation_space
                ) for agent in sim.agents.values()
            }
        super().__init__(policies=policies, **kwargs)
        self.output_dir = output_dir

    @property
    def output_dir(self):
        """
        The directory for where to dump the episode data.
        """
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        assert type(value) is str, "Output directory must be a string."
        if not os.path.exists(value):
            os.makedirs(value)
        self._output_dir = value

    def train(self, iterations=5, render=False, **kwargs):
        """
        Generate episodes of data.

        Nothing is technically trained here. We just generate and dump the data
        and visualize the simulation if requested.

        Args:
            iterations: The number of episodes to generate.
            render: Set to True to visualize the simulation.
        """
        # TOOD:
        # Output directory
        for i in range(iterations):
            observations, actions, rewards, dones = self.generate_episode(render=render, **kwargs)

            # Setup dump files
            with open(os.path.join(self.output_dir, f"Episode_{i}.txt"), 'w') as debug_dump:
                debug_dump.write("Observations:\n")
                pprint(observations, stream=debug_dump)
                debug_dump.write("\nActions:\n")
                pprint(actions, stream=debug_dump)
                debug_dump.write("\nRewards:\n")
                pprint(rewards, stream=debug_dump)
                debug_dump.write("\nDones:\n")
                pprint(dones, stream=debug_dump)
