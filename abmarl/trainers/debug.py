
import os
from pprint import pprint
import time

from abmarl.policies.policy import RandomPolicy
from abmarl.sim.agent_based_simulation import Agent

from abmarl.trainers.base import MultiPolicyTrainer


class DebugTrainer(MultiPolicyTrainer):
    """
    Debug the training setup.

    The DebugTrainer generates episodes using the simulation and policies. Rather
    than training those policies, The DebugTrainer simply dumps the observations,
    actions, rewards, and dones to disk.

    The DebugTrainer can be run without policies. In this case, it generates a
    random policy for each agent. This effectively debug the simulation without
    having to debug the policy setup too.
    """
    def __init__(self, policies=None, name=None, output_dir=None, **kwargs):
        if not policies:
            self.sim = kwargs['sim']
            # Create random policies
            self.policies = {
                agent.id: RandomPolicy(
                    action_space=agent.action_space,
                    observation_space=agent.observation_space
                ) for agent in self.sim.agents.values() if isinstance(agent, Agent)
            }
            self.policy_mapping_fn = lambda agent_id: agent_id
        else:
            super().__init__(policies=policies, **kwargs)
        self.name = name
        self.output_dir = output_dir

    @property
    def name(self):
        """
        The name of the experiment.

        If name is not specified, then we just use "DEBUG". We append the name
        with the date and time.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            value = 'DEBUG'
        else:
            assert type(value) is str, "Name must be a string."
        self._name = '{}_{}'.format(value, time.strftime('%Y-%m-%d_%H-%M'))

    @property
    def output_dir(self):
        """
        The directory for where to dump the episode data.

        If the output dir is not specified, then we use "~/abmarl_results/". We
        append the experiment name to the end of the directory.
        """
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        if value is None:
            value = os.path.join(os.path.expanduser("~"), 'abmarl_results')
        else:
            assert type(value) is str, "Output directory must be a string."
            value = os.path.join(value, 'abmarl_results')
        output_dir = os.path.join(value, self.name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._output_dir = output_dir

    def train(self, iterations=5, render=False, **kwargs):
        """
        Generate episodes and write write to disk.

        Nothing is trained here. We just generate and dump the data
        and visualize the simulation if requested.

        Args:
            iterations: The number of episodes to generate.
            render: Set to True to visualize the simulation.
        """
        for i in range(iterations):
            with open(
                os.path.join(self.output_dir, f"Episode_{i}_by_event.txt"), 'w'
            ) as event_dump, open(
                os.path.join(self.output_dir, f"Episode_{i}_by_agent.txt"), 'w'
            ) as agent_dump:

                observations, actions, rewards, dones = self.generate_episode(
                    render=render,
                    log=event_dump,
                    **kwargs
                )
                agent_dump.write("Observations:\n")
                pprint(observations, stream=agent_dump)
                agent_dump.write("\nActions:\n")
                pprint(actions, stream=agent_dump)
                agent_dump.write("\nRewards:\n")
                pprint(rewards, stream=agent_dump)
                agent_dump.write("\nDones:\n")
                pprint(dones, stream=agent_dump)

        print(f"Files written to {self.output_dir}")
