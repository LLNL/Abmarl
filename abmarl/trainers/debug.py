
import os
from pprint import pprint

from abmarl.trainers.base import MultiPolicyTrainer

class DebugTrainer(MultiPolicyTrainer):
    """
    Debug the training setup.
    """
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
            with open(os.path.join(output_dir, f"Episode_{i}.txt"), 'w') as debug_dump:
                debug_dump.write("Observations:\n")
                pprint(observations, stream=debug_dump)
                debug_dump.write("\nActions:\n")
                pprint(actions, stream=debug_dump)
                debug_dump.write("\nRewards:\n")
                pprint(rewards, stream=debug_dump)
                debug_dump.write("\nDones:\n")
                pprint(dones, stream=debug_dump)
