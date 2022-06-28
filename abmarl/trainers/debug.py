
import os
from pprint import pprint

from matplotlib import pyplot as plt

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
        # Simulation and agents controller
        for i in range(iterations):
            self.generate_episode(render=render, **kwargs)

            # Setup dump files
            with open(os.path.join(output_dir, f"Episode_{i}.txt"), 'w') as debug_dump:
                obs = sim.reset()
                done = {agent: False for agent in obs}
                debug_dump.write("Reset:\n")
                pprint(obs, stream=debug_dump)
                for j in range(parameters.steps_per_episode): # Data generation
                    action = {
                        agent_id: agents[agent_id].action_space.sample()
                        for agent_id in obs if not done[agent_id]
                    }
                    obs, reward, done, info = sim.step(action)
                    debug_dump.write(f"\nStep {j}:\n")
                    pprint(action, stream=debug_dump)
                    pprint(obs, stream=debug_dump)
                    pprint(reward, stream=debug_dump)
                    pprint(done, stream=debug_dump)
                    if done['__all__']:
                        break
