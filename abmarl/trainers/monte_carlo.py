
import numpy as np

from abmarl.trainers import SinglePolicyTrainer
from abmarl.tools import numpy_utils as npu


class OnPolicyMonteCarloTrainer(SinglePolicyTrainer):
    def train(self, iterations=10_000, gamma=0.9, **kwargs):
        """
        Implements on-policy monte carlo.
        """
        state_action_returns = {}

        for i in range(iterations):
            states, actions, rewards, _ = self.generate_episode(**kwargs)
            states = next(iter(states.values()))
            states.pop() # Pop off the terminating state.
            states = np.stack(states)
            actions = np.stack(next(iter(actions.values())))
            rewards = next(iter(rewards.values()))
            G = 0
            for i in reversed(range(len(states))):
                state, action, reward = states[i], actions[i], rewards[i]
                G = gamma * G + reward
                if not npu.array_in_array(state, states[:i]):
                    if not (npu.array_in_array(action, actions[:i])):
                        try:
                            state_action_returns[(state, action)].append(G)
                        except KeyError:
                            state_action_returns[(state, action)] = [G]
                        self.policy.update(
                            state, action, np.mean(state_action_returns[(state, action)])
                        )
