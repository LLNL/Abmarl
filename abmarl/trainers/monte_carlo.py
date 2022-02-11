
from gym.spaces import Discrete
import numpy as np

from abmarl.trainers import SinglePolicyTrainer
from abmarl.tools import numpy_utils as npu

class OnPolicyMonteCarloTrainer(SinglePolicyTrainer):
    def train(self, iterations=10_000, gamma=0.9, **kwargs):
        state_action_returns = {}

        for i in range(iterations):
            states, actions, rewards = self.generate_episode(**kwargs)
            states = np.stack(states)
            actions = np.stack(actions)
            G = 0
            for i in reversed(range(len(states))):
                state, action, reward = states[i], actions[i], rewards[i]
                G = gamma * G + reward
                if not (npu.array_in_array(state, states[:i]) and
                        npu.array_in_array(action, actions[:i])):
                    if (state, action) not in state_action_returns:
                        state_action_returns[(state, action)] = [G]
                    else:
                        state_action_returns[(state, action)].append(G)
                    # TODO: It is very dumb that access the the policy must be
                    # through policies (or through _policy). We should modify
                    # the single agent trainer to also allow access through policy.
                    self.policies['policy'].q_table[state, action] = np.mean(state_action_returns[(state, action)])
                    # TODO: This assumes that the policy is a QTablePolicy, injecting
                    # unnecessary dependency. We need to change the policy interface
                    # to create an "update" function.

        return self.sim, self.policies['policy'].q_table, self.policies['policy']
