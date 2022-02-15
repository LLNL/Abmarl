
import numpy as np

from abmarl.trainers import SinglePolicyTrainer
from abmarl.tools import numpy_utils as npu

class OnPolicyMonteCarloTrainer(SinglePolicyTrainer):
    def train(self, iterations=10_000, gamma=0.9, **kwargs):
        state_action_returns = {}

        for i in range(iterations):
            states, actions, rewards = self.generate_episode(**kwargs)
            # TODO: Here we assume that there is not only a single policy but that
            # there is also a single agent. We do this because we need to figure
            # out how to concatenate the data from multiple agents training the
            # same policy. Until then, we'll just use the first agent's experiences.
            states = next(iter(states.values()))
            states.pop() # Pop off the terminating state.
            # TODO: How does terminating state pop off work when there are multiple
            # agents?
            states = np.stack(states)
            actions = np.stack(next(iter(actions.values())))
            rewards = next(iter(rewards.values()))
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
                    self.policy.q_table[state, action] = np.mean(state_action_returns[(state, action)])
                    # TODO: This assumes that the policy is a QTablePolicy, injecting
                    # unnecessary dependency. We need to change the policy interface
                    # to create an "update" function.

        return self.sim, self.policy.q_table, self.policy
        # TODO: Why return these?
