
from gym.spaces import Discrete
import numpy as np

from abmarl.trainers import SinglePolicyTrainer
from abmarl.tools import numpy_utils as npu

class OnPolicyMonteCarloTrainer(SinglePolicyTrainer):
    def train(self, iterations=10_000, gamma=0.9, **kwargs):
        # TODO: The creation of the q-table should happen with the policy, not
        # here in train.
        obs_space = self.policies['policy'].observation_space
        act_space = self.policies['policy'].action_space
        assert isinstance(obs_space, Discrete)
        assert isinstance(act_space, Discrete)
        q_table = np.random.normal(0, 1, size=(obs_space.n, act_space.n))
        policy = RandomFirstActionPolicy(q_table)
        # END todo.
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
                    q_table[state, action] = np.mean(state_action_returns[(state, action)])

        return self.sim, q_table, policy

