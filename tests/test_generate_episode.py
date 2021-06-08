from abmarl.algs.generate_episode import generate_episode


class Sim:
    def reset(self):
        self.count = 0
        return self.count

    def step(self, action):
        self.count = action
        done = True if abs(self.count) > 100 else False
        return self.count, action, done, {}


class Policy:
    def reset(self):
        self.first_guess = True

    def act(self, obs):
        action = obs + 1 if self.first_guess else -2 * obs
        self.first_guess = False
        return action


def test_generate_episode():
    states, actions, rewards = generate_episode(Sim(), Policy())
    assert states == [0, 1, -2, 4, -8, 16, -32, 64]
    assert actions == [1, -2, 4, -8, 16, -32, 64, -128]
    assert rewards == [1, -2, 4, -8, 16, -32, 64, -128]
