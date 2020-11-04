
from admiral.envs.corridor import Corridor
from admiral.managers import AllStepManager, TurnBasedManager
from admiral.algs import monte_carlo

env = AllStepManager(Corridor.build())
env, q_table, policy = monte_carlo.off_policy(env, iteration=100, horizon=10)

for i in range(5):
    print("Episode{}".format(i))
    obs = env.reset()
    env.unwrapped.render()
    for _ in range(10):
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
        env.unwrapped.render()
        if done:
            break
