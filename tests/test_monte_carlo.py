
from admiral.algs.monte_carlo import exploring_starts, epsilon_soft, off_policy
from admiral.envs.corridor import MultiCorridor as Corridor
from admiral.managers import AllStepManager
from admiral.envs.wrappers import RavelDiscreteWrapper
from admiral.external import GymWrapper
from admiral.pols import RandomFirstActionPolicy, EpsilonSoftPolicy, GreedyPolicy

def test_exploring_starts_corridor():
    env = AllStepManager(RavelDiscreteWrapper(Corridor(num_agents=1)))
    env, q_table, policy = exploring_starts(env, iteration=100, horizon=10)
    
    assert isinstance(env, GymWrapper)
    assert isinstance(env.env, AllStepManager)
    assert isinstance(env.env.env, RavelDiscreteWrapper)
    assert isinstance(env.env.env.env, Corridor)

    assert q_table.shape == (env.observation_space.n, env.action_space.n)
    assert isinstance(policy, RandomFirstActionPolicy)

def test_epsilon_soft():
    env = AllStepManager(RavelDiscreteWrapper(Corridor(num_agents=1)))
    env, q_table, policy = epsilon_soft(env, iteration=1000, horizon=20)
    
    assert isinstance(env, GymWrapper)
    assert isinstance(env.env, AllStepManager)
    assert isinstance(env.env.env, RavelDiscreteWrapper)
    assert isinstance(env.env.env.env, Corridor)

    assert q_table.shape == (env.observation_space.n, env.action_space.n)
    assert isinstance(policy, EpsilonSoftPolicy)

    obs = env.reset()
    for _ in range(10):
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    assert done

def test_off_policy():
    env = AllStepManager(RavelDiscreteWrapper(Corridor(num_agents=1)))
    env, q_table, policy = off_policy(env, iteration=100, horizon=10)
    
    assert isinstance(env, GymWrapper)
    assert isinstance(env.env, AllStepManager)
    assert isinstance(env.env.env, RavelDiscreteWrapper)
    assert isinstance(env.env.env.env, Corridor)

    assert q_table.shape == (env.observation_space.n, env.action_space.n)
    assert isinstance(policy, GreedyPolicy)

    obs = env.reset()
    for _ in range(10):
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    assert done
