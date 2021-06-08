# This probably shouldn't go in algs, but we'll move it later after we've figured out the
# architecture a bit more.

def generate_episode(sim, policy, horizon=200):
    """
    Generate an episode from a policy acting on an simulation.

    Returns: sequence of state, action, reward.
    """
    obs = sim.reset()
    policy.reset() # Reset the policy too so that it knows its the beginning of the episode.
    states, actions, rewards = [], [], []
    states.append(obs)
    for _ in range(horizon):
        action = policy.act(obs)
        obs, reward, done, _ = sim.step(action)
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        if done:
            break

    states.pop() # Pop off the terminating state
    return states, actions, rewards
