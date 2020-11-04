
def run(env, agent):
    """
    Analyze the behavior of your trained policies using the environment and agent from your RL
    experiment.
    """

    # Run the simulation with actions chosen from the trained policies
    policy_agent_mapping = agent.config['multiagent']['policy_mapping_fn']
    for episode in range(num_episodes):
        print('Episode: {}'.format(episode))
        obs = env.reset()
        while True: # Run until the episode ends
            # Get actions from policies
            joint_action = {}
            for agent_id, agent_obs in obs.items():
                policy_id = policy_agent_mapping(agent_id)
                action = agent.compute_action(agent_obs, policy_id=policy_id)
                joint_action[agent_id] = action
            # Step the environment
            obs, reward, done, info = env.step(joint_action)
            if done['__all__']:
                break
