def run(env, trainer):
    """
    Analyze the behavior of your trained policies using the environment and agent from your RL
    experiment. The environment is likely wrapped by the MultiAgentWrapper; you
    can use the unwrapped property to get the Simulation Manager.
    """

    env = env.unwrapped

    # Run the simulation with actions chosen from the trained policies
    policy_agent_mapping = trainer.config['multiagent']['policy_mapping_fn']
    # for episode in range(num_episodes):
    for episode in range(100):
        print('Episode: {}'.format(episode))
        obs = env.reset()
        while True: # Run until the episode ends
            # Get actions from policies
            joint_action = {}
            for agent_id, agent_obs in obs.items():
                policy_id = policy_agent_mapping(agent_id)
                action = trainer.compute_action(agent_obs, policy_id=policy_id)
                joint_action[agent_id] = action
            # Step the environment
            obs, reward, done, info = env.step(joint_action)
            if done['__all__']:
                break
