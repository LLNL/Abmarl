def run(env, agent):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sim = env.unwrapped

    # Create a grid
    grid = np.zeros((sim.env.region, sim.env.region))
    attack = np.zeros((sim.env.region, sim.env.region))

    # Run the trained policy
    policy_agent_mapping = agent.config['multiagent']['policy_mapping_fn']
    for episode in range(100): # Run 100 trajectories
        print('Episode: {}'.format(episode))
        obs = sim.reset()
        done = {agent: False for agent in obs}
        pox, poy = sim.agents['predator0'].position
        grid[pox, poy] += 1
        while True:
            joint_action = {}
            for agent_id, agent_obs in obs.items():
                if done[agent_id]: continue # Don't get actions for dead agents
                policy_id = policy_agent_mapping(agent_id)
                action = agent.compute_action(agent_obs, policy_id=policy_id)
                joint_action[agent_id] = action
            obs, _, done, _ = sim.step(joint_action)
            pox, poy = sim.agents['predator0'].position
            grid[pox, poy] += 1
            if joint_action['predator0']['attack'] == 1: # This is the attack action
                attack[pox, poy] += 1
            if done['__all__']:
                break

    plt.figure(1)
    plt.title("Position concentration")
    sns.heatmap(np.flipud(np.transpose(grid)), linewidth=0.5)

    plt.figure(2)
    plt.title("Attack action frequency")
    sns.heatmap(np.flipud(np.transpose(attack)), linewidth=0.5)

    plt.show()
