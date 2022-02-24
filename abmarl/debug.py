from abmarl.tools import utils as adu


def run(full_config_path, parameters):
    """Debug the SimulationManagers from the config_file."""

    # Load the experiment as a module
    experiment_mod = adu.custom_import_module(full_config_path)
    title = "DEBUG_" + experiment_mod.params['experiment']['title']

    # Copy the configuration module to the output directory
    import os
    import shutil
    import time
    base = experiment_mod.params['ray_tune'].get('local_dir', os.path.expanduser("~"))
    output_dir = os.path.join(
        base, 'abmarl_results/{}_{}'.format(
            title, time.strftime('%Y-%m-%d_%H-%M')
        )
    )
    experiment_mod.params['ray_tune']['local_dir'] = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(full_config_path, output_dir)

    # Simulation loop
    from pprint import pprint
    if parameters.render:
        from matplotlib import pyplot as plt
    sim = experiment_mod.params['experiment']['sim_creator'](
        experiment_mod.params['ray_tune']['config']['env_config']
    )
    agents = sim.unwrapped.agents
    for i in range(parameters.episodes):
        # Setup dump files
        with open(os.path.join(output_dir, f"Episode_{i}.txt"), 'w') as debug_dump:
            if parameters.render:
                fig = plt.figure()
            obs = sim.reset()
            done = {agent: False for agent in obs}
            if parameters.render:
                sim.render(fig=fig)
                plt.pause(1e-16)
            debug_dump.write("Reset:\n")
            pprint(obs, stream=debug_dump)
            for j in range(parameters.steps_per_episode): # Data generation
                action = {
                    agent_id: agents[agent_id].action_space.sample()
                    for agent_id in obs if not done[agent_id]
                }
                obs, reward, done, info = sim.step(action)
                if parameters.render:
                    sim.render(fig=fig)
                    plt.pause(1e-16)
                debug_dump.write(f"\nStep {j}:\n")
                pprint(action, stream=debug_dump)
                pprint(obs, stream=debug_dump)
                pprint(reward, stream=debug_dump)
                pprint(done, stream=debug_dump)
                if done['__all__']:
                    break
            if parameters.render:
                plt.close(fig)
