
import os

import ray

from admiral.tools import utils as adu

def _start():
    """The elements that are common to both analyze and visualize."""
    pass

def _finish():
    """Finish off the evaluation run"""
    ray.shutdown()

def run_analysis(full_trained_directory, full_subscript, parameters):
    """Analyze MARL policies from a saved policy through an analysis script"""
    env, agent = adu.extract_env_and_agents_from_experiment(full_trained_directory, parameters.checkpoint)

    # Load the analysis module and run it
    analysis_mod = adu.custom_import_module(full_subscript)
    analysis_mod.run(env.unwrapped, agent)

def run_visualize(full_trained_directory, parameters):
    """Visualize MARL policies from a saved policy"""
    env, agent = adu.extract_env_and_agents_from_experiment(full_trained_directory, parameters.checkpoint)

    # Determine if we are single- or multi-agent case.
    def _multi_get_action(obs, done=None, env=None, policy_agent_mapping=None, **kwargs):
        joint_action = {}
        if done is None:
            done = {agent: False for agent in obs}
        for agent_id, agent_obs in obs.items():
            if done[agent_id]: continue # Don't get actions for done agents
            policy_id = policy_agent_mapping(agent_id)
            action = agent.compute_action(agent_obs, policy_id=policy_id, \
                explore=parameters.no_explore)
            joint_action[agent_id] = action
        return joint_action
    
    def _single_get_action(obs, agent=None, **kwargs):
        return agent.compute_action(obs, explore=parameters.no_explore)

    def _multi_get_done(done):
        return done['__all__']
    
    def _single_get_done(done):
        return done
    
    policy_agent_mapping = None
    from ray.rllib.env import MultiAgentEnv
    if isinstance(env, MultiAgentEnv):
        policy_agent_mapping = agent.config['multiagent']['policy_mapping_fn']
        _get_action = _multi_get_action
        _get_done = _multi_get_done
    else:
        _get_action = _single_get_action
        _get_done = _single_get_done

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    for episode in range(parameters.episodes):
        print('Episode: {}'.format(episode))
        obs = env.reset()
        done = None
        all_done = False
        fig = plt.figure()

        def gen_frame_until_done():
            nonlocal all_done
            i = 0
            while not all_done:
                i += 1
                yield i

        def animate(i):
            nonlocal obs, done
            env.render(fig=fig)
            plt.pause(1e-16)
            action = _get_action(obs, done=done, env=env, agent=agent, policy_agent_mapping=policy_agent_mapping)
            obs, _, done, _ = env.step(action)
            if _get_done(done):
                nonlocal all_done
                all_done = True
                env.render(fig=fig)
                plt.pause(1e-16)
                plt.close(fig)

        anim = FuncAnimation(fig, animate, frames=gen_frame_until_done, repeat=False, \
            interval=parameters.frame_delay)
        if parameters.record:
            anim.save(os.path.join(full_trained_directory, 'Episode_{}.mp4'.format(episode)))
        plt.show(block=False)
        while all_done is False:
            plt.pause(1)
