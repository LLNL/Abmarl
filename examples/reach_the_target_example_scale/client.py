
import argparse

from ray.rllib.env.policy_client import PolicyClient

from abmarl.tools import utils as adu

parser = argparse.ArgumentParser()
parser.add_argument(
    '--server_address',
    type=str,
    default='localhost',
    help='The ip address of the remote server.'
)
parser.add_argument(
    '--port',
    type=int,
    default=9900,
    help='The client should connect to this server port.'
)
parser.add_argument(
    "--inference-mode", type=str, default="local", choices=["local", "remote"])


if __name__ == "__main__":
    args = parser.parse_args()

    # Get the simulation from the configuration file
    experiment_mod = adu.custom_import_module('/Users/rusu1/Abmarl/examples/reach_the_target_example.py')
    sim = experiment_mod.sim

    # policy client
    client = PolicyClient(
        f"http://{args.server_address}:{args.port}",
        inference_mode=args.inference_mode
    )

    # Start data generation
    for i in range(experiment_mod.params['ray_tune']['stop']['episodes_total']):
        eid = client.start_episode(training_enabled=True)
        obs = sim.reset()
        done = {agent: False for agent in obs}
        for j in range(experiment_mod.params['ray_tune']['config']['horizon']):
            action_t = client.get_action(eid, obs)
            action_dict = {
                agent_id: action
                for agent_id, action in action_t.items() if not done[agent_id]
            }
            obs, reward, done, info = sim.step(action_dict)
            client.log_returns(eid, reward, multiagent_done_dict=done, info=info)
            if done['__all__']:
                break
        client.end_episode(eid, obs)
