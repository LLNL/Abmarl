
import argparse

from ray.rllib.env.policy_client import PolicyClient

parser = argparse.ArgumentParser()
parser.add_argument(
    "--inference-mode", type=str, default="local", choices=["local", "remote"])
parser.add_argument(
    "--ip-head",
    type=str,
    default='localhost:9900',
    help="The ip address and port to connect to on the server. This should match the ip_head " \
        "given to the server node, and the port can be incremented if there are multiple " \
        "workers listening on the server."
)

if __name__ == "__main__":
    args = parser.parse_args()

    # server's address
    address, port = args.ip_head.split(':')
    port = 9900
    ip_head = 'http://' + address + ":" + str(port)
    # TODO: Process head node ip without splitting the port

    # simulation environment
    from abmarl.sim.corridor import MultiCorridor
    from abmarl.managers import TurnBasedManager
    from abmarl.external import MultiAgentWrapper
    env = MultiAgentWrapper(TurnBasedManager(MultiCorridor()))

    # policy client
    client = PolicyClient(ip_head, inference_mode=args.inference_mode)
    
    # Start data generation
    obs = env.reset()
    eid = client.start_episode(training_enabled=True)
    while True:
        action = client.get_action(eid, obs)
        obs, reward, done, info = env.step(action)
        client.log_returns(eid, reward, info=info)
        if done['__all__']:
            client.end_episode(eid, obs)
            obs = env.reset()
            eid = client.start_episode(training_enabled=True)
