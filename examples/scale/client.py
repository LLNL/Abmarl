
import argparse

from ray.rllib.env.policy_client import PolicyClient

parser = argparse.ArgumentParser()
parser.add_argument(
    "--inference-mode", type=str, default="local", choices=["local", "remote"])
parser.add_argument(
    "--ip-head",
    type=str,
    default='localhost',
    help="The ip address of the remote server."
)

if __name__ == "__main__":
    args = parser.parse_args()

    # server's address
    address = args.ip_head
    port = 9900
    ip_head = 'http://' + address + ":" + str(port)

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
    done = {agent: False for agent in obs}
    while True:
        action_t = client.get_action(eid, obs)
        action_dict = {
            agent_id: action
            for agent_id, action in action_t.items() if not done[agent_id]
        }
        obs, reward, done, info = env.step(action_dict)
        client.log_returns(eid, reward, info=info)
        if done['__all__']:
            client.end_episode(eid, obs)
            obs = env.reset()
            done = {agent: False for agent in obs}
            eid = client.start_episode(training_enabled=True)
