
import argparse

from ray.rllib.env.policy_client import PolicyClient

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", type=str, default='CartPole-v0', help="The environment on which to train.",
    choices=["CartPole-v0", "PyCorridor", "CppCorridor"])
parser.add_argument(
    "--inference-mode", type=str, default="local", choices=["local", "remote"])
parser.add_argument(
    "--off-policy",
    action="store_true",
    help="Whether to take random instead of on-policy actions.")
parser.add_argument(
    "--stop-reward",
    type=int,
    default=9999,
    help="Stop once the specified reward is reached.")
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

    # simulation environment
    if args.env == "PyCorridor":
        from sim.simple_corridor import SimpleCorridor
        env = SimpleCorridor()
    elif args.env == "CppCorridor":
        from gym.spaces import Discrete, Box
        import numpy as np
        from build.simple_corridor import SimpleCorridor
        from sim.cpp_wrapper import CppWrapper
        env = CppWrapper(SimpleCorridor(), Discrete(2), Box(0.0, 5, shape=(1, ), dtype=np.float32))
    else:
        import gym
        env = gym.make(args.env)

    # policy client
    client = PolicyClient(ip_head, inference_mode=args.inference_mode)
    
    # Start data generation
    obs = env.reset()
    eid = client.start_episode(training_enabled=True)
    rewards = 0
    while True:
        if args.off_policy:
            action = env.action_space.sample()
            client.log_action(eid, obs, action)
        else:
            action = client.get_action(eid, obs)
        obs, reward, done, info = env.step(action)
        rewards += reward
        client.log_returns(eid, reward, info=info)
        if done:
            print("Total reward:", rewards)
            if rewards >= args.stop_reward:
                print("Target reward achieved, exiting")
                exit(0)
            rewards = 0
            client.end_episode(eid, obs)
            obs = env.reset()
            eid = client.start_episode(training_enabled=True)
