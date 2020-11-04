from admiral.envs.predator_prey import PredatorPrey_v1

from getkey import getkey, keys

key_action_mapping = {
    keys.DOWN: 0,
    keys.LEFT: 1,
    keys.UP: 2,
    keys.RIGHT: 3,
    keys.ENTER: 4,
    keys.ESC: -1,
    keys.Q: -1
}

env = PredatorPrey_v1.build({'view': 4})

obs = env.reset()
env.render()
while True:
    action = key_action_mapping.get(getkey(), None)
    if action is not None:
        if action == -1:
            break
        next_obs, _, done, _ = env.step(action)
        env.render()
        obs = next_obs
        if done:
            break