
from admiral.envs.predator_prey import PredatorPreyEnv, Predator, Prey
from admiral.envs.wrappers import CommunicationWrapper, RavelDiscreteWrapper
from admiral.managers import TurnBasedManager, AllStepManager

region = 6
predators = [Predator(id=f'predator{i}', attack=1) for i in range(2)]
prey = [Prey(id=f'prey{i}') for i in range(7)]
agents = predators + prey

env_config = {
    'region': region,
    'max_steps': 200,
    'observation_mode': PredatorPreyEnv.ObservationMode.DISTANCE,
    'agents': agents,
}
env = AllStepManager(CommunicationWrapper(PredatorPreyEnv.build(env_config)))
# env = TurnBasedManager(CommunicationWrapper(PredatorPreyEnv.build(env_config)))
obs = env.reset()
done = {agent_id: False for agent_id in env.agents}
env.render()
while True:
    action = {agent_id: env.agents[agent_id].action_space.sample() for agent_id in obs if not done[agent_id]}
    obs, reward, done, info = env.step(action)
    env.render()
    if done['__all__']:
        break
