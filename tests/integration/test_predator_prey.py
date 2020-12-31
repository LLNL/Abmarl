
from gym.spaces import Box, Dict, MultiBinary
import numpy as np

from admiral.envs.components.examples.predator_prey_example import PredatorPreyEnv, PredatorAgent, PreyAgent
from admiral.managers import AllStepManager

agents = {
    'prey0': PreyAgent(id='prey0', starting_position=np.array([2, 2]), position_view_range=4, initial_health=0.5, team=0, move_range=1, max_harvest=0.5, resource_view_range=4),
    'prey1': PreyAgent(id='prey1', starting_position=np.array([2, 2]), position_view_range=4, initial_health=0.5, team=0, move_range=1, max_harvest=0.5, resource_view_range=4),
    'predator0': PredatorAgent(id='predator0', starting_position=np.array([0, 0]), position_view_range=2, initial_health=0.5, team=1, move_range=1, attack_range=1, attack_strength=0.24)
}
original_resources = np.array([
    [0.43384617, 0.        , 0.        , 0.36753862, 0.3241253 ],
    [0.84682462, 0.34216225, 0.46882695, 0.23949859, 0.64573111],
    [0.86477947, 0.61520966, 0.449564  , 0.97582266, 0.26494157],
    [0.        , 0.        , 0.89888709, 0.        , 0.        ],
    [0.        , 0.17875298, 0.97128372, 0.        , 0.94036929]
])
env = AllStepManager(PredatorPreyEnv(
    region=5,
    agents=agents,
    number_of_teams=2,
    entropy=0.1,
    original_resources=original_resources,
))

def test_env_init():
    unwrapped_env = env.unwrapped
    # Assertions on state handlers
    assert unwrapped_env.agents == agents
    assert unwrapped_env.position_state.agents == agents
    assert unwrapped_env.position_state.region == 5
    assert unwrapped_env.life_state.agents == agents
    assert unwrapped_env.life_state.entropy == 0.1
    assert unwrapped_env.resource_state.agents == agents
    assert np.allclose(unwrapped_env.resource_state.original_resources, original_resources)
    assert unwrapped_env.resource_state.region == 5
    assert unwrapped_env.resource_state.min_value == 0.1
    assert unwrapped_env.resource_state.max_value == 1.0
    assert unwrapped_env.resource_state.regrow_rate == 0.04
    assert unwrapped_env.resource_state.coverage == 0.75
    assert unwrapped_env.team_state.number_of_teams == 2

    # Assertions on observation handlers
    assert unwrapped_env.position_observer.position == unwrapped_env.position_state
    assert unwrapped_env.position_observer.agents == agents
    for agent in agents.values():
        assert agent.observation_space['position'] == Dict({
            other.id: Box(0, 5, (2,), np.int) for other in agents.values()
        }) # TODO: Use a different position observer
    assert unwrapped_env.resource_observer.resources == unwrapped_env.resource_state
    assert unwrapped_env.resource_observer.agents == agents
    for agent in agents.values():
        if isinstance(agent, PreyAgent):
            assert agent.observation_space['resources'] == Box(0, 1.0, (9, 9), np.float)
    assert unwrapped_env.health_observer.agents == agents
    for agent in agents.values():
        assert agent.observation_space['health'] == Dict({
            other.id: Box(0, 1.0, (1,), np.float) for other in agents.values()
        })
    assert unwrapped_env.life_observer.agents == agents
    for agent in agents.values():
        assert agent.observation_space['life'] == Dict({
            other.id: Box(0, 1, (1,), np.int) for other in agents.values()
        })
    assert unwrapped_env.team_observer.team == unwrapped_env.team_state
    assert unwrapped_env.team_observer.agents == agents
    for agent in agents.values():
        assert agent.observation_space['team'] == Dict({
            other.id: Box(0, 2, (1,), np.int) for other in agents.values()
        })
    
    # Assertions on actor handlers
    assert unwrapped_env.move_actor.position == unwrapped_env.position_state
    assert unwrapped_env.move_actor.agents == agents
    for agent in agents.values():
        assert agent.action_space == Box(-agent.move_range, agent.move_range, (2,), np.int)
    assert unwrapped_env.resource_actor.resources == unwrapped_env.resource_state
    assert unwrapped_env.resource_actor.agents == agents
    for agent in agents.values():
        if isinstance(agent, PreyAgent):
            assert agent.action_space['harvest'] == Box(0, agent.max_harvest, (1,), np.float)
    assert unwrapped_env.attack_actor.agents == agents
    for agent in agent.values():
        if isinstance(agent, PredatorAgent):
            assert agent.action_space['attack'] == MultiBinary(1)
    
    # Assertions on done handlers
    # TODO: team dead done should take the team state manager as input instead of number of teams
    assert unwrapped_env.done.agents == agents



    env.reset()
    print({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
    fig = plt.gcf()
    env.render(fig=fig)

    for _ in range(50):
        action_dict = {agent.id: agent.action_space.sample() for agent in env.agents.values() if agent.is_alive}
        env.step(action_dict)
        env.render(fig=fig)
        print(env.get_all_done())
        x = []

test_integrated_environment()

