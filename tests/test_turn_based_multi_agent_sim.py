
from abmarl.examples import MultiAgentGymSpacesSim
from abmarl.managers import TurnBasedManager


def test_only_learning_agents_in_output():
    sim = MultiAgentGymSpacesSim()
    sim.dones = [1] * 4
    wrapped_sim = TurnBasedManager(sim)
    obs = wrapped_sim.reset()
    obs, reward, done, info = wrapped_sim.step({
        agent_id: sim.agents[agent_id].action_space.sample()
        for agent_id in obs
    })
    assert 'agent4' not in obs
    assert 'agent4' not in reward
    assert 'agent4' not in done
    assert 'agent4' not in info
