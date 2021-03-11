
from gym.spaces import Dict, Discrete

from admiral.envs import SimpleAgent

def test_sample_from_spaces():
    simple_agent = SimpleAgent(
        id=12,
        action_space={1: Discrete(12), 2: Discrete(30)},
        observation_space={1: Discrete(100), 2: Dict({'a': Discrete(50), 'b': Discrete(16)})}
    )
    simple_agent.finalize()
    assert simple_agent.action_space.sample() == {1: 6, 2: 22}
    assert simple_agent.observation_space.sample() == {1: 21, 2: {'a': 21, 'b': 6}}
