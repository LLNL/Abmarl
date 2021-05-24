
import numpy as np

from admiral.managers import TurnBasedManager, AllStepManager
from admiral.envs.examples.corridor import MultiCorridor as Corridor

def test_managers_are_same_for_single_agent():
    turn_based_env = TurnBasedManager(Corridor(num_agents=1))
    all_step_env = AllStepManager(Corridor(num_agents=1))

    np.random.seed(5)
    turn_based_reset = turn_based_env.reset()
    np.random.seed(5)
    all_step_reset = all_step_env.reset()
    assert turn_based_reset == all_step_reset

    assert turn_based_env.step({'agent0': 0}) == all_step_env.step({'agent0': 0})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
    assert turn_based_env.step({'agent0': 1}) == all_step_env.step({'agent0': 1})
    assert turn_based_env.step({'agent0': 0}) == all_step_env.step({'agent0': 0})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
