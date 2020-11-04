
from admiral.managers import TurnBasedManager, AllStepManager
from admiral.envs.corridor import Corridor

def test_managers_are_same_for_single_agent():
    turn_based_env = TurnBasedManager(Corridor.build())
    all_step_env = AllStepManager(Corridor.build())

    assert turn_based_env.reset() == all_step_env.reset()

    assert turn_based_env.step({'agent0': 0}) == all_step_env.step({'agent0': 0})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
    assert turn_based_env.step({'agent0': 1}) == all_step_env.step({'agent0': 1})
    assert turn_based_env.step({'agent0': 0}) == all_step_env.step({'agent0': 0})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
    assert turn_based_env.step({'agent0': 2}) == all_step_env.step({'agent0': 2})
