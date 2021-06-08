import numpy as np

from abmarl.managers import TurnBasedManager, AllStepManager
from abmarl.sim.corridor import MultiCorridor as Corridor


def test_managers_are_same_for_single_agent():
    turn_based_sim = TurnBasedManager(Corridor(num_agents=1))
    all_step_sim = AllStepManager(Corridor(num_agents=1))

    np.random.seed(5)
    turn_based_reset = turn_based_sim.reset()
    np.random.seed(5)
    all_step_reset = all_step_sim.reset()
    assert turn_based_reset == all_step_reset

    assert turn_based_sim.step({'agent0': 0}) == all_step_sim.step({'agent0': 0})
    assert turn_based_sim.step({'agent0': 2}) == all_step_sim.step({'agent0': 2})
    assert turn_based_sim.step({'agent0': 2}) == all_step_sim.step({'agent0': 2})
    assert turn_based_sim.step({'agent0': 1}) == all_step_sim.step({'agent0': 1})
    assert turn_based_sim.step({'agent0': 0}) == all_step_sim.step({'agent0': 0})
    assert turn_based_sim.step({'agent0': 2}) == all_step_sim.step({'agent0': 2})
    assert turn_based_sim.step({'agent0': 2}) == all_step_sim.step({'agent0': 2})
    assert turn_based_sim.step({'agent0': 2}) == all_step_sim.step({'agent0': 2})
    assert turn_based_sim.step({'agent0': 2}) == all_step_sim.step({'agent0': 2})
