from abmarl.sim.wrappers import SARWrapper
from .helpers import MultiAgentSim


# Individual wrappers
class ObservationWrapper(SARWrapper):
    def wrap_observation(self, from_agent, observation):
        return "Wrap Observation: " + observation

    def unwrap_observation(self, from_agent, observation):
        return "Unwrap Observation: " + observation


class ActionWrapper(SARWrapper):
    def wrap_action(self, from_agent, action):
        return "Wrap Action: " + str(action)

    def unwrap_action(self, from_agent, action):
        return "Unwrap Action: " + str(action)


class RewardWrapper(SARWrapper):
    def wrap_reward(self, reward):
        return "Wrap Reward: " + reward

    def unwrap_reward(self, reward):
        return "Unwrap Reward: " + reward


# Combination wrappers
class ObservationActionWrapper(ObservationWrapper, ActionWrapper):
    pass


class ObservationRewardWrapper(ObservationWrapper, RewardWrapper):
    pass


class ActionRewardWrapper(ActionWrapper, RewardWrapper):
    pass


class ObservationActionRewardWrapper(ObservationWrapper, ActionWrapper, RewardWrapper):
    pass


# Tests
def test_sar_wrapped():
    sim = MultiAgentSim(5)
    wrapped_sim = SARWrapper(sim)
    assert wrapped_sim.sim == sim
    assert wrapped_sim.unwrapped == sim
    assert wrapped_sim.agents == sim.agents


def test_wrap_observation():
    sim = ObservationWrapper(MultiAgentSim())
    sim.reset()
    assert sim.get_obs('agent0') == 'Wrap Observation: Obs from agent0'
    assert sim.get_obs('agent1') == 'Wrap Observation: Obs from agent1'
    assert sim.get_obs('agent2') == 'Wrap Observation: Obs from agent2'

    sim.step({agent_id: i for i, agent_id in enumerate(sim.agents)})
    assert sim.get_obs('agent0') == 'Wrap Observation: Obs from agent0'
    assert sim.get_obs('agent1') == 'Wrap Observation: Obs from agent1'
    assert sim.get_obs('agent2') == 'Wrap Observation: Obs from agent2'
    assert sim.get_reward('agent0') == 'Reward from agent0'
    assert sim.get_reward('agent1') == 'Reward from agent1'
    assert sim.get_reward('agent2') == 'Reward from agent2'
    assert sim.get_done('agent0') == 'Done from agent0'
    assert sim.get_done('agent1') == 'Done from agent1'
    assert sim.get_done('agent2') == 'Done from agent2'
    assert sim.get_all_done() == "Done from all agents and/or simulation."
    assert sim.get_info('agent0') == {'Action from agent0': 0}
    assert sim.get_info('agent1') == {'Action from agent1': 1}
    assert sim.get_info('agent2') == {'Action from agent2': 2}


def test_wrap_action():
    sim = ActionWrapper(MultiAgentSim())
    sim.reset()
    assert sim.get_obs('agent0') == 'Obs from agent0'
    assert sim.get_obs('agent1') == 'Obs from agent1'
    assert sim.get_obs('agent2') == 'Obs from agent2'

    sim.step({agent_id: i for i, agent_id in enumerate(sim.agents)})
    assert sim.get_obs('agent0') == 'Obs from agent0'
    assert sim.get_obs('agent1') == 'Obs from agent1'
    assert sim.get_obs('agent2') == 'Obs from agent2'
    assert sim.get_reward('agent0') == 'Reward from agent0'
    assert sim.get_reward('agent1') == 'Reward from agent1'
    assert sim.get_reward('agent2') == 'Reward from agent2'
    assert sim.get_done('agent0') == 'Done from agent0'
    assert sim.get_done('agent1') == 'Done from agent1'
    assert sim.get_done('agent2') == 'Done from agent2'
    assert sim.get_all_done() == "Done from all agents and/or simulation."
    assert sim.get_info('agent0') == {'Action from agent0': 'Wrap Action: 0'}
    assert sim.get_info('agent1') == {'Action from agent1': 'Wrap Action: 1'}
    assert sim.get_info('agent2') == {'Action from agent2': 'Wrap Action: 2'}


def test_wrap_reward():
    sim = RewardWrapper(MultiAgentSim())
    sim.reset()
    assert sim.get_obs('agent0') == 'Obs from agent0'
    assert sim.get_obs('agent1') == 'Obs from agent1'
    assert sim.get_obs('agent2') == 'Obs from agent2'

    sim.step({agent_id: i for i, agent_id in enumerate(sim.agents)})
    assert sim.get_obs('agent0') == 'Obs from agent0'
    assert sim.get_obs('agent1') == 'Obs from agent1'
    assert sim.get_obs('agent2') == 'Obs from agent2'
    assert sim.get_reward('agent0') == 'Wrap Reward: Reward from agent0'
    assert sim.get_reward('agent1') == 'Wrap Reward: Reward from agent1'
    assert sim.get_reward('agent2') == 'Wrap Reward: Reward from agent2'
    assert sim.get_done('agent0') == 'Done from agent0'
    assert sim.get_done('agent1') == 'Done from agent1'
    assert sim.get_done('agent2') == 'Done from agent2'
    assert sim.get_all_done() == "Done from all agents and/or simulation."
    assert sim.get_info('agent0') == {'Action from agent0': 0}
    assert sim.get_info('agent1') == {'Action from agent1': 1}
    assert sim.get_info('agent2') == {'Action from agent2': 2}


def test_wrap_observation_action():
    sim = ObservationActionWrapper(MultiAgentSim())
    sim.reset()
    assert sim.get_obs('agent0') == 'Wrap Observation: Obs from agent0'
    assert sim.get_obs('agent1') == 'Wrap Observation: Obs from agent1'
    assert sim.get_obs('agent2') == 'Wrap Observation: Obs from agent2'

    sim.step({agent_id: i for i, agent_id in enumerate(sim.agents)})
    assert sim.get_obs('agent0') == 'Wrap Observation: Obs from agent0'
    assert sim.get_obs('agent1') == 'Wrap Observation: Obs from agent1'
    assert sim.get_obs('agent2') == 'Wrap Observation: Obs from agent2'
    assert sim.get_reward('agent0') == 'Reward from agent0'
    assert sim.get_reward('agent1') == 'Reward from agent1'
    assert sim.get_reward('agent2') == 'Reward from agent2'
    assert sim.get_done('agent0') == 'Done from agent0'
    assert sim.get_done('agent1') == 'Done from agent1'
    assert sim.get_done('agent2') == 'Done from agent2'
    assert sim.get_all_done() == "Done from all agents and/or simulation."
    assert sim.get_info('agent0') == {'Action from agent0': 'Wrap Action: 0'}
    assert sim.get_info('agent1') == {'Action from agent1': 'Wrap Action: 1'}
    assert sim.get_info('agent2') == {'Action from agent2': 'Wrap Action: 2'}


def test_wrap_observation_reward():
    sim = ObservationRewardWrapper(MultiAgentSim())
    sim.reset()
    assert sim.get_obs('agent0') == 'Wrap Observation: Obs from agent0'
    assert sim.get_obs('agent1') == 'Wrap Observation: Obs from agent1'
    assert sim.get_obs('agent2') == 'Wrap Observation: Obs from agent2'

    sim.step({agent_id: i for i, agent_id in enumerate(sim.agents)})
    assert sim.get_obs('agent0') == 'Wrap Observation: Obs from agent0'
    assert sim.get_obs('agent1') == 'Wrap Observation: Obs from agent1'
    assert sim.get_obs('agent2') == 'Wrap Observation: Obs from agent2'
    assert sim.get_reward('agent0') == 'Wrap Reward: Reward from agent0'
    assert sim.get_reward('agent1') == 'Wrap Reward: Reward from agent1'
    assert sim.get_reward('agent2') == 'Wrap Reward: Reward from agent2'
    assert sim.get_done('agent0') == 'Done from agent0'
    assert sim.get_done('agent1') == 'Done from agent1'
    assert sim.get_done('agent2') == 'Done from agent2'
    assert sim.get_info('agent0') == {'Action from agent0': 0}
    assert sim.get_info('agent1') == {'Action from agent1': 1}
    assert sim.get_info('agent2') == {'Action from agent2': 2}


def test_wrap_action_reward():
    sim = ActionRewardWrapper(MultiAgentSim())
    sim.reset()
    assert sim.get_obs('agent0') == 'Obs from agent0'
    assert sim.get_obs('agent1') == 'Obs from agent1'
    assert sim.get_obs('agent2') == 'Obs from agent2'

    sim.step({agent_id: i for i, agent_id in enumerate(sim.agents)})
    assert sim.get_obs('agent0') == 'Obs from agent0'
    assert sim.get_obs('agent1') == 'Obs from agent1'
    assert sim.get_obs('agent2') == 'Obs from agent2'
    assert sim.get_reward('agent0') == 'Wrap Reward: Reward from agent0'
    assert sim.get_reward('agent1') == 'Wrap Reward: Reward from agent1'
    assert sim.get_reward('agent2') == 'Wrap Reward: Reward from agent2'
    assert sim.get_done('agent0') == 'Done from agent0'
    assert sim.get_done('agent1') == 'Done from agent1'
    assert sim.get_done('agent2') == 'Done from agent2'
    assert sim.get_info('agent0') == {'Action from agent0': 'Wrap Action: 0'}
    assert sim.get_info('agent1') == {'Action from agent1': 'Wrap Action: 1'}
    assert sim.get_info('agent2') == {'Action from agent2': 'Wrap Action: 2'}


def test_wrap_observation_action_reward():
    sim = ObservationActionRewardWrapper(MultiAgentSim())
    sim.reset()
    assert sim.get_obs('agent0') == 'Wrap Observation: Obs from agent0'
    assert sim.get_obs('agent1') == 'Wrap Observation: Obs from agent1'
    assert sim.get_obs('agent2') == 'Wrap Observation: Obs from agent2'

    sim.step({agent_id: i for i, agent_id in enumerate(sim.agents)})
    assert sim.get_obs('agent0') == 'Wrap Observation: Obs from agent0'
    assert sim.get_obs('agent1') == 'Wrap Observation: Obs from agent1'
    assert sim.get_obs('agent2') == 'Wrap Observation: Obs from agent2'
    assert sim.get_reward('agent0') == 'Wrap Reward: Reward from agent0'
    assert sim.get_reward('agent1') == 'Wrap Reward: Reward from agent1'
    assert sim.get_reward('agent2') == 'Wrap Reward: Reward from agent2'
    assert sim.get_done('agent0') == 'Done from agent0'
    assert sim.get_done('agent1') == 'Done from agent1'
    assert sim.get_done('agent2') == 'Done from agent2'
    assert sim.get_info('agent0') == {'Action from agent0': 'Wrap Action: 0'}
    assert sim.get_info('agent1') == {'Action from agent1': 'Wrap Action: 1'}
    assert sim.get_info('agent2') == {'Action from agent2': 'Wrap Action: 2'}
