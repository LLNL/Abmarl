
import numpy as np
import pytest

from abmarl.sim.gridworld.agent import MovingAgent
from abmarl.sim.gridworld.actor import MoveActor, SelectiveAttackActor
from abmarl.sim.gridworld.state import PositionState, HealthState
from abmarl.sim.gridworld.observer import AbsoluteEncodingObserver, PositionCenteredEncodingObserver
from abmarl.sim.gridworld.smart import SmartGridWorldSimulation
from abmarl.examples.sim.reach_the_target import ActiveDone, TargetDone, OnlyAgentLeftDone, \
    BarrierAgent, TargetAgent, RunningAgent


class SmartReachTheTarget(SmartGridWorldSimulation):
    def __init__(self, target=None, **kwargs):
        super().__init__(**kwargs)
        self.target = target

        # Action Components
        self.move_actor = MoveActor(**kwargs)
        self.attack_actor = SelectiveAttackActor(**kwargs)

        # Done components
        self.active_done = ActiveDone(**kwargs)
        self.target_done = TargetDone(target=self.target, **kwargs)
        self.only_agent_done = OnlyAgentLeftDone(**kwargs)

        self.finalize()

    def step(self, action_dict, **kwargs):
        # Process the attacks
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.active:
                attack_status, attacked_agents = \
                    self.attack_actor.process_action(agent, action, **kwargs)
                if attack_status: # Attack was attempted
                    if not attacked_agents: # Attack failed
                        self.rewards[agent_id] -= 0.1
                    else:
                        for attacked_agent in attacked_agents:
                            if not attacked_agent.active: # Agent has died
                                self.rewards[attacked_agent.id] -= 1
                                self.rewards[agent_id] += 1

        # Process the moves
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if isinstance(agent, MovingAgent):
                if agent.active:
                    move_result = self.move_actor.process_action(agent, action, **kwargs)
                    if not move_result:
                        self.rewards[agent_id] -= 0.1
                if self.target_done.get_done(agent):
                    self.rewards[agent_id] += 1
                    self.grid.remove(agent, agent.position)
                    agent.active = False

        # Entropy penalty for the runners
        for agent_id in action_dict:
            if isinstance(self.agents[agent_id], RunningAgent):
                self.rewards[agent_id] -= 0.01


    def get_done(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        if isinstance(agent, RunningAgent):
            return self.active_done.get_done(agent, **kwargs) \
                or self.target_done.get_done(agent, **kwargs)
        elif isinstance(agent, TargetAgent):
            return self.only_agent_done.get_done(agent, **kwargs)

    def get_all_done(self, **kwargs):
        return self.only_agent_done.get_all_done(**kwargs)


def test_smart_sim_components():
    grid_size = 7
    corners = [
        np.array([0, 0], dtype=int),
        np.array([grid_size - 1, 0], dtype=int),
        np.array([0, grid_size - 1], dtype=int),
        np.array([grid_size - 1, grid_size - 1], dtype=int),
    ]
    agents = {
        **{
            f'barrier{i}': BarrierAgent(
                id=f'barrier{i}'
            ) for i in range(10)
        },
        **{
            f'runner{i}': RunningAgent(
                id=f'runner{i}',
                move_range=2,
                view_range=int(grid_size / 2),
                initial_health=1,
                initial_position=corners[i]
            ) for i in range(4)
        },
        'target': TargetAgent(
            view_range=grid_size,
            attack_range=1,
            attack_strength=1,
            attack_accuracy=1,
            initial_position=np.array([int(grid_size / 2), int(grid_size / 2)], dtype=int)
        )
    }
    overlapping = {
        2: {3},
        3: {1, 2, 3}
    }
    attack_mapping = {
        2: {3}
    }

    sim = SmartReachTheTarget.build_sim(
        grid_size, grid_size,
        agents=agents,
        target=agents['target'],
        states={PositionState},
        observers={AbsoluteEncodingObserver},
        overlapping=overlapping,
        attack_mapping=attack_mapping
    )
    assert len(sim._states) == 1
    assert isinstance(next(iter(sim._states)), PositionState)
    assert len(sim._observers) == 1
    assert isinstance(next(iter(sim._observers)), AbsoluteEncodingObserver)

    sim = SmartReachTheTarget.build_sim(
        grid_size, grid_size,
        agents=agents,
        target=agents['target'],
        states={'PositionState'},
        observers={'AbsoluteEncodingObserver'},
        overlapping=overlapping,
        attack_mapping=attack_mapping
    )
    assert len(sim._states) == 1
    assert isinstance(next(iter(sim._states)), PositionState)
    assert len(sim._observers) == 1
    assert isinstance(next(iter(sim._observers)), AbsoluteEncodingObserver)

    sim = SmartReachTheTarget.build_sim(
        grid_size, grid_size,
        agents=agents,
        target=agents['target'],
        states={HealthState},
        observers={PositionCenteredEncodingObserver},
        overlapping=overlapping,
        attack_mapping=attack_mapping
    )
    assert len(sim._states) == 1
    assert isinstance(next(iter(sim._states)), HealthState)
    assert len(sim._observers) == 1
    assert isinstance(next(iter(sim._observers)), PositionCenteredEncodingObserver)

    sim = SmartReachTheTarget.build_sim(
        grid_size, grid_size,
        agents=agents,
        target=agents['target'],
        states={'HealthState'},
        observers={'PositionCenteredEncodingObserver'},
        overlapping=overlapping,
        attack_mapping=attack_mapping
    )
    assert len(sim._states) == 1
    assert isinstance(next(iter(sim._states)), HealthState)
    assert len(sim._observers) == 1
    assert isinstance(next(iter(sim._observers)), PositionCenteredEncodingObserver)

    sim = SmartReachTheTarget.build_sim(
        grid_size, grid_size,
        agents=agents,
        target=agents['target'],
        states={PositionState, HealthState},
        observers={AbsoluteEncodingObserver, PositionCenteredEncodingObserver},
        overlapping=overlapping,
        attack_mapping=attack_mapping
    )
    assert len(sim._states) == 2
    for state in sim._states:
        assert isinstance(state, (PositionState, HealthState))
    assert len(sim._observers) == 2
    for observer in sim._observers:
        assert isinstance(observer, (AbsoluteEncodingObserver, PositionCenteredEncodingObserver))

    sim = SmartReachTheTarget.build_sim(
        grid_size, grid_size,
        agents=agents,
        target=agents['target'],
        states={'PositionState', HealthState},
        observers={AbsoluteEncodingObserver, 'PositionCenteredEncodingObserver'},
        overlapping=overlapping,
        attack_mapping=attack_mapping
    )
    assert len(sim._states) == 2
    for state in sim._states:
        assert isinstance(state, (PositionState, HealthState))
    assert len(sim._observers) == 2
    for observer in sim._observers:
        assert isinstance(observer, (AbsoluteEncodingObserver, PositionCenteredEncodingObserver))

    sim = SmartReachTheTarget.build_sim(
        grid_size, grid_size,
        agents=agents,
        target=agents['target'],
        states={'PositionState', 'HealthState'},
        observers={'AbsoluteEncodingObserver', 'PositionCenteredEncodingObserver'},
        overlapping=overlapping,
        attack_mapping=attack_mapping
    )
    assert len(sim._states) == 2
    for state in sim._states:
        assert isinstance(state, (PositionState, HealthState))
    assert len(sim._observers) == 2
    for observer in sim._observers:
        assert isinstance(observer, (AbsoluteEncodingObserver, PositionCenteredEncodingObserver))

    with pytest.raises(AssertionError):
        sim = SmartReachTheTarget.build_sim(
            grid_size, grid_size,
            agents=agents,
            target=agents['target'],
            observers={'AbsoluteEncodingObserver', 'HealthState'},
            states={'PositionState', 'PositionCenteredEncodingObserver'},
            overlapping=overlapping,
            attack_mapping=attack_mapping
        )

    with pytest.raises(AssertionError):
        sim = SmartReachTheTarget.build_sim(
            grid_size, grid_size,
            agents=agents,
            target=agents['target'],
            observers={'AbsoluteEncodingObserver', 'PositionCenteredEncodingObserver'},
            overlapping=overlapping,
            attack_mapping=attack_mapping
        )
        sim.reset()

    with pytest.raises(AssertionError):
        sim = SmartReachTheTarget.build_sim(
            grid_size, grid_size,
            agents=agents,
            target=agents['target'],
            states={'PositionState', 'HealthState'},
            overlapping=overlapping,
            attack_mapping=attack_mapping
        )
        sim.get_obs('target')
