

from abmarl.sim.gridworld.base import GridWorldAgent
from abmarl.sim.gridworld.smart import SmartGridWorldSimulation
from abmarl.sim.gridworld.agent import MovingAgent, GridObservingAgent
from abmarl.sim.gridworld.actor import MoveActor


class WallAgent(GridWorldAgent): pass


class TargetAgent(GridWorldAgent): pass


class TrafficAgent(MovingAgent, GridObservingAgent):
    def __init__(self, **kwargs):
        super().__init__(
            view_range=3,
            move_range=1,
            **kwargs
        )


class TrafficCorridorSimulation(SmartGridWorldSimulation):
    """
    Traffic Corridor Game.
    Agents must cross a corridor to get from their starting locations to their target
    locations. Agents cannot overlap each other, so they have to coordinate their movement
    through the corridor.

    Example grid:
    _ R W W W B _
    b _ _ _ _ _ r
    _ R W W W B _

    Red agents (R) must reach their target (r) right side, blue agents (B) must
    reach their traget (b) on the left, the corridor is only one cell wide, so they
    have to take turns.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.move_actor = MoveActor(**kwargs)
        self.finalize()

    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            move_result = self.move_actor.process_action(agent, action, **kwargs)
            if not move_result:
                self.rewards[agent.id] -= 0.1

            if self.get_done(agent_id):
                self.rewards[agent_id] += 1.0
