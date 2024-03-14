
from abmarl.sim.gridworld.agent import MovingAgent, AttackingAgent, \
    GridObservingAgent, HealthAgent
from abmarl.sim.gridworld.smart import SmartGridWorldSimulation
from abmarl.sim.gridworld.actor import MoveActor, RestrictedSelectiveAttackActor


class ResourceAgent(HealthAgent):
    def __init__(
        self,
        encoding=1,
        render_shape='s',
        render_color='g',
        **kwargs
    ):
        super().__init__(
            encoding=encoding,
            render_shape=render_shape,
            render_color=render_color,
            **kwargs
        )


class PreyAgent(HealthAgent, MovingAgent, AttackingAgent, GridObservingAgent):
    def __init__(
        self,
        encoding=2,
        render_color='b',
        move_range=1,
        attack_range=1,
        attack_strength=1,
        attack_accuracy=1,
        view_range=3,
        **kwargs
    ):
        super().__init__(
            encoding=encoding,
            render_color=render_color,
            move_range=move_range,
            attack_range=attack_range,
            attack_strength=attack_strength,
            attack_accuracy=attack_accuracy,
            view_range=view_range,
            **kwargs
        )


class PredatorAgent(HealthAgent, MovingAgent, AttackingAgent, GridObservingAgent):
    def __init__(
        self,
        encoding=3,
        render_color='r',
        render_shape='d',
        move_range=1,
        attack_range=2,
        attack_strength=1,
        attack_accuracy=1,
        view_range=3,
        **kwargs
    ):
        super().__init__(
            encoding=encoding,
            render_color=render_color,
            render_shape=render_shape,
            move_range=move_range,
            attack_range=attack_range,
            attack_strength=attack_strength,
            attack_accuracy=attack_accuracy,
            view_range=view_range,
            **kwargs
        )


class PredatorPreyResourcesSim(SmartGridWorldSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.move_actor = MoveActor(**kwargs)
        self.attack_actor = RestrictedSelectiveAttackActor(**kwargs)

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
            if agent.active:
                move_result = self.move_actor.process_action(agent, action, **kwargs)
                if not move_result:
                    self.rewards[agent_id] -= 0.1
            if self.target_done.get_done(agent):
                self.rewards[agent_id] += 1
                self.grid.remove(agent, agent.position)
                agent.active = False

        # Entropy penalty
        for agent_id in action_dict:
                self.rewards[agent_id] -= 0.01        
