
from abmarl.sim.gridworld.smart import SmartGridWorldSimulation
from abmarl.sim.gridworld.actor import MoveActor, RestrictedSelectiveAttackActor
from abmarl.sim.gridworld.done import ActiveDone


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

    def get_all_done(self, **kwargs):
        return False
        # TODO: Need 394, encoding target destruction
        
