
from abmarl.examples import MultiCorridor
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.managers import TurnBasedManager
from abmarl.policies.q_table_policy import EpsilonSoftPolicy
from abmarl.trainers import DebugTrainer

# Build the simulation.
sim = TurnBasedManager(
    RavelDiscreteWrapper(
        MultiCorridor(end=5, num_agents=2)
    )
)
agents = sim.sim.agents

# Create the policies
policies = {
    agent.id: EpsilonSoftPolicy(
        observation_space=agent.observation_space,
        action_space=agent.action_space
    ) for agent in agents.values()
}

policy_mapping_fn = lambda agent_id: agent_id

# Setup the Debugger
debugger = DebugTrainer(
    sim=sim,
    policies=policies,
    policy_mapping_fn=policy_mapping_fn,
    name="Epsilon_Soft_Multi_Corridor"
)
debugger.train(iterations=4, render=True, horizon=20)
