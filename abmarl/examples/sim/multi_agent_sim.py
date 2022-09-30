
from gym.spaces import Discrete, MultiBinary, MultiDiscrete, Box, Tuple, Dict

from abmarl.sim import PrincipleAgent, Agent, AgentBasedSimulation


class EmptyABS(AgentBasedSimulation):
    def __init__(self, agents=None):
        self.agents = agents

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass

    def get_obs(self):
        pass

    def get_reward(self):
        pass

    def get_done(self):
        pass

    def get_all_done(self):
        pass

    def get_info(self):
        pass


class MultiAgentSim(EmptyABS):
    def __init__(self, num_agents=3, num_principle_agent=2):
        self.rewards = [0, 1, 2, 3, 4, 5, 6]
        self.dones = [3, 12, 5, 34]
        self.step_count = 0
        self.agents = {
            **{
                'agent' + str(i): Agent(
                    id='agent'+str(i), observation_space=Discrete(2), action_space=Discrete(2)
                ) for i in range(num_agents)
            },
            **{
                'principle_agent' + str(i): PrincipleAgent(
                    id='principle_agent'+str(i)
                ) for i in range(num_principle_agent)
            }
        }

    def reset(self):
        self.step_count = 0
        self.action = {
            agent.id: None for agent in self.agents.values() if isinstance(agent, Agent)
        }

    def step(self, action_dict):
        self.step_count += 1
        for agent_id, action in action_dict.items():
            self.action[agent_id] = action

    def get_obs(self, agent_id, **kwargs):
        return "Obs from " + agent_id

    def get_reward(self, agent_id, **kwargs):
        return "Reward from " + agent_id

    def get_done(self, agent_id, **kwargs):
        return "Done from " + agent_id

    def get_all_done(self, **kwargs):
        return "Done from all agents and/or simulation."

    def get_info(self, agent_id, **kwargs):
        return {'Action from ' + agent_id: self.action[agent_id]}


class MultiAgentGymSpacesSim(MultiAgentSim):
    def __init__(self):
        super().__init__()
        self.agents = {
            'agent0': Agent(
                id='agent0',
                observation_space=MultiBinary(4),
                action_space=Tuple((
                    Dict({
                        'first': Discrete(4),
                        'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                    }),
                    MultiBinary(3)
                )),
                null_observation=[0, 0, 0, 0]
            ),
            'agent1': Agent(
                id='agent1',
                observation_space=Box(low=0, high=1, shape=(1,), dtype=int),
                action_space=MultiDiscrete([4, 6, 2])
            ),
            'agent2': Agent(
                id='agent2',
                observation_space=MultiDiscrete([2, 2]),
                action_space=Dict({'alpha': MultiBinary(3)})
            ),
            'agent3': Agent(
                id='agent3',
                observation_space=Dict({
                    'first': Discrete(4),
                    'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                }),
                action_space=Tuple((Discrete(3), MultiDiscrete([10, 10]), Discrete(2)))
            ),
            'agent4': PrincipleAgent(
                id='agent4'
            )
        }

        self.finalize()

    def get_obs(self, agent_id, **kwargs):
        if agent_id == 'agent0':
            return [0, 0, 0, 1]
        elif agent_id == 'agent1':
            return [0]
        elif agent_id == 'agent2':
            return [1, 0]
        elif agent_id == 'agent3':
            return {'first': 1, 'second': [3, 1]}

    def get_reward(self, agent_id, **kwargs):
        return {
            'agent0': 2,
            'agent1': 3,
            'agent2': 5,
            'agent3': 7,
        }[agent_id]

    def get_done(self, agent_id, **kwargs):
        return self.step_count >= self.dones[int(agent_id[-1])]

    def get_all_done(self, **kwargs):
        for agent in self.agents.values():
            if not isinstance(agent, Agent): continue
            if not self.get_done(agent.id):
                return False
        return True

    def get_info(self, agent_id, **kwargs):
        return self.action[agent_id]


class MultiAgentContinuousGymSpaceSim(MultiAgentSim):
    def __init__(self):
        self.params = {'params': "there are none"}
        self.agents = {
            'agent0': Agent(
                id='agent0',
                observation_space=MultiBinary(4),
                action_space=Tuple((
                    Dict({'first': Discrete(4), 'second': Box(low=-1, high=3, shape=(2,))}),
                    MultiBinary(3)
                ))
            ),
            'agent1': Agent(
                id='agent1',
                observation_space=Box(low=0, high=1, shape=(1,)),
                action_space=MultiDiscrete([4, 6, 2])
            ),
            'agent2': Agent(
                id='agent2',
                observation_space=MultiDiscrete([2, 2]),
                action_space=Dict({'alpha': MultiBinary(3)})
            ),
            'agent3': Agent(
                id='agent3',
                observation_space=Dict({
                    'first': Discrete(4),
                    'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                }),
                action_space=Tuple((Discrete(3), MultiDiscrete([10, 10]), Discrete(2)))
            )
        }

        self.finalize()

    def get_obs(self, agent_id, **kwargs):
        if agent_id == 'agent0':
            return [0, 1, 1, 0]
        elif agent_id == 'agent1':
            return [0.98]
        elif agent_id == 'agent2':
            return [1, 0]
        elif agent_id == 'agent3':
            return {'first': 1, 'second': [-1, 1]}

    def get_info(self, agent_id, **kwargs):
        return self.action[agent_id]


class MultiAgentSameSpacesSim(AgentBasedSimulation):
    def __init__(self):
        self.rewards = [0, 1, 2, 3, 4, 5, 6]
        self.ith_call = -1
        self.dones = [12, 25]
        self.step_count = 0
        self.agents = {
            'agent0': Agent(
                id='agent0',
                observation_space=MultiBinary(4),
                action_space=Tuple((
                    Dict({
                        'first': Discrete(4),
                        'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                    }),
                    MultiBinary(3)
                ))
            ),
            'agent1': PrincipleAgent(
                id='agent1',
            ),
            'agent2': Agent(
                id='agent2',
                observation_space=MultiBinary(4),
                action_space=Tuple((
                    Dict({
                        'first': Discrete(4),
                        'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                    }),
                    MultiBinary(3)
                ))
            ),
            'agent3': PrincipleAgent(
                id='agent3',
            ),
            'agent4': PrincipleAgent(
                id='agent4'
            )
        }

    def render(self):
        pass

    def reset(self):
        self.action = {
            'agent0': None,
            'agent2': None,
        }

    def step(self, action_dict):
        self.step_count += 1
        for agent_id, action in action_dict.items():
            self.action[agent_id] = action

    def get_obs(self, agent_id, **kwargs):
        if agent_id == 'agent0':
            return [0, 0, 0, 1]
        elif agent_id == 'agent2':
            return [1, 0, 1, 0]

    def get_reward(self, agent_id, **kwargs):
        self.ith_call = (self.ith_call + 1) % 7
        return self.rewards[self.ith_call]

    def get_done(self, agent_id, **kwargs):
        if agent_id == 'agent0':
            return self.step_count >= self.dones[0]
        elif agent_id == "agent2":
            return self.step_count >= self.dones[1]

    def get_all_done(self, **kwargs):
        for agent in self.agents.values():
            if not isinstance(agent, Agent): continue
            if not self.get_done(agent.id):
                return False
        return True

    def get_info(self, agent_id, **kwargs):
        return self.action[agent_id]
