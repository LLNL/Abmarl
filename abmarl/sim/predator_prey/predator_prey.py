from abc import ABC, abstractmethod
from enum import IntEnum

from gym.spaces import Box, Discrete, Dict
import numpy as np

from abmarl.sim import PrincipleAgent, AgentBasedSimulation


class PredatorPreyAgent(PrincipleAgent, ABC):
    """
    In addition to their own agent-type-specific parameters, every Agent in the
    Predator Prey simulation will have the following parameters:

    move: int
        The maximum movement range. 0 means the agent can only stay, 1 means the agent
        can move one square away (including diagonals), 2 means two, and so on.
        Default 1.

    view: int
        How much of the region the agent can observe.
        Default whole region.
    """
    @abstractmethod
    def __init__(self, move=None, view=None, **kwargs):
        super().__init__(**kwargs)
        self.move = move
        self.view = view

    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and self.move is not None and self.view is not None


class Prey(PredatorPreyAgent):
    """
    In addition to the shared parameters, Prey must have the following property:
    harvest_amount: float
        How much of the resource the prey will try to harvest if it chooses the
        harvest action. Default 0.4
    """
    def __init__(self, harvest_amount=None, **kwargs):
        super().__init__(**kwargs)
        self.harvest_amount = harvest_amount

    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and self.harvest_amount is not None

    @property
    def value(self):
        """
        The enumerated value of this agent is 1.
        """
        return 1


class Predator(PredatorPreyAgent):
    """
    In addition to the shared parameters, Predators must have the following property:

    attack: int
        The effective range of the attack. 0 means only effective on the same square, 1
        means effective at range 1, 2 at range 2, and so on.
        Default 0.
    """
    def __init__(self, attack=None, **kwargs):
        super().__init__(**kwargs)
        self.attack = attack

    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and self.attack is not None

    @property
    def value(self):
        """
        The enumerated value of this agent is 1.
        """
        return 2


class PredatorPreySimulation(AgentBasedSimulation):
    """
    Each agent observes other agents around its own location. Both predators and
    prey agents can move around. Predators' goal is to approach prey and attack
    it. Preys' goal is to stay alive for as long as possible.

    Note: we recommend that you use the class build function to instantiate the simulation because
    it has smart config checking for the simulation and will create agents that are configured to
    work with the simulation.
    """

    class ObservationMode(IntEnum):
        GRID = 0
        DISTANCE = 1

    class ActionStatus(IntEnum):
        BAD_MOVE = 0
        GOOD_MOVE = 1
        NO_MOVE = 2
        BAD_ATTACK = 3
        GOOD_ATTACK = 4
        EATEN = 5
        BAD_HARVEST = 6
        GOOD_HARVEST = 7

    def __init__(self, config):
        self.region = config['region']
        self.max_steps = config['max_steps']
        self.agents = config['agents']
        self.reward_map = config['rewards']

    def reset(self, **kwargs):
        """
        Randomly pick positions for each of the agents.
        """
        self.step_count = 0

        # Randomly assign agent positions, using row-column indexing
        for agent in self.agents.values():
            agent.position = np.random.randint(0, self.region, 2)

        # Holding for all agents that have died. Agents
        # in the cememtery are effectively removed from the simulation. They don't
        # appear in other agents' observations and they don't have observations
        # of their own, except for the one step in which they died.
        self.cemetery = set()

        # Track the agents' rewards over multiple steps.
        self.rewards = {agent_id: 0 for agent_id in self.agents}

    def step(self, joint_actions, **kwargs):
        """
        The simulation will update its state with the joint actions from all
        the agents. All agents can choose to move up to the amount of squares
        enabled in their move configuration. In addition, predators can choose
        to ATTACK.
        """
        self.step_count += 1

        # We want to make sure that only agents that are still alive have sent us actions
        for agent_id in joint_actions:
            assert agent_id not in self.cemetery

        # Initial setup
        for agent_id in joint_actions:
            self.rewards[agent_id] = 0 # Reset the reward of the acting agent(s).

        # Process the predators first
        for predator_id, action in joint_actions.items():
            predator = self.agents[predator_id]
            if type(predator) == Prey: continue # Process the predators first
            # Attack takes precedent over move
            if action['attack'] == 1:
                action_status = self._process_attack_action(predator)
            else:
                action_status = self._process_move_action(predator, action['move'])
            self.rewards[predator_id] = self.reward_map['predator'][action_status]

        # The prey are processed differently for Grid and Distance modes because
        # grid mode supports resources on the grid.

    def get_reward(self, agent_id, **kwargs):
        return self.rewards[agent_id]

    def get_done(self, agent_id, **kwargs):
        """
        Agent is done if it is not alive or in the morgue.
        """
        if agent_id in self.cemetery:
            return True
        else:
            return False

    def get_all_done(self, **kwargs):
        """
        If there are no prey left, the simulation is done.
        """
        if self.step_count >= self.max_steps:
            return True
        for agent in self.agents.values():
            if type(agent) == Prey and agent.id not in self.cemetery:
                return False
        return True

    def get_info(self, agent_id, **kwargs):
        """
        Just return an empty dictionary becuase this simulation does not track
        any info.
        """
        return {}

    def _process_move_action(self, agent, action):
        """
        The simulation will attempt to move the agent according to its action.
        If that move is successful, the agent will move and we will return GOOD_MOVE.
        If that move is unsuccessful, the agent will not move and we will return
        BAD_MOVE. Moves can be unsuccessful if the agent moves against a wall.
        If the agent chooses to stay, then we do not move the agent and return NO_MOVE.

        This should only be called if the agent has chosen to move or stay still.
        """
        action = np.rint(action)
        if all(action == [0, 0]):
            return self.ActionStatus.NO_MOVE
        elif 0 <= agent.position[0] + action[0] < self.region and \
                0 <= agent.position[1] + action[1] < self.region:
            # Still inside the boundary, good move
            agent.position[0] += action[0]
            agent.position[1] += action[1]
            return self.ActionStatus.GOOD_MOVE
        else:
            return self.ActionStatus.BAD_MOVE

    def _process_attack_action(self, predator):
        """
        The simulation will process the predator's attack action. If that attack
        is successful, the prey will be added to the morgue and we will return
        GOOD_ATTACK. If the attack is unsuccessful, then we will return BAD_ATTACK.

        This should only be called if the agent chooses to attack. Only predators
        can choose to attack.
        """
        for prey in self.agents.values():
            if type(prey) == Predator: continue # Not a prey
            if prey.id in self.cemetery: continue # Ignore already dead agents
            if abs(predator.position[0] - prey.position[0]) <= predator.attack \
                    and abs(predator.position[1] - prey.position[1]) <= predator.attack:
                # Good attack, prey is eaten:
                self.cemetery.add(prey.id)
                self.rewards[prey.id] += self.reward_map['prey'][self.ActionStatus.EATEN]
                return self.ActionStatus.GOOD_ATTACK
        return self.ActionStatus.BAD_ATTACK

    def _process_harvest_action(self, prey):
        """
        The simulation will process the prey's harvest action by calling the resources
        harvest api. If the amount harvested is the same as the amount attempted
        to harvest, then it was a good harvest. Otherwise, the agent over-harvested,
        or harvested a resource that wasn't ready yet, and so it was a bad harvest.

        This should only be called if the agent chooses to harvest. Only prey
        can choose to harvest.
        """
        harvested_amount = self.resources.harvest(tuple(prey.position), prey.harvest_amount)
        if harvested_amount == prey.harvest_amount:
            return self.ActionStatus.GOOD_HARVEST
        else:
            return self.ActionStatus.BAD_HARVEST

    @classmethod
    def build(cls, sim_config={}):
        """
        Args:
            region: int
                The size of the discrete space.
                Region must be >= 2.
                Default 10.
            max_steps: int
                The maximum number of steps per episode.
                Must be >= 1.
                Default 200.
            observation_mode: ObservationMode enum
                Either GRID or DISTANCE. In GRID, the agents see a grid of values around them as
                large as their view. In DISTANCE, the agents see the distance between themselves and
                other agents that they can see. Note: communication only works with
                DISTANCE observation mode.
                Default GRID.
            rewards: dict
                A dictionary that maps the various action status to a reward per each
                agent type. Any agent type that you create must have mappings for all
                possible action statuses for that agent type. The default is {
                    'predator': {
                        BAD_MOVE: -region,
                        GOOD_MOVE: -1,
                        NO_MOVE: 0,
                        BAD_ATTACK: -region,
                        GOOD_ATTACK: region**2
                    },
                    'prey': {
                        BAD_MOVE: -2,
                        GOOD_MOVE: region,
                        NO_MOVE: region,
                        EATEN: -region**2
                    },
                }
            resources: dictionary of resource-related parameters.
                See GridResources documentation for more information.
            agents: list of PredatorPreyAgent objects.
                You can set the parameters for each of the agent that will override
                the default parameters. For example,
                    agents = [
                        Prey(id='prey0', view=7, move=2),
                        Predator(id='predator1', view=3, attack=2),
                        Prey(id='prey2', view=5, move=3),
                        Predator(id='predator3', view=2, move=2, attack=1),
                        Predator(id='predator4', view=0, attack=3)
                    ]

        Returns:
            Configured instance of PredatorPreySimulation with configured PredatorPreyAgents.
        """
        config = {  # default config
            'region': 10,
            'max_steps': 200,
            'observation_mode': cls.ObservationMode.GRID,
            'resources': {} # Use the defaults in GridResources
            # 'rewards': # Determined based on the size of the region. See below.
            # 'agents': # Determine based on the size of the region. See below.
        }

        # --- region --- #
        if 'region' in sim_config:
            region = sim_config['region']
            if type(region) is not int or region < 2:
                raise TypeError("region must be an integer greater than 2.")
            else:
                config['region'] = region

        # Assign this here because we must use the right size of the region.
        config['agents'] = [
            Prey(id='prey0', view=config['region']-1, move=1, harvest_amount=0.1),
            Predator(id='predator0', view=config['region']-1, move=1, attack=0)
        ]
        # Assign this here so that we can coordinate rewards with region size.
        config['rewards'] = {
            'predator': {
                cls.ActionStatus.BAD_MOVE: -config['region'],
                cls.ActionStatus.GOOD_MOVE: -1,
                cls.ActionStatus.NO_MOVE: 0,
                cls.ActionStatus.BAD_ATTACK: -config['region'],
                cls.ActionStatus.GOOD_ATTACK: config['region']**2,
            },
            'prey': {
                cls.ActionStatus.BAD_MOVE: -config['region'],
                cls.ActionStatus.GOOD_MOVE: -1,
                cls.ActionStatus.NO_MOVE: 0,
                cls.ActionStatus.EATEN: -config['region']**2,
                cls.ActionStatus.BAD_HARVEST: -config['region'],
                cls.ActionStatus.GOOD_HARVEST: config['region'],
            },
        }

        # --- max_steps --- #
        if 'max_steps' in sim_config:
            max_steps = sim_config['max_steps']
            if type(max_steps) is not int or max_steps < 1:
                raise TypeError("max_steps must be an integer at least 1.")
            else:
                config['max_steps'] = max_steps

        # --- observation_mode --- #
        if 'observation_mode' in sim_config:
            observation_mode = sim_config['observation_mode']
            if observation_mode not in cls.ObservationMode:
                raise TypeError("observation_mode must be either GRID or DISTANCE.")
            else:
                config['observation_mode'] = observation_mode

        # --- rewards --- #
        if 'rewards' in sim_config:
            rewards = sim_config['rewards']
            if type(rewards) is not dict:
                raise TypeError("rewards must be a dict (see docstring).")
            else:
                config['rewards'] = rewards

        # --- resources --- #
        from abmarl.sim.modules import GridResources
        if 'resources' not in sim_config:
            sim_config['resources'] = {}
        sim_config['resources']['region'] = config['region']
        config['resources'] = GridResources.build(sim_config['resources'])

        # --- agents --- #
        if 'agents' in sim_config:
            agents = sim_config['agents']
            if type(agents) is not list:
                raise TypeError(
                    "agents must be a list of PredatorPreyAgent objects. "
                    "Each element in the list is an agent's configuration. See "
                    "PredatorPreyAgent docstring for more information."
                )
            else:
                for agent in agents:
                    if not isinstance(agent, PredatorPreyAgent):
                        raise TypeError("Every agent must be an instance of PredatorPreyAgent.")

                    if agent.view is None:
                        agent.view = config['region'] - 1
                    elif type(agent.view) is not int or agent.view < 0 or \
                            agent.view > config['region'] - 1:
                        raise TypeError(
                            f"{agent['id']} must have a view that is an integer "
                            f"between 0 and {config['region'] - 1}"
                        )

                    if agent.move is None:
                        agent.move = 1
                    elif type(agent.move) is not int or agent.move < 0 or \
                            agent.move > config['region'] - 1:
                        raise TypeError(
                            f"{agent['id']} must have a move that is an integer "
                            f"between 0 and {config['region'] - 1}"
                        )

                    if type(agent) is Predator:
                        if agent.attack is None:
                            agent.attack = 0
                        elif type(agent.attack) is not int or agent.attack < 0 or \
                                agent.attack > config['region']:
                            raise TypeError(
                                f"{agent['id']} must have an attack that is an integer "
                                f"between 0 and {config['region']}"
                            )

                    if type(agent) is Prey:
                        if agent.harvest_amount is None:
                            agent.harvest_amount = 0.4
                        elif type(agent.harvest_amount) is not float or agent.harvest_amount < 0:
                            raise TypeError(
                                f"{agent['id']} must have a harvest amount that is a float "
                                "greater than 0."
                            )

                config['agents'] = agents

        if config['observation_mode'] == cls.ObservationMode.GRID:
            obs_space_builder = lambda agent: Dict({
                'agents': Box(-1, 2, (2*agent.view+1, 2*agent.view+1), np.int),
                'resources': Box(
                    -1., config['resources'].max_value, (2*agent.view+1, 2*agent.view+1), np.float
                )
            })
            prey_action_space_builder = lambda agent: Dict({
                'harvest': Discrete(2),
                'move': Box(-agent.move-0.5, agent.move+0.5, (2,))
            })
        else:
            obs_space_builder = lambda agent: Dict({
                other_agent.id: Box(-config['region']+1, config['region']-1, (3,), np.int)
                for other_agent in config['agents'] if other_agent.id != agent.id
            })
            prey_action_space_builder = lambda agent: Box(-agent.move-0.5, agent.move+0.5, (2,))

        for agent in config['agents']:
            if type(agent) is Prey:
                agent.observation_space = obs_space_builder(agent)
                agent.action_space = prey_action_space_builder(agent)
            else:
                agent.observation_space = obs_space_builder(agent)
                agent.action_space = Dict({
                    'attack': Discrete(2),
                    'move': Box(-agent.move-0.5, agent.move+0.5, (2,)),
                })
        config['agents'] = {agent.id: agent for agent in config['agents']}

        if config['observation_mode'] == cls.ObservationMode.GRID:
            return PredatorPreySimGridObs(config)
        else:
            return PredatorPreySimDistanceObs(config)


class PredatorPreySimGridObs(PredatorPreySimulation):
    """
    PredatorPreySimulation where observations are of the grid and the items/agents on
    that grid up to the view.
    """
    def __init__(self, config):
        super().__init__(config)
        self.resources = config['resources']

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.resources.reset(**kwargs)

    def step(self, joint_actions, **kwargs):
        super().step(joint_actions, **kwargs)

        for prey_id, action in joint_actions.items():
            prey = self.agents[prey_id]
            if type(prey) == Predator: continue # Process the prey now
            if prey_id in self.cemetery: # This prey was eaten by a predator in this time step.
                continue
            if action['harvest'] == 1:
                action_status = self._process_harvest_action(prey)
            else:
                action_status = self._process_move_action(prey, action['move'])
            self.rewards[prey_id] = self.reward_map['prey'][action_status]

        # Now process the other pieces of the simulation
        self.resources.regrow()

    def render(self, *args, fig=None, **kwargs):
        """
        Visualize the state of the simulation. If a figure is received, then we
        will draw but not actually plot because we assume the caller will do the
        work (e.g. with an Animation object). If there is no figure received, then
        we will draw and plot the simulation. Call the resources render function
        too to plot the resources heatmap.
        """
        draw_now = fig is None
        if draw_now:
            from matplotlib import pyplot as plt
            fig = plt.gcf()

        fig.clear()
        ax = self.resources.render(fig=fig)

        prey_x = [
            agent.position[1] + 0.5 for agent in self.agents.values()
            if type(agent) == Prey and agent.id not in self.cemetery
        ]
        prey_y = [
            self.region - 0.5 - agent.position[0] for agent in self.agents.values()
            if type(agent) == Prey and agent.id not in self.cemetery
        ]
        ax.scatter(prey_x, prey_y, marker='s', s=200, edgecolor='black', facecolor='gray')

        predator_x = [
            agent.position[1] + 0.5 for agent in self.agents.values()
            if type(agent) == Predator and agent.id not in self.cemetery
        ]
        predator_y = [
            self.region - 0.5 - agent.position[0] for agent in self.agents.values()
            if type(agent) == Predator and agent.id not in self.cemetery
        ]
        ax.scatter(predator_x, predator_y, s=200, marker='o', edgecolor='black', facecolor='gray')

        if draw_now:
            plt.plot()
            plt.pause(1e-17)

        return ax

    def get_obs(self, my_id, **kwargs):
        """
        Each agent observes a grid of values surrounding its location, whose size
        is determiend by the agent's view. There are two channels in this grid:
        an agents channel and a resources channel.
        """
        return {
            'agents': self._observe_other_agents(my_id, **kwargs),
            'resources': self._observe_resources(my_id, **kwargs),
        }

    def _observe_other_agents(self, my_id, **kwargs):
        """
        These cells are filled with the value of the agent's type, including -1
        for out of bounds and 0 for empty square. If there are multiple agents
        on the same cell, then we prioritize the agent that is of a different
        type. For example, a prey will only see a predator on a cell that a predator
        and another prey both occupy.
        """
        my_agent = self.agents[my_id]
        signal = np.zeros((my_agent.view*2+1, my_agent.view*2+1))

        # --- Determine the boundaries of the agents' grids --- #
        # For left and top, we just do: view - x,y >= 0
        # For the right and bottom, we just do region - x,y - 1 - view > 0
        if my_agent.view - my_agent.position[0] >= 0: # Top end
            signal[0:my_agent.view - my_agent.position[0], :] = -1
        if my_agent.view - my_agent.position[1] >= 0: # Left end
            signal[:, 0:my_agent.view - my_agent.position[1]] = -1
        if self.region - my_agent.position[0] - my_agent.view - 1 < 0: # Bottom end
            signal[self.region - my_agent.position[0] - my_agent.view - 1:, :] = -1
        if self.region - my_agent.position[1] - my_agent.view - 1 < 0: # Right end
            signal[:, self.region - my_agent.position[1] - my_agent.view - 1:] = -1

        # --- Determine the positions of all the other alive agents --- #
        for other_id, other_agent in self.agents.items():
            if other_id == my_id or other_id in self.cemetery: continue
            r_diff = other_agent.position[0] - my_agent.position[0]
            c_diff = other_agent.position[1] - my_agent.position[1]
            if -my_agent.view <= r_diff <= my_agent.view and \
                    -my_agent.view <= c_diff <= my_agent.view:
                r_diff += my_agent.view
                c_diff += my_agent.view
                if signal[r_diff, c_diff] != 0: # Already another agent here
                    if type(my_agent) != type(other_agent):
                        signal[r_diff, c_diff] = other_agent.value
                else:
                    signal[r_diff, c_diff] = other_agent.value

        return signal

    def _observe_resources(self, agent_id, **kwargs):
        """
        These cells are filled with the values of the resources surrounding the
        agent.
        """
        agent = self.agents[agent_id]
        signal = -np.ones((agent.view*2+1, agent.view*2+1))

        # Derived by considering each square in the resources as an "agent" and
        # then applied the agent diff logic from above. The resulting for-loop
        # can be written in the below vectorized form.
        (r, c) = agent.position
        r_lower = max([0, r-agent.view])
        r_upper = min([self.region-1, r+agent.view])+1
        c_lower = max([0, c-agent.view])
        c_upper = min([self.region-1, c+agent.view])+1
        signal[
            (r_lower+agent.view-r):(r_upper+agent.view-r),
            (c_lower+agent.view-c):(c_upper+agent.view-c)
        ] = self.resources.resources[r_lower:r_upper, c_lower:c_upper]
        return signal


class PredatorPreySimDistanceObs(PredatorPreySimulation):
    """
    PredatorPrey simulation where observations are of the distance from each
    other agent within the view.
    """
    def step(self, joint_actions, **kwargs):
        super().step(joint_actions, **kwargs)
        for prey_id, action in joint_actions.items():
            prey = self.agents[prey_id]
            if type(prey) == Predator: continue # Process the prey now
            if prey_id in self.cemetery: # This prey was eaten by a predator in this time step.
                continue
            action_status = self._process_move_action(prey, action)
            self.rewards[prey_id] = self.reward_map['prey'][action_status]

    def render(self, *args, fig=None, **kwargs):
        """
        Visualize the state of the simulation. If a figure is received, then we
        will draw but not actually plot because we assume the caller will do the
        work (e.g. with an Animation object). If there is no figure received, then
        we will draw and plot the simulation.
        """
        draw_now = fig is None
        if draw_now:
            from matplotlib import pyplot as plt
            fig = plt.gcf()

        fig.clear()
        ax = fig.gca()
        ax.set(xlim=(-0.5, self.region - 0.5), ylim=(-0.5, self.region - 0.5))
        ax.set_xticks(np.arange(-0.5, self.region - 0.5, 1.))
        ax.set_yticks(np.arange(-0.5, self.region - 0.5, 1.))
        ax.grid(linewidth=5)

        prey_x = [
            agent.position[1] for agent in self.agents.values()
            if type(agent) == Prey and agent.id not in self.cemetery
        ]
        prey_y = [
            self.region - 1 - agent.position[0] for agent in self.agents.values()
            if type(agent) == Prey and agent.id not in self.cemetery
        ]
        ax.scatter(prey_x, prey_y, marker='s', s=200, edgecolor='black', facecolor='gray')

        predator_x = [
            agent.position[1] for agent in self.agents.values()
            if type(agent) == Predator and agent.id not in self.cemetery
        ]
        predator_y = [
            self.region - 1 - agent.position[0] for agent in self.agents.values()
            if type(agent) == Predator and agent.id not in self.cemetery
        ]
        ax.scatter(predator_x, predator_y, s=200, marker='o', edgecolor='black', facecolor='gray')

        if draw_now:
            plt.plot()
            plt.pause(1e-17)

        return ax

    def get_obs(self, my_id, fusion_matrix={}, **kwargs):
        """
        Agents observe a distance from itself to other agents only if the other
        agents are visible (i.e. within the agent's view). If agents are not visible,
        then the observation "slot" is empty.

        Via communication an agent's observations can be combined with other agents.
        The fusion_matrix dictates which observations to share.
        """
        my_agent = self.agents[my_id]
        my_obs = {}

        # --- Determine the positions of all the other alive agents --- #
        for other_id in self.agents:
            if my_id == other_id: continue
            my_obs[other_id] = np.zeros(3, dtype=np.int)
        # Fill values for agents that are still alive
        for other_id, other_agent in self.agents.items():
            if other_id == my_id or other_id in self.cemetery: continue
            r_diff = other_agent.position[0] - my_agent.position[0]
            c_diff = other_agent.position[1] - my_agent.position[1]
            if -my_agent.view <= c_diff <= my_agent.view and \
                    -my_agent.view <= r_diff <= my_agent.view:
                my_obs[other_id] = np.array((r_diff, c_diff, other_agent.value))

        # --- Get the observations from other agents --- #
        for sending_agent_id, message in fusion_matrix.items():
            # Only receive messages from alive agents
            if sending_agent_id not in self.cemetery and message:
                for spied_agent_id, distance_type in self.get_obs(sending_agent_id).items():
                    # Don't receive a message about yourself or other agents
                    # that you already see
                    if spied_agent_id != my_id and \
                            my_obs[spied_agent_id][2] == 0 and \
                            distance_type[2] != 0: # We actually see an agent here
                        spied_agent = self.agents[spied_agent_id]
                        r_diff = spied_agent.position[0] - my_agent.position[0]
                        c_diff = spied_agent.position[1] - my_agent.position[1]
                        my_obs[spied_agent_id] = np.array([r_diff, c_diff, spied_agent.value])
                # Also see the relative location of the sending agent itself
                sending_agent = self.agents[sending_agent_id]
                c_diff = sending_agent.position[1] - my_agent.position[1]
                r_diff = sending_agent.position[0] - my_agent.position[0]
                my_obs[sending_agent_id] = np.array([r_diff, c_diff, sending_agent.value])

        return my_obs
