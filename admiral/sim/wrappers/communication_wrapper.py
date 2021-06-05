from .wrapper import Wrapper

from gym.spaces import Discrete, Dict


class CommunicationHandshakeWrapper(Wrapper):
    """
    Agents can share their observations with one another according to this communication
    protocol. We allow agents to send and receive messages from any other agent
    in the simulation. When an agent chooses to send, recipient agents will see in their next
    observation that there are incoming messages. Agents can choose to receive
    or ignore the message. If an agent chooses to receive, its observation space
    is fused with the sending agent's observation. That fusion happens according
    to the fuse_observation function, which must be implemented by the wrapped simulation.

    Agents' observation and action spaces are converted into dictionaries. The
    action and observation from the simulation are keyed on 'action' and 'obs',
    respectively. We add 'message_buffer' to the observation, which shows incoming
    messages; and 'receive' and 'send' to the action, which are the two communication
    actions the agents can take.
    """
    def __init__(self, sim):
        super().__init__(sim)

        # Augment the agents' action and observation spaces.
        # We use a dict keyed off the agents' id to Discrete(2) instead of just
        # MultiBinary(num_agents-1) because MultiBinary only gives us
        # T/F at some indicies. We would need additional mapping information to
        # map from the index of the MultiBinary observation/action to the respective
        # agent. Using a dict gives us that information automatically.
        for agent in self.agents.values():
            action_space_helper = {'action': agent.action_space}
            action_space_helper['send'] = Dict({
                other_id: Discrete(2) for other_id in self.agents if other_id != agent.id
            })
            action_space_helper['receive'] = Dict({
                other_id: Discrete(2) for other_id in self.agents if other_id != agent.id
            })
            agent.action_space = Dict(action_space_helper)

            obs_space_helper = {'obs': agent.observation_space}
            obs_space_helper['message_buffer'] = Dict({
                other_id: Discrete(2) for other_id in self.agents if other_id != agent.id
            })
            agent.observation_space = Dict(obs_space_helper)

    def reset(self, **kwargs):
        """
        Set the internal communication state to the null state and reset the wrapped
        simulation.
        """
        self.message_buffer = {}
        self.received_message = {}
        for my_id in self.agents:
            self.message_buffer[my_id] = {
                other_id: False for other_id in self.agents if other_id != my_id
            }
            self.received_message[my_id] = {
                other_id: False for other_id in self.agents if other_id != my_id
            }
        self.sim.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        """
        First, we process the receive actions, which will be used to determine
        any message that are successfully communicated among agents. Then, we call
        the step function from the wrapped simulation. Finally, we process the
        send actions to update the message buffer observations.
        """
        # Process receive actions
        for receiving_agent, action in action_dict.items():
            self.received_message[receiving_agent] = {
                sending_agent: True if self.message_buffer[receiving_agent][sending_agent] and
                action['receive'][sending_agent] else False
                for sending_agent in self.received_message[receiving_agent]
            }
        # Reset the message buffer
        for my_id in self.agents:
            self.message_buffer[my_id] = {
                other_id: False for other_id in self.agents if other_id != my_id
            }

        # Wrapped simulation takes a step
        sim_only_action = {
            agent_id: action_dict[agent_id]['action'] for agent_id in action_dict
        }
        self.sim.step(sim_only_action, **kwargs)

        # Process send actions
        for sending_agent, action in action_dict.items():
            for receiving_agent, message in action['send'].items():
                self.message_buffer[receiving_agent][sending_agent] = message

    def get_obs(self, agent_id, **kwargs):
        """
        The (fused) observation from the wrapped simulation is keyed on 'obs'
        and we add 'message_buffer' for incoming messages.
        """
        obs_from_sim = self.sim.get_obs(agent_id, fusion_matrix=self.received_message[agent_id])
        return {'obs': obs_from_sim, 'message_buffer': self.message_buffer[agent_id]}
