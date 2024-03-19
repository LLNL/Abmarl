
import random

import numpy as np

from abmarl.sim import Agent
from abmarl.sim.gridworld.agent import MovingAgent, GridObservingAgent, MoneyAgent, PricingAgent
from abmarl.sim.gridworld.smart import SmartGridWorldSimulation
from abmarl.sim.gridworld.actor import CrossMoveActor, PriceSettingActor


class BuyerAgent(MoneyAgent):
    def __init__(
        self,
        encoding=1,
        render_shape='s',
        **kwargs
    ):
        super().__init__(
            encoding=encoding,
            render_shape=render_shape,
            **kwargs
        )


class SellerAgent(MoneyAgent, MovingAgent, GridObservingAgent, PricingAgent):
    def __init__(
        self,
        encoding=2,
        move_range=1,
        view_range="FULL",
        **kwargs
    ):
        super().__init__(
            encoding=encoding,
            move_range=move_range,
            view_range=view_range,
            **kwargs
        )


class HotelingsLawSim(SmartGridWorldSimulation):
    def __init__(self, costs=None, **kwargs):
        super().__init__(**kwargs)
        self.costs = costs

        self.move_actor = CrossMoveActor(**kwargs)
        self.price_actor = PriceSettingActor(**kwargs)

        self.finalize()

    @property
    def costs(self):
        """
        Specify the costs of moving and the daily operating cost.
        """
        return self._costs
    
    @costs.setter
    def costs(self, value):
        if value is not None:
            assert type(value) is dict, "Costs must be a dictionary."
            for event, reward in value.items():
                assert event in ['moving', 'entropy'], \
                    "Supported costs: 'moving', 'entropy'."
                assert type(reward) in [int, float], f"Cost for {event} must be numerical."
            self._reward_scheme = value
        else:
            self._costs = {
                'moving': -25,
                'entropy': -10,
            }

    def score_seller(self, buyer_position, seller_position, seller_price):
        return np.linalg.norm(buyer_position - seller_position) + seller_price

    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]

            # Process move action
            if action != 0: # If the agent chooses to move
                agent.income += self.costs['moving']
            self.move_actor.process_action(agent, action, **kwargs)

            # Process price setting action
            self.price_actor.process_action(agent, action, **kwargs)

        # Update the buyers
        for buyer in self.agents.values():
            if isinstance(buyer, BuyerAgent):
                preference = {
                    seller.id: self.score_seller(buyer.position, seller.position, seller.price)
                    for seller in self.agents.values() if isinstance(seller, SellerAgent)
                }
                min_value = min(preference.values())
                if min_value > buyer.money:
                    buyer.preference = None
                    buyer.render_color = 'gray'
                else:
                    buyer.preference = random.choice([seller for seller, pref in preference.items() if pref == min_value])
                    buyer.render_color = buyer.preference.render_color
                    buyer.preference.income += buyer.preference.price

        # Update money, active status, and calculate rewards
        for seller in self.agents.values():
            if isinstance(seller, SellerAgent):
                seller.income += self.costs['entropy'] # Daily operating cost
                self.rewards[seller.id] += seller.income
                seller.money += seller.income
                seller.income = 0
                if seller.money <= 0:
                    seller.active = False
                    self.grid.remove(seller, seller.position)
        
