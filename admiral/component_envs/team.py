
from admiral.envs import Agent

def TeamAgent(team=None, **kwargs):
    return {
        **Agent(**kwargs),
        'team': team,
    }
