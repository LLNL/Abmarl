import numpy as np

from admiral.envs.modules import GridResources


def test_builder():
    env = GridResources.build()
    assert env.region == 10
    assert env.max_value == 1.
    assert env.min_value == 0.1
    assert env.revive_rate == 0.04
    assert env.coverage == 0.75


def test_builder_custom():
    env = GridResources.build({
        'region': 5,
        'max_value': 2.,
        'min_value': 0.01,
        'revive_rate': 0.5,
        'coverage': 0.4
    })
    assert env.region == 5
    assert env.max_value == 2.
    assert env.min_value == 0.01
    assert env.revive_rate == 0.5
    assert env.coverage == 0.4


def test_reset():
    np.random.seed(24)
    env = GridResources.build({'region': 5})
    env.reset()
    assert ((env.resources <= env.max_value) & (env.resources >= 0.)).all()


def test_harvest_and_regrow():
    np.random.seed(24)
    env = GridResources.build()
    env.reset()

    # Normal action with harvest and replenish
    value_before = {
        (4,5) : env.resources[(4,5)],
        (3,3) : env.resources[(3,3)]
    }
    assert env.harvest((4,5), 0.7) == 0.7
    assert env.harvest((3,3), 0.1) == 0.1
    env.regrow()
    assert env.resources[(4,5)] == value_before[(4,5)] - 0.7 + 0.04
    assert env.resources[(3,3)] == value_before[(3,3)] - 0.1 + 0.04

    # action that has depleted one of the resources
    value_before = {
        (4,5) : env.resources[(4,5)],
        (2,1) : env.resources[(2,1)]
    }
    assert env.harvest((4,5), 0.7) == value_before[(4,5)]
    assert env.harvest((2,1), 0.15) == 0.15
    env.regrow()
    assert env.resources[(4,5)] == 0.
    assert env.resources[(2,1)] == value_before[(2,1)] - 0.15 + 0.04

    # Check that the depleted resources do not restore
    value_before = {
        (2,1) : env.resources[(2,1)]
    }
    env.regrow()
    assert env.resources[(4,5)] == 0.
    assert env.resources[(2,1)] == value_before[(2,1)] + 0.04

    # Check that nothing is above maximum value
    for _ in range(25):
        env.regrow()
    assert (env.resources <= env.max_value).all()
