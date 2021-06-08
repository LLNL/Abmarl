import numpy as np

from abmarl.sim.modules import GridResources


def test_builder():
    sim = GridResources.build()
    assert sim.region == 10
    assert sim.max_value == 1.
    assert sim.min_value == 0.1
    assert sim.revive_rate == 0.04
    assert sim.coverage == 0.75


def test_builder_custom():
    sim = GridResources.build({
        'region': 5,
        'max_value': 2.,
        'min_value': 0.01,
        'revive_rate': 0.5,
        'coverage': 0.4
    })
    assert sim.region == 5
    assert sim.max_value == 2.
    assert sim.min_value == 0.01
    assert sim.revive_rate == 0.5
    assert sim.coverage == 0.4


def test_reset():
    np.random.seed(24)
    sim = GridResources.build({'region': 5})
    sim.reset()
    assert ((sim.resources <= sim.max_value) & (sim.resources >= 0.)).all()


def test_harvest_and_regrow():
    np.random.seed(24)
    sim = GridResources.build()
    sim.reset()

    # Normal action with harvest and replenish
    value_before = {
        (4,5) : sim.resources[(4,5)],
        (3,3) : sim.resources[(3,3)]
    }
    assert sim.harvest((4,5), 0.7) == 0.7
    assert sim.harvest((3,3), 0.1) == 0.1
    sim.regrow()
    assert sim.resources[(4,5)] == value_before[(4,5)] - 0.7 + 0.04
    assert sim.resources[(3,3)] == value_before[(3,3)] - 0.1 + 0.04

    # action that has depleted one of the resources
    value_before = {
        (4,5) : sim.resources[(4,5)],
        (2,1) : sim.resources[(2,1)]
    }
    assert sim.harvest((4,5), 0.7) == value_before[(4,5)]
    assert sim.harvest((2,1), 0.15) == 0.15
    sim.regrow()
    assert sim.resources[(4,5)] == 0.
    assert sim.resources[(2,1)] == value_before[(2,1)] - 0.15 + 0.04

    # Check that the depleted resources do not restore
    value_before = {
        (2,1) : sim.resources[(2,1)]
    }
    sim.regrow()
    assert sim.resources[(4,5)] == 0.
    assert sim.resources[(2,1)] == value_before[(2,1)] + 0.04

    # Check that nothing is above maximum value
    for _ in range(25):
        sim.regrow()
    assert (sim.resources <= sim.max_value).all()
