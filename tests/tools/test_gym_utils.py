from gym.spaces import Discrete, Dict
import pytest

from abmarl.tools import gym_utils as gu


def test_check_space():
    space = {
        1: Discrete(1),
        2: {
            1: {},
            2: Discrete(2),
        },
        3: {
            1: Dict({
                1: Discrete(1),
            }),
            2: {
                1: Discrete(2),
                2: Discrete(2),
            },
            3: Dict({
                1: Discrete(3),
                2: Discrete(3),
                3: Discrete(3),
            })
        }
    }
    assert gu.check_space(space)
    assert not gu.check_space(space, True)

    space = {
        1: Discrete(1),
        2: {
            1: {},
            2: Discrete(2),
        },
        3: {
            1: Dict({
                1: Discrete(1),
            }),
            2: {
                1: 2, # This is not  gym space or dict
                2: Discrete(2),
            },
            3: Dict({
                1: Discrete(3),
                2: Discrete(3),
                3: Discrete(3),
            })
        }
    }
    assert not gu.check_space(space)


def test_make_dict():
    space = {
        1: Discrete(1),
        2: {
            1: {},
            2: Discrete(2),
        },
        3: {
            1: Dict({
                1: Discrete(1),
            }),
            2: {
                1: Discrete(2),
                2: Discrete(2),
            },
            3: Dict({
                1: Discrete(3),
                2: Discrete(3),
                3: Discrete(3),
            })
        }
    }

    space = gu.make_dict(space)

    assert space == Dict({
        1: Discrete(1),
        2: Dict({
            1: Dict(),
            2: Discrete(2),
        }),
        3: Dict({
            1: Dict({
                1: Discrete(1),
            }),
            2: Dict({
                1: Discrete(2),
                2: Discrete(2),
            }),
            3: Dict({
                1: Discrete(3),
                2: Discrete(3),
                3: Discrete(3),
            })
        })
    })


def test_make_dict_fail():
    space = {
        1: Discrete(1),
        2: {
            1: Discrete(2),
            2: {},
        },
        3: {
            1: Dict({
                1: Discrete(1),
            }),
            2: {
                1: 2, # This is not a gym space or dict
                2: Discrete(2),
            },
            3: Dict({
                1: Discrete(3),
                2: Discrete(3),
                3: Discrete(3),
            })
        }
    }
    with pytest.raises(AssertionError):
        gu.make_dict(space)
