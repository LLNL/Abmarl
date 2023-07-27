from gym.spaces import Discrete, Dict, Box as GymBox
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


def test_box_space():
    gb = GymBox(0, 4, (1,), int)
    ab = gu.Box(0, 4, (1,), int)
    assert isinstance(ab, GymBox)
    assert [3] in gb
    assert 3 not in gb
    assert [3] in ab
    assert 3 in ab
    assert 3.3 not in ab
    assert 5 not in ab
    assert -1 not in ab
    assert [3, 4] not in ab
