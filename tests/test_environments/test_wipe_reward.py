"""Regression tests for Wipe environment reward accounting."""

from types import SimpleNamespace

import numpy as np

from robosuite.environments.manipulation.wipe import Wipe


def test_sparse_reward_includes_each_new_marker():
    """Sparse rewards should include the documented per-marker reward."""
    marker = SimpleNamespace(visual_geoms=["marker"])
    robot = SimpleNamespace(
        arms=["right"],
        recent_ee_forcetorques={"right": SimpleNamespace(current=np.zeros(6))},
        robot_model=object(),
        gripper={
            "right": SimpleNamespace(important_geoms={"corners": []}),
        },
        check_q_limits=lambda: False,
    )
    sim = SimpleNamespace(
        model=SimpleNamespace(
            geom_name2id=lambda _: 0,
            geom_rgba=np.ones((1, 4)),
        ),
    )
    wipe = SimpleNamespace(
        robots=[robot],
        check_contact=lambda _: False,
        reward_shaping=False,
        collisions=0,
        wiped_markers=[],
        _get_active_markers=lambda _: [marker],
        sim=sim,
        unit_wiped_reward=5.0,
        num_markers=2,
        print_results=False,
        reward_scale=None,
    )

    reward = Wipe.reward(wipe)

    assert reward == wipe.unit_wiped_reward
    assert wipe.wiped_markers == [marker]
    assert sim.model.geom_rgba[0, 3] == 0
