"""
Tests the joint-space mass matrix that Controller.update() computes for the arm.

MuJoCo 3.10 changed mj_fullM(m, dst, M) to mj_fullM(m, d, dst), and 3.11 removed MjData.qM. This checks, on
whichever MuJoCo version is installed, that a controller step runs and that the controller's mass matrix matches
one built column by column with mj_mulM, which has the same signature on every supported MuJoCo version.
"""
import mujoco
import numpy as np

import robosuite as suite


def test_controller_mass_matrix_matches_mj_mulM():
    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    env.reset()
    env.step(np.zeros(env.action_dim))

    model, data = env.sim.model._model, env.sim.data._data
    controller = env.robots[0].part_controllers["right"]
    controller.update(force=True)

    full = np.zeros((model.nv, model.nv))
    column = np.zeros(model.nv)
    for i in range(model.nv):
        unit = np.zeros(model.nv)
        unit[i] = 1.0
        mujoco.mj_mulM(model, data, column, unit)
        full[:, i] = column
    expected = full[controller.qvel_index, :][:, controller.qvel_index]

    np.testing.assert_allclose(controller.mass_matrix, expected, rtol=1e-10, atol=1e-12)
    env.close()
