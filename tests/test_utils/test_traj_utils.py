"""Check linear interpolation setpoints without running a robot simulation."""

import numpy as np
import pytest

from robosuite.utils.traj_utils import LinearInterpolator


@pytest.mark.parametrize("total_steps", [1, 2, 5])
@pytest.mark.parametrize("start, goal", [([0.0], [10.0]), ([2.0, -3.0, 4.0], [-3.0, 7.0, 4.0])])
def test_linear_position_setpoints(total_steps, start, goal):
    start, goal = np.array(start), np.array(goal)
    interpolator = LinearInterpolator(len(start), total_steps * 10, 10, ramp_ratio=1.0)
    interpolator.set_goal(start)
    # Finish the first ramp before starting one from a nonzero position.
    for _ in range(total_steps):
        interpolator.get_interpolated_goal()
    interpolator.set_goal(goal)

    for step in range(1, total_steps + 1):
        expected = start + (goal - start) * step / total_steps
        np.testing.assert_allclose(interpolator.get_interpolated_goal(), expected)

    # Repeated controller calls after the ramp must hold the goal.
    for _ in range(3):
        np.testing.assert_allclose(interpolator.get_interpolated_goal(), goal)
    np.testing.assert_array_equal(interpolator.start, start)
    np.testing.assert_array_equal(interpolator.goal, goal)


def test_linear_position_new_goal_restarts_ramp():
    interpolator = LinearInterpolator(1, 50, 10, ramp_ratio=1.0)
    interpolator.set_goal(np.array([10.0]))
    for _ in range(5):
        interpolator.get_interpolated_goal()
    interpolator.set_goal(np.array([-5.0]))
    np.testing.assert_allclose(interpolator.get_interpolated_goal(), [7.0])
    np.testing.assert_allclose(interpolator.get_interpolated_goal(), [4.0])


def test_linear_position_fractional_step_count():
    interpolator = LinearInterpolator(1, 100, 30, ramp_ratio=1.0)
    interpolator.set_goal(np.array([8.0]))
    np.testing.assert_equal(interpolator.total_steps, 4)
    for expected in [2.0, 4.0, 6.0, 8.0, 8.0]:
        np.testing.assert_allclose(interpolator.get_interpolated_goal(), [expected])


@pytest.mark.parametrize("orientation, dim", [("euler", 3), ("quat", 4)])
def test_linear_orientation_still_reaches_goal(orientation, dim):
    goal = np.array([0.0, 0.0, np.pi / 2]) if dim == 3 else np.array([0.0, 0.0, 1.0, 0.0])
    interpolator = LinearInterpolator(dim, 50, 10, ramp_ratio=1.0, ori_interpolate=orientation)
    interpolator.set_goal(goal)
    for _ in range(5):
        result = interpolator.get_interpolated_goal()
    np.testing.assert_allclose(result, goal, atol=1e-7)
    np.testing.assert_allclose(interpolator.get_interpolated_goal(), goal, atol=1e-7)
