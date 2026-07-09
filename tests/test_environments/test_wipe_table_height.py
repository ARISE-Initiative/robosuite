"""
Test that the Wipe environment's randomized table height does not drift across
resets (issue #815). The table z-offset should stay centered on the base height
with per-episode variation, instead of accumulating a fixed delta every reset.
"""
import robosuite as suite
from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG


def test_wipe_table_height_no_drift():
    config = {k: v for k, v in DEFAULT_WIPE_CONFIG.items()}
    config["table_height_std"] = 0.02
    base_z = config["table_offset"][2]

    env = suite.make(
        "Wipe",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        task_config=config,
    )
    try:
        heights = []
        for _ in range(6):
            env.reset()
            heights.append(float(env.table_offset[2]))
    finally:
        env.close()

    # the height must not accumulate (ramp) across resets
    is_monotonic = all(heights[i] < heights[i + 1] for i in range(len(heights) - 1))
    assert not is_monotonic, f"table height drifts across resets: {heights}"

    # it should stay centered on the base height
    assert all(abs(h - base_z) < 0.1 for h in heights), heights

    # and vary per episode when a std is set
    assert len(set(heights)) > 1, heights
