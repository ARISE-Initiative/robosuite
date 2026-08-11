"""Regression tests for the Jaco three-finger gripper sensors."""

from pathlib import Path

import mujoco
import numpy as np


ASSET_PATH = (
    Path(__file__).parents[2] / "robosuite" / "models" / "assets" / "grippers" / "jaco_three_finger_gripper.xml"
)


def test_force_torque_sensors_are_well_conditioned():
    model = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    readings = []
    for name in ("force_ee", "torque_ee"):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        site_id = model.sensor_objid[sensor_id]
        site_body_id = model.site_bodyid[site_id]
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, site_body_id) == "right_gripper"

        start = model.sensor_adr[sensor_id]
        readings.append(data.sensordata[start : start + model.sensor_dim[sensor_id]])

    readings = np.concatenate(readings)
    assert np.all(np.isfinite(readings))
    assert np.max(np.abs(readings)) < 1e3
