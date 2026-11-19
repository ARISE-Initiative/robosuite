"""SonicWholeBodyController ("SONIC_WBC"): applies SONIC's joint targets with SONIC's PD law.

The env action is the joint-position target of every motor, in motor order (29 body, 7 left hand,
7 right hand; see robosuite.utils.sonic.bridge). Each part's JointPositionVelocityController applies

    tau = kp * (target - q) - kd * dq,    clipped to the motor's effort limit

with SONIC's constant gains. An all-zero action means "no SONIC command" and applies no torque:
robocasa's kitchen reset settles the scene with zero-action steps, and live drivers send zeros
until SONIC engages. A startup elastic band holds the pelvis at its spawn pose until release_band().
"""
import os

import mujoco
import numpy as np
from robosuite.controllers.composite.composite_controller import (
    CompositeController, register_composite_controller)
from robosuite.utils.sonic.bridge import MOTOR_KD, MOTOR_KP, NUM_HAND_MOTORS, SonicG1MotorMap
from scipy.spatial.transform import Rotation

# Startup band spring and damper gains (gear_sonic's ElasticBand)
BAND_KP_POS, BAND_KD_POS, BAND_KP_ANG, BAND_KD_ANG = 10000, 1000, 1000, 10


@register_composite_controller
class SonicWholeBodyController(CompositeController):
    name = "SONIC_WBC"

    def __init__(self, sim, robot_model, grippers):
        super().__init__(sim, robot_model, grippers)
        import gear_sonic
        import yaml

        # joint and effort limits, and the DDS settings SonicRobosuiteBridge needs
        config_path = os.path.join(os.path.dirname(gear_sonic.__file__),
                                   "utils", "mujoco_sim", "wbc_configs", "g1_29dof_sonic_model12.yaml")
        with open(config_path) as config_file:
            self.sonic_config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.motor_map = SonicG1MotorMap(sim, self.sonic_config)
        self.part_motors = {}  # part name -> motor index of each of the part's joints
        self.action = np.zeros(len(MOTOR_KP))
        self.band_enabled = True
        self._band_anchor = None  # pelvis (position, rotation) when the band was first applied

    def _init_controllers(self):
        super()._init_controllers()
        motor_of_joint = {int(joint): motor for motor, joint in enumerate(self.motor_map.joint_ids)}
        for part, part_controller in self.part_controllers.items():
            motors = np.array([motor_of_joint[int(joint)] for joint in part_controller.joint_index])
            self.part_motors[part] = motors
            effort = self.motor_map.effort_limit[motors]
            part_controller.torque_limits = np.array([-effort, effort])
            # SONIC adds no gravity compensation (set here: a config entry would not reach the gripper parts)
            part_controller.use_torque_compensation = False

    @property
    def action_limits(self):
        hand_limit = np.full(2 * NUM_HAND_MOTORS, np.pi)
        low = np.concatenate([self.sonic_config["motor_pos_lower_limit_list"], -hand_limit])
        high = np.concatenate([self.sonic_config["motor_pos_upper_limit_list"], hand_limit])
        return low, high

    def set_goal(self, all_action):
        self.action = np.asarray(all_action, dtype=float)

    def update_state(self):
        pass  # the parts are joint-space controllers; they have no base frame to update

    def release_band(self):
        self.band_enabled = False

    def toggle_band(self):
        self.band_enabled = not self.band_enabled

    def _apply_band(self):
        """Hold the pelvis at its spawn pose with a spring-damper force and torque (xfrc_applied
        only), so the floating base stays up until SONIC is balancing."""
        model, data = self.sim.model._model, self.sim.data._data
        pelvis = self.motor_map.pelvis_body_id
        if not self.band_enabled:
            data.xfrc_applied[pelvis] = 0.0
            return
        quat = data.xquat[pelvis]
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        if self._band_anchor is None:
            self._band_anchor = (np.array(data.xpos[pelvis]), rotation)
        anchor_position, anchor_rotation = self._band_anchor
        velocity = np.zeros(6)  # [angular, linear] in the world frame
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, velocity, 0)
        force = BAND_KP_POS * (anchor_position - data.xpos[pelvis]) - BAND_KD_POS * velocity[3:6]
        torque = -BAND_KP_ANG * (rotation * anchor_rotation.inv()).as_rotvec() - BAND_KD_ANG * velocity[0:3]
        data.xfrc_applied[pelvis] = np.concatenate([force, torque])

    def run_controller(self, enabled_parts):
        self._apply_band()
        if not self.action.any():
            self.sim.data.ctrl[self.motor_map.actuator_ids] = 0.0
            return {}
        torques = {}
        for part, part_controller in self.part_controllers.items():
            if enabled_parts.get(part, False):
                motors = self.part_motors[part]
                part_controller.set_pd_command(self.action[motors], kp=MOTOR_KP[motors], kd=MOTOR_KD[motors])
                torques[part] = part_controller.run_controller()
        return torques
