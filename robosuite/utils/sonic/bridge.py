"""Bridge between robosuite's SonicG1 and the SONIC C++ whole-body controller.

SonicG1MotorMap locates SONIC's motors in the MuJoCo model and builds the robot state SONIC reads.
SonicRobosuiteBridge exchanges that state and SONIC's joint targets over Unitree DDS.

Motor order, used for every array here and for the SONIC_WBC action: 29 body motors, then 7
left-hand and 7 right-hand motors.
"""
import mujoco
import numpy as np

NUM_BODY_MOTORS = 29
NUM_HAND_MOTORS = 7
BODY = slice(0, NUM_BODY_MOTORS)
LEFT_HAND = slice(NUM_BODY_MOTORS, NUM_BODY_MOTORS + NUM_HAND_MOTORS)
RIGHT_HAND = slice(NUM_BODY_MOTORS + NUM_HAND_MOTORS, NUM_BODY_MOTORS + 2 * NUM_HAND_MOTORS)

# SONIC's PD gains, in motor order. They are constants of SONIC's C++ deploy, sent as float32 with
# every command: kps / kds in gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/
# policy_parameters.hpp (body) and kp 1.5 / kd 0.1 in dex3_hands.hpp (hands). The bridge raises if
# SONIC ever sends different ones.
MOTOR_KP = np.float32(
    [
        99.09843,
        99.09843,
        40.179237,
        99.09843,
        28.501245,
        28.501245,
        99.09843,
        99.09843,
        40.179237,
        99.09843,
        28.501245,
        28.501245,
        40.179237,
        28.501245,
        28.501245,
        14.250623,
        14.250623,
        14.250623,
        14.250623,
        14.250623,
        16.778328,
        16.778328,
        14.250623,
        14.250623,
        14.250623,
        14.250623,
        14.250623,
        16.778328,
        16.778328,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
    ]
).astype(float)
MOTOR_KD = np.float32(
    [
        6.3088017,
        6.3088017,
        2.5578897,
        6.3088017,
        1.8144457,
        1.8144457,
        6.3088017,
        6.3088017,
        2.5578897,
        6.3088017,
        1.8144457,
        1.8144457,
        2.5578897,
        1.8144457,
        1.8144457,
        0.90722287,
        0.90722287,
        0.90722287,
        0.90722287,
        0.90722287,
        1.0681415,
        1.0681415,
        0.90722287,
        0.90722287,
        0.90722287,
        0.90722287,
        0.90722287,
        1.0681415,
        1.0681415,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
).astype(float)

_BODY_JOINT_KEYWORDS = ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]

_dds_initialized = False  # Unitree DDS can be initialized once per process


class SonicG1MotorMap:
    """Where each SONIC motor is in the MuJoCo model, as flat arrays in motor order, plus the robot's
    floating base and torso, and the robot state SONIC reads from them."""

    def __init__(self, sim, sonic_config):
        self.sim = sim
        model = sim.model._model
        actuator_joints = model.actuator_trnid[:, 0]
        actuator_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) for joint in actuator_joints]
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) for body in range(model.nbody)]

        def actuators_named(*keywords):
            return [
                actuator
                for actuator, name in enumerate(actuator_joint_names)
                if any(keyword in name for keyword in keywords)
            ]

        # Body motors are in actuator order (the Unitree motor order). Each hand's motors are in
        # joint order, which is how gear_sonic's DDS bridge indexes the Dex3 state and command.
        body = actuators_named(*_BODY_JOINT_KEYWORDS)
        left_hand = sorted(actuators_named("left_hand"), key=lambda actuator: actuator_joints[actuator])
        right_hand = sorted(actuators_named("right_hand"), key=lambda actuator: actuator_joints[actuator])
        assert (len(body), len(left_hand), len(right_hand)) == (
            NUM_BODY_MOTORS,
            NUM_HAND_MOTORS,
            NUM_HAND_MOTORS,
        ), "SONIC_WBC needs the SonicG1 with both of its Dex3 hands"

        self.actuator_ids = np.array(body + left_hand + right_hand)
        self.joint_ids = actuator_joints[self.actuator_ids]
        self.joint_names = [actuator_joint_names[actuator] for actuator in self.actuator_ids]
        self.qpos_adr = model.jnt_qposadr[self.joint_ids]
        self.qvel_adr = model.jnt_dofadr[self.joint_ids]
        # SONIC's config lists effort limits in its source model's order: body without the right
        # arm, left hand, right arm, right hand.
        effort = np.array(sonic_config["motor_effort_limit_list"])
        self.effort_limit = np.concatenate([effort[0:22], effort[29:36], effort[22:29], effort[36:43]])

        base_joint = next(
            joint
            for joint in range(model.njnt)
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE and "pelvis" in body_names[model.jnt_bodyid[joint]]
        )
        self.pelvis_body_id = int(model.jnt_bodyid[base_joint])
        self.base_qpos_adr = int(model.jnt_qposadr[base_joint])
        self.base_qvel_adr = int(model.jnt_dofadr[base_joint])
        self.torso_body_id = next(body for body, name in enumerate(body_names) if "torso_link" in name)

    def build_obs(self):
        """The robot state SONIC reads, keyed as gear_sonic's UnitreeSdk2Bridge.PublishLowState expects."""
        model, data = self.sim.model._model, self.sim.data._data
        base_qpos, base_qvel = self.base_qpos_adr, self.base_qvel_adr
        torso_velocity = np.zeros(6)  # mj_objectVelocity returns [angular, linear]; SONIC reads [linear, angular]
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.torso_body_id, torso_velocity, 1)
        return {
            "floating_base_pose": np.array(data.qpos[base_qpos : base_qpos + 7]),
            "floating_base_vel": np.array(data.qvel[base_qvel : base_qvel + 6]),
            "floating_base_acc": np.array(data.qacc[base_qvel : base_qvel + 6]),
            "secondary_imu_quat": np.array(data.xquat[self.torso_body_id]),
            "secondary_imu_vel": np.concatenate([torso_velocity[3:6], torso_velocity[0:3]]),
            "body_q": data.qpos[self.qpos_adr[BODY]],
            "body_dq": data.qvel[self.qvel_adr[BODY]],
            "body_ddq": data.qacc[self.qvel_adr[BODY]],
            "body_tau_est": data.actuator_force[self.actuator_ids[BODY]],
            "left_hand_q": data.qpos[self.qpos_adr[LEFT_HAND]],
            "left_hand_dq": data.qvel[self.qvel_adr[LEFT_HAND]],
            "right_hand_q": data.qpos[self.qpos_adr[RIGHT_HAND]],
            "right_hand_dq": data.qvel[self.qvel_adr[RIGHT_HAND]],
            "time": float(data.time),
        }


class SonicRobosuiteBridge:
    """Sends the robot state to the SONIC C++ controller and reads back its joint targets, over
    Unitree DDS (gear_sonic's UnitreeSdk2Bridge)."""

    def __init__(self, sonic_config):
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        global _dds_initialized
        if not _dds_initialized:
            ChannelFactoryInitialize(sonic_config["DOMAIN_ID"], sonic_config["INTERFACE"])
            _dds_initialized = True
        self.unitree_dds = UnitreeSdk2Bridge(sonic_config)

    def reset(self):
        """Forget the commands received so far (call after an env reset)."""
        self.unitree_dds.reset()

    def get_sonic_action(self, env):
        """Publish the robot state to SONIC and return its 43 joint targets in motor order, or None
        while SONIC is not commanding the robot: before its body and both hand commands have
        arrived, and after its damping-only stop command (zero kp)."""
        dds = self.unitree_dds
        dds.PublishLowState(env.robots[0].composite_controller.motor_map.build_obs())

        with dds.low_cmd_lock, dds.left_hand_cmd_lock, dds.right_hand_cmd_lock:
            if not (dds.low_cmd_received and dds.left_hand_cmd_received and dds.right_hand_cmd_received):
                return None
            motor_cmds = (
                list(dds.low_cmd.motor_cmd)[:NUM_BODY_MOTORS]
                + list(dds.left_hand_cmd.motor_cmd)[:NUM_HAND_MOTORS]
                + list(dds.right_hand_cmd.motor_cmd)[:NUM_HAND_MOTORS]
            )
        kp = np.array([motor_cmd.kp for motor_cmd in motor_cmds])
        kd = np.array([motor_cmd.kd for motor_cmd in motor_cmds])
        if not kp[BODY].any():
            return None
        if not (np.array_equal(kp, MOTOR_KP) and np.array_equal(kd, MOTOR_KD)):
            raise RuntimeError(
                "SONIC is sending PD gains that differ from robosuite's MOTOR_KP / MOTOR_KD. SONIC_WBC "
                "supports only SONIC's default gains (no --motor-kp-scale / --motor-kd-scale)."
            )
        return np.array([motor_cmd.q for motor_cmd in motor_cmds])
