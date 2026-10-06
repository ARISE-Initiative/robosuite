"""
Tests the joint address helpers in robosuite.utils.binding_utils.

MuJoCo 3.12 and later return jnt_type entries as numpy integers that compare unequal to
mujoco.mjtJoint members when the enum is on the left-hand side (as in ``x in (enum, enum)``).
This checks that get_joint_qpos_addr / get_joint_qvel_addr handle every joint type.
"""
from robosuite.utils.binding_utils import MjSim

XML = """
<mujoco>
  <worldbody>
    <body name="free_body"><freejoint name="free"/><geom size="0.1"/></body>
    <body name="ball_body" pos="1 0 0"><joint name="ball" type="ball"/><geom size="0.1"/></body>
    <body name="hinge_body" pos="2 0 0"><joint name="hinge" type="hinge"/><geom size="0.1"/></body>
    <body name="slide_body" pos="3 0 0"><joint name="slide" type="slide"/><geom size="0.1"/></body>
  </worldbody>
</mujoco>
"""


def test_joint_addrs_all_joint_types():
    model = MjSim.from_xml_string(XML).model
    # qpos layout: free (7), ball (4), hinge (1), slide (1)
    assert model.get_joint_qpos_addr("free") == (0, 7)
    assert model.get_joint_qpos_addr("ball") == (7, 11)
    assert model.get_joint_qpos_addr("hinge") == 11
    assert model.get_joint_qpos_addr("slide") == 12
    # qvel layout: free (6), ball (3), hinge (1), slide (1)
    assert model.get_joint_qvel_addr("free") == (0, 6)
    assert model.get_joint_qvel_addr("ball") == (6, 9)
    assert model.get_joint_qvel_addr("hinge") == 9
    assert model.get_joint_qvel_addr("slide") == 10
