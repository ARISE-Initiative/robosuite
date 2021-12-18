"""
Tests panda gripper on grabbing task
"""
from robosuite.models.grippers import GripperTester, PandaGripper


def test_panda_gripper():
    panda_gripper_tester(False)


def panda_gripper_tester(render, total_iters=1, test_y=True):
    gripper = PandaGripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=-0.10,
        gripper_high_pos=0.01,
        render=render,
    )
    tester.start_simulation()
    tester.loop(total_iters=total_iters, test_y=test_y)
    tester.close()


if __name__ == "__main__":
    panda_gripper_tester(True, 20, True)
    panda_gripper_tester(True, 20, True)
