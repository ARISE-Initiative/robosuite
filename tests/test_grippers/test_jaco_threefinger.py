from robosuite.models.grippers import GripperTester, JacoThreeFingerGripper


def test_robotiq():
    robotiq_tester(False)


def robotiq_tester(render,
                   total_iters=1,
                   test_y=True):
    gripper = JacoThreeFingerGripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=0.01,
        gripper_high_pos=0.1,
        box_size=[0.025] * 3,
        render=render,
    )
    tester.start_simulation()
    tester.loop(total_iters=total_iters,
                test_y=test_y)


if __name__ == "__main__":
    robotiq_tester(True, 20, False)
