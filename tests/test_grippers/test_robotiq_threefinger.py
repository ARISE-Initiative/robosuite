from robosuite.models.grippers import GripperTester, RobotiqThreeFingerGripper


def test_robotiq_three_finger():
    robotiq_three_finger_tester(False)


def robotiq_three_finger_tester(render,
                                total_iters=1,
                                test_y=True):
    gripper = RobotiqThreeFingerGripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=-0.02,
        gripper_high_pos=0.1,
        box_size=[0.035] * 3,
        box_density=500,
        render=render,
    )
    tester.start_simulation()
    tester.loop(total_iters=total_iters,
                test_y=test_y)
    tester.close()


if __name__ == "__main__":
    robotiq_three_finger_tester(True, 20, False)
