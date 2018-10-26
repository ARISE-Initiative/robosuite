from robosuite.models.grippers import GripperTester, PR2Gripper


def test_pr2():
    pr2_tester(False)


def pr2_tester(render,
               total_iters=1,
               test_y=True):
    gripper = PR2Gripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=-0.02,
        gripper_high_pos=0.05,
        render=render,
    )
    tester.start_simulation()
    tester.loop(total_iters=total_iters,
                test_y=test_y)


if __name__ == "__main__":
    pr2_tester(True, 20, False)
