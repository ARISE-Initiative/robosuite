from robosuite.models.grippers import GripperTester, Robotiq140Gripper


def test_robotiq():
    robotiq_tester(False)


def robotiq_tester(render, total_iters=1, test_y=True):
    gripper = Robotiq140Gripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=0.03,
        gripper_high_pos=0.1,
        box_size=[0.015] * 3,
        render=render,
    )
    tester.start_simulation()
    tester.viewer.set_camera(0)
    tester.loop(total_iters=total_iters, test_y=test_y)
    tester.close()


if __name__ == "__main__":
    robotiq_tester(True, 20, False)
