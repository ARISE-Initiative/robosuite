"""
Convenience script to tune a camera view in a mujoco environment.
Allows keyboard presses to move a camera around in the viewer, and
then prints the final position and quaternion you should set
for your camera in the mujoco XML file.
"""

import time
import argparse
import glfw
import xml.etree.ElementTree as ET
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import find_elements, find_parent
from robosuite.utils.camera_utils import CameraMover

# some settings
DELTA_POS_KEY_PRESS = 0.05      # delta camera position per key press
DELTA_ROT_KEY_PRESS = 1         # delta camera angle per key press


class KeyboardHandler:
    def __init__(self, camera_mover):
        """
        Store internal state here.

        Args:
            camera_mover (CameraMover): Playback camera class
        cam_body_id (int): id corresponding to parent body of camera element
        """
        self.camera_mover = camera_mover

    def on_press(self, window, key, scancode, action, mods):
        """
        Key handler for key presses.

        Args:
            window: [NOT USED]
            key (int): keycode corresponding to the key that was pressed
            scancode: [NOT USED]
            action: [NOT USED]
            mods: [NOT USED]
        """
        # controls for moving position
        if key == glfw.KEY_W:
            # move forward
            self.camera_mover.move_camera(direction=[0., 0., -1.], scale=DELTA_POS_KEY_PRESS)
        elif key == glfw.KEY_S:
            # move backward
            self.camera_mover.move_camera(direction=[0., 0., 1.], scale=DELTA_POS_KEY_PRESS)
        elif key == glfw.KEY_A:
            # move left
            self.camera_mover.move_camera(direction=[-1., 0., 0.], scale=DELTA_POS_KEY_PRESS)
        elif key == glfw.KEY_D:
            # move right
            self.camera_mover.move_camera(direction=[1., 0., 0.], scale=DELTA_POS_KEY_PRESS)
        elif key == glfw.KEY_R:
            # move up
            self.camera_mover.move_camera(direction=[0., 1., 0.], scale=DELTA_POS_KEY_PRESS)
        elif key == glfw.KEY_F:
            # move down
            self.camera_mover.move_camera(direction=[0., -1., 0.], scale=DELTA_POS_KEY_PRESS)

        # controls for moving rotation
        elif key == glfw.KEY_UP:
            # rotate up
            self.camera_mover.rotate_camera(point=None, axis=[1., 0., 0.], angle=DELTA_ROT_KEY_PRESS)
        elif key == glfw.KEY_DOWN:
            # rotate down
            self.camera_mover.rotate_camera(point=None, axis=[-1., 0., 0.], angle=DELTA_ROT_KEY_PRESS)
        elif key == glfw.KEY_LEFT:
            # rotate left
            self.camera_mover.rotate_camera(point=None, axis=[0., 1., 0.], angle=DELTA_ROT_KEY_PRESS)
        elif key == glfw.KEY_RIGHT:
            # rotate right
            self.camera_mover.rotate_camera(point=None, axis=[0., -1., 0.], angle=DELTA_ROT_KEY_PRESS)
        elif key == glfw.KEY_PERIOD:
            # rotate counterclockwise
            self.camera_mover.rotate_camera(point=None, axis=[0., 0., 1.], angle=DELTA_ROT_KEY_PRESS)
        elif key == glfw.KEY_SLASH:
            # rotate clockwise
            self.camera_mover.rotate_camera(point=None, axis=[0., 0., -1.], angle=DELTA_ROT_KEY_PRESS)

    def on_release(self, window, key, scancode, action, mods):
        """
        Key handler for key releases.

        Args:
            window: [NOT USED]
            key: [NOT USED]
            scancode: [NOT USED]
            action: [NOT USED]
            mods: [NOT USED]
        """
        pass


def print_command(char, info):
    """
    Prints out the command + relevant info entered by user

    Args:
        char (str): Command entered
        info (str): Any additional info to print
    """
    char += " " * (10 - len(char))
    print("{}\t{}".format(char, info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Sawyer", help="Which robot(s) to use in the env")
    args = parser.parse_args()

    print("\nWelcome to the camera tuning script! You will be able to tune a camera view")
    print("by moving it around using your keyboard. The controls are printed below.")

    print("")
    print_command("Keys", "Command")
    print_command("w-s", "zoom the camera in/out")
    print_command("a-d", "pan the camera left/right")
    print_command("r-f", "pan the camera up/down")
    print_command("arrow keys", "rotate the camera to change view direction")
    print_command(".-/", "rotate the camera view without changing view direction")
    print("")

    # read camera XML tag from user input
    inp = input("\nPlease paste a camera name below \n"
                "OR xml tag below (e.g. <camera ... />) \n"
                "OR leave blank for an example:\n")

    if len(inp) == 0:
        if args.env != "Lift":
            raise Exception("ERROR: env must be Lift to run default example.")
        print("\nUsing an example tag corresponding to the frontview camera.")
        print("This xml tag was copied from robosuite/models/assets/arenas/table_arena.xml")
        inp = '<camera mode="fixed" name="frontview" pos="1.6 0 1.45" quat="0.56 0.43 0.43 0.56"/>'

    # remember the tag and infer some properties
    from_tag = "<" in inp
    notify_str = "NOTE: using the following xml tag:\n" if from_tag else \
        "NOTE: using the following camera (initialized at default sim location)\n"

    print(notify_str)
    print("{}\n".format(inp))

    cam_tree = ET.fromstring(inp) if from_tag else ET.Element("camera", attrib={"name": inp})
    CAMERA_NAME = cam_tree.get("name")

    # make the environment
    env = robosuite.make(
        args.env,
        robots=args.robots,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )
    env.reset()

    # Create the camera mover
    camera_mover = CameraMover(
        env=env,
        camera=CAMERA_NAME,
    )

    # Infer initial camera pose
    if from_tag:
        initial_file_camera_pos = np.array(cam_tree.get("pos").split(" ")).astype(float)
        initial_file_camera_quat = T.convert_quat(np.array(cam_tree.get("quat").split(" ")).astype(float), to='xyzw')
        # Set these values as well
        camera_mover.set_camera_pose(pos=initial_file_camera_pos, quat=initial_file_camera_quat)
    else:
        initial_file_camera_pos, initial_file_camera_quat = camera_mover.get_camera_pose()
    # Define initial file camera pose
    initial_file_camera_pose = T.make_pose(initial_file_camera_pos, T.quat2mat(initial_file_camera_quat))

    # remember difference between camera pose in initial tag and absolute camera pose in world
    initial_world_camera_pos, initial_world_camera_quat = camera_mover.get_camera_pose()
    initial_world_camera_pose = T.make_pose(initial_world_camera_pos, T.quat2mat(initial_world_camera_quat))
    world_in_file = initial_file_camera_pose.dot(T.pose_inv(initial_world_camera_pose))

    # Make sure we're using the camera that we're modifying
    camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
    env.viewer.set_camera(camera_id=camera_id)

    # register callbacks to handle key presses in the viewer
    key_handler = KeyboardHandler(camera_mover=camera_mover)
    env.viewer.add_keypress_callback("any", key_handler.on_press)
    env.viewer.add_keyup_callback("any", key_handler.on_release)
    env.viewer.add_keyrepeat_callback("any", key_handler.on_press)

    # just spin to let user interact with glfw window
    spin_count = 0
    while True:
        action = np.zeros(env.action_dim)
        obs, reward, done, _ = env.step(action)
        env.render()
        spin_count += 1
        if spin_count % 500 == 0:
            # convert from world coordinates to file coordinates (xml subtree)
            camera_pos, camera_quat = camera_mover.get_camera_pose()
            world_camera_pose = T.make_pose(camera_pos, T.quat2mat(camera_quat))
            file_camera_pose = world_in_file.dot(world_camera_pose)
            # TODO: Figure out why numba causes black screen of death (specifically, during mat2pose --> mat2quat call below)
            camera_pos, camera_quat = T.mat2pose(file_camera_pose)
            camera_quat = T.convert_quat(camera_quat, to='wxyz')

            print("\n\ncurrent camera tag you should copy")
            cam_tree.set("pos", "{} {} {}".format(camera_pos[0], camera_pos[1], camera_pos[2]))
            cam_tree.set("quat", "{} {} {} {}".format(camera_quat[0], camera_quat[1], camera_quat[2], camera_quat[3]))
            print(ET.tostring(cam_tree, encoding="utf8").decode("utf8"))




