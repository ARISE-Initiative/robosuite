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

# some settings
DELTA_POS_KEY_PRESS = 0.05      # delta camera position per key press
DELTA_ROT_KEY_PRESS = 1         # delta camera angle per key press


def modify_xml_for_camera_movement(xml, camera_name):
    """
    Cameras in mujoco are 'fixed', so they can't be moved by default.
    Although it's possible to hack position movement, rotation movement
    does not work. An alternative is to attach a camera to a body element,
    and move the body.

    This function modifies the camera with name @camera_name in the xml
    by attaching it to a body element that we can then manipulate. In this
    way, we can move the camera by moving the body.

    See http://www.mujoco.org/forum/index.php?threads/move-camera.2201/ for
    further details.

    xml (str): Mujoco sim XML file as a string
    camera_name (str): Name of camera to tune
    """
    tree = ET.fromstring(xml)

    # find the correct camera
    camera_elem = None
    cameras = find_elements(root=tree, tags="camera", return_first=False)
    for camera in cameras:
        if camera.get("name") == camera_name:
            camera_elem = camera
            break
    assert camera_elem is not None, "No valid camera name found, options are: {}"\
        .format([camera.get("name") for camera in cameras])

    # Find parent element of the camera element
    parent = find_parent(root=tree, child=camera_elem)
    assert parent is not None

    # add camera body
    cam_body = ET.SubElement(parent, "body")
    cam_body.set("name", "cameramover")
    cam_body.set("pos", camera_elem.get("pos"))
    cam_body.set("quat", camera_elem.get("quat"))
    new_camera = ET.SubElement(cam_body, "camera")
    new_camera.set("mode", "fixed")
    new_camera.set("name", camera_elem.get("name"))
    new_camera.set("pos", "0 0 0")
    # Also need to define inertia
    inertial = ET.SubElement(cam_body, "inertial")
    inertial.set("diaginertia", "1e-08 1e-08 1e-08")
    inertial.set("mass", "1e-08")
    inertial.set("pos", "0 0 0")

    # remove old camera element
    parent.remove(camera_elem)

    return ET.tostring(tree, encoding="utf8").decode("utf8")


def move_camera(env, direction, scale, cam_body_id):
    """
    Move the camera view along a direction (in the camera frame).

    Args:
        direction (np.arry): 3-array for where to move camera in camera frame
        scale (float): how much to move along that direction
        cam_body_id (int): id corresponding to parent body of camera element
    """

    # current camera pose
    camera_pos = np.array(env.sim.model.body_pos[cam_body_id])
    camera_rot = T.quat2mat(T.convert_quat(env.sim.model.body_quat[cam_body_id], to='xyzw'))

    # move along camera frame axis and set new position
    camera_pos += scale * camera_rot.dot(direction)
    env.sim.model.body_pos[cam_body_id] = camera_pos
    env.sim.forward()


def rotate_camera(env, direction, angle, cam_body_id):
    """
    Rotate the camera view about a direction (in the camera frame).

    Args:
        direction (np.array): 3-array for where to move camera in camera frame
        angle (float): how much to rotate about that direction
        cam_body_id (int): id corresponding to parent body of camera element
    """

    # current camera rotation
    camera_pos = np.array(env.sim.model.body_pos[cam_body_id])
    camera_rot = T.quat2mat(T.convert_quat(env.sim.model.body_quat[cam_body_id], to='xyzw'))

    # rotate by angle and direction to get new camera rotation
    rad = np.pi * angle / 180.0
    R = T.rotation_matrix(rad, direction, point=None)
    camera_rot = camera_rot.dot(R[:3, :3])

    # set new rotation
    env.sim.model.body_quat[cam_body_id] = T.convert_quat(T.mat2quat(camera_rot), to='wxyz')
    env.sim.forward()


class KeyboardHandler:
    def __init__(self, env, cam_body_id):
        """
        Store internal state here.

        Args:
            env (MujocoEnv): Environment to use
        cam_body_id (int): id corresponding to parent body of camera element
        """
        self.env = env
        self.cam_body_id = cam_body_id

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
            move_camera(env=self.env, direction=[0., 0., -1.], scale=DELTA_POS_KEY_PRESS, cam_body_id=self.cam_body_id)
        elif key == glfw.KEY_S:
            # move backward
            move_camera(env=self.env, direction=[0., 0., 1.], scale=DELTA_POS_KEY_PRESS, cam_body_id=self.cam_body_id)
        elif key == glfw.KEY_A:
            # move left
            move_camera(env=self.env, direction=[-1., 0., 0.], scale=DELTA_POS_KEY_PRESS, cam_body_id=self.cam_body_id)
        elif key == glfw.KEY_D:
            # move right
            move_camera(env=self.env, direction=[1., 0., 0.], scale=DELTA_POS_KEY_PRESS, cam_body_id=self.cam_body_id)
        elif key == glfw.KEY_R:
            # move up
            move_camera(env=self.env, direction=[0., 1., 0.], scale=DELTA_POS_KEY_PRESS, cam_body_id=self.cam_body_id)
        elif key == glfw.KEY_F:
            # move down
            move_camera(env=self.env, direction=[0., -1., 0.], scale=DELTA_POS_KEY_PRESS, cam_body_id=self.cam_body_id)

        # controls for moving rotation
        elif key == glfw.KEY_UP:
            # rotate up
            rotate_camera(env=self.env, direction=[1., 0., 0.], angle=DELTA_ROT_KEY_PRESS, cam_body_id=self.cam_body_id)
        elif key == glfw.KEY_DOWN:
            # rotate down
            rotate_camera(env=self.env, direction=[-1., 0., 0.], angle=DELTA_ROT_KEY_PRESS, cam_body_id=self.cam_body_id)
        elif key == glfw.KEY_LEFT:
            # rotate left
            rotate_camera(env=self.env, direction=[0., 1., 0.], angle=DELTA_ROT_KEY_PRESS, cam_body_id=self.cam_body_id)
        elif key == glfw.KEY_RIGHT:
            # rotate right
            rotate_camera(env=self.env, direction=[0., -1., 0.], angle=DELTA_ROT_KEY_PRESS, cam_body_id=self.cam_body_id)
        elif key == glfw.KEY_PERIOD:
            # rotate counterclockwise
            rotate_camera(env=self.env, direction=[0., 0., 1.], angle=DELTA_ROT_KEY_PRESS, cam_body_id=self.cam_body_id)
        elif key == glfw.KEY_SLASH:
            # rotate clockwise
            rotate_camera(env=self.env, direction=[0., 0., -1.], angle=DELTA_ROT_KEY_PRESS, cam_body_id=self.cam_body_id)

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
    initial_mjstate = env.sim.get_state().flatten()
    xml = env.sim.model.get_xml()

    # add body to camera to be able to move it around
    xml = modify_xml_for_camera_movement(xml, camera_name=CAMERA_NAME)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(initial_mjstate)
    env.sim.forward()

    camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
    env.viewer.set_camera(camera_id=camera_id)

    # Store camera mover id
    cam_body_id = env.sim.model.body_name2id("cameramover")

    # Infer initial camera pose
    if from_tag:
        initial_file_camera_pos = np.array(cam_tree.get("pos").split(" ")).astype(float)
        initial_file_camera_quat = T.convert_quat(np.array(cam_tree.get("quat").split(" ")).astype(float), to='xyzw')
    else:
        initial_file_camera_pos = np.array(env.sim.model.body_pos[cam_body_id])
        initial_file_camera_quat = T.convert_quat(np.array(env.sim.model.body_quat[cam_body_id]), to='xyzw')
    initial_file_camera_pose = T.make_pose(initial_file_camera_pos, T.quat2mat(initial_file_camera_quat))

    # remember difference between camera pose in initial tag
    # and absolute camera pose in world
    initial_world_camera_pos = np.array(env.sim.model.body_pos[cam_body_id])
    initial_world_camera_quat = T.convert_quat(env.sim.model.body_quat[cam_body_id], to='xyzw')
    initial_world_camera_pose = T.make_pose(initial_world_camera_pos, T.quat2mat(initial_world_camera_quat))
    world_in_file = initial_file_camera_pose.dot(T.pose_inv(initial_world_camera_pose))

    # register callbacks to handle key presses in the viewer
    key_handler = KeyboardHandler(env=env, cam_body_id=cam_body_id)
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
            camera_pos = np.array(env.sim.model.body_pos[cam_body_id])
            camera_quat = T.convert_quat(env.sim.model.body_quat[cam_body_id], to='xyzw')
            world_camera_pose = T.make_pose(camera_pos, T.quat2mat(camera_quat))
            file_camera_pose = world_in_file.dot(world_camera_pose)
            # TODO: Figure out why numba causes black screen of death (specifically, during mat2pose --> mat2quat call below)
            camera_pos, camera_quat = T.mat2pose(file_camera_pose)
            camera_quat = T.convert_quat(camera_quat, to='wxyz')

            print("\n\ncurrent camera tag you should copy")
            cam_tree.set("pos", "{} {} {}".format(camera_pos[0], camera_pos[1], camera_pos[2]))
            cam_tree.set("quat", "{} {} {} {}".format(camera_quat[0], camera_quat[1], camera_quat[2], camera_quat[3]))
            print(ET.tostring(cam_tree, encoding="utf8").decode("utf8"))




