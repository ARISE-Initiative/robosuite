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

# some settings
DELTA_POS_KEY_PRESS = 0.05 # delta camera position per key press
DELTA_ROT_KEY_PRESS = 1 # delta camera angle per key press

def modify_xml_for_camera_movement(xml, camera_name):
    """
    Cameras in mujoco are 'fixed', so they can't be moved by default.
    Although it's possible to hack position movement, rotation movement
    does not work. An alternative is to attach a camera to a mocap body,
    and move the mocap body.

    This function modifies the camera with name @camera_name in the xml
    by attaching it to a mocap body that can move around freely. In this
    way, we can move the camera by moving the mocap body.

    See http://www.mujoco.org/forum/index.php?threads/move-camera.2201/ for
    further details.
    """
    tree = ET.fromstring(xml)
    wb = tree.find("worldbody")

    # find the correct camera
    camera_elem = None
    cameras = wb.findall("camera")
    for camera in cameras:
        if camera.get("name") == camera_name:
            camera_elem = camera
            break
    assert(camera_elem is not None)

    # add mocap body
    mocap = ET.SubElement(wb, "body")
    mocap.set("name", "cameramover")
    mocap.set("mocap", "true")
    mocap.set("pos", camera.get("pos"))
    mocap.set("quat", camera.get("quat"))
    new_camera = ET.SubElement(mocap, "camera")
    new_camera.set("mode", "fixed")
    new_camera.set("name", camera.get("name"))
    new_camera.set("pos", "0 0 0")

    # remove old camera element
    wb.remove(camera_elem)

    return ET.tostring(tree, encoding="utf8").decode("utf8")

def move_camera(env, direction, scale, camera_id):
    """
    Move the camera view along a direction (in the camera frame).
    :param direction: a 3-dim numpy array for where to move camera in camera frame
    :param scale: a float for how much to move along that direction
    :param camera_id: which camera to modify
    """

    # current camera pose
    camera_pos = np.array(env.sim.data.get_mocap_pos("cameramover"))
    camera_rot = T.quat2mat(T.convert_quat(env.sim.data.get_mocap_quat("cameramover"), to='xyzw'))

    # move along camera frame axis and set new position
    camera_pos += scale * camera_rot.dot(direction) 
    env.sim.data.set_mocap_pos("cameramover", camera_pos)
    env.sim.forward()

def rotate_camera(env, direction, angle, camera_id):
    """
    Rotate the camera view about a direction (in the camera frame).
    :param direction: a 3-dim numpy array for where to move camera in camera frame
    :param angle: a float for how much to rotate about that direction
    :param camera_id: which camera to modify
    """

    # current camera rotation
    camera_rot = T.quat2mat(T.convert_quat(env.sim.data.get_mocap_quat("cameramover"), to='xyzw'))

    # rotate by angle and direction to get new camera rotation
    rad = np.pi * angle / 180.0
    R = T.rotation_matrix(rad, direction, point=None)
    camera_rot = camera_rot.dot(R[:3, :3])

    # set new rotation
    env.sim.data.set_mocap_quat("cameramover", T.convert_quat(T.mat2quat(camera_rot), to='wxyz'))
    env.sim.forward()


class KeyboardHandler:
    def __init__(self, env, camera_id):
        """
        Store internal state here.
        """
        self.env = env
        self.camera_id = camera_id

    def on_press(self, window, key, scancode, action, mods):
        """
        Key handler for key presses.
        """

        # controls for moving position
        if key == glfw.KEY_W:
            # move forward
            move_camera(env=self.env, direction=[0., 0., -1.], scale=DELTA_POS_KEY_PRESS, camera_id=self.camera_id)
        elif key == glfw.KEY_S:
            # move backward
            move_camera(env=self.env, direction=[0., 0., 1.], scale=DELTA_POS_KEY_PRESS, camera_id=self.camera_id)
        elif key == glfw.KEY_A:
            # move left
            move_camera(env=self.env, direction=[-1., 0., 0.], scale=DELTA_POS_KEY_PRESS, camera_id=self.camera_id)
        elif key == glfw.KEY_D:
            # move right
            move_camera(env=self.env, direction=[1., 0., 0.], scale=DELTA_POS_KEY_PRESS, camera_id=self.camera_id)
        elif key == glfw.KEY_R:
            # move up
            move_camera(env=self.env, direction=[0., 1., 0.], scale=DELTA_POS_KEY_PRESS, camera_id=self.camera_id)
        elif key == glfw.KEY_F:
            # move down
            move_camera(env=self.env, direction=[0., -1., 0.], scale=DELTA_POS_KEY_PRESS, camera_id=self.camera_id)


        # controls for moving rotation
        elif key == glfw.KEY_UP:
            # rotate up
            rotate_camera(env=self.env, direction=[1., 0., 0.], angle=DELTA_ROT_KEY_PRESS, camera_id=self.camera_id)
        elif key == glfw.KEY_DOWN:
            # rotate down
            rotate_camera(env=self.env, direction=[-1., 0., 0.], angle=DELTA_ROT_KEY_PRESS, camera_id=self.camera_id)
        elif key == glfw.KEY_LEFT:
            # rotate left
            rotate_camera(env=self.env, direction=[0., 1., 0.], angle=DELTA_ROT_KEY_PRESS, camera_id=self.camera_id)
        elif key == glfw.KEY_RIGHT:
            # rotate right
            rotate_camera(env=self.env, direction=[0., -1., 0.], angle=DELTA_ROT_KEY_PRESS, camera_id=self.camera_id)
        elif key == glfw.KEY_PERIOD:
            # rotate counterclockwise
            rotate_camera(env=self.env, direction=[0., 0., 1.], angle=DELTA_ROT_KEY_PRESS, camera_id=self.camera_id)
        elif key == glfw.KEY_SLASH:
            # rotate clockwise
            rotate_camera(env=self.env, direction=[0., 0., -1.], angle=DELTA_ROT_KEY_PRESS, camera_id=self.camera_id)


    def on_release(self, window, key, scancode, action, mods):
        """
        Key handler for key releases.
        """
        pass

def print_command(char, info):
    char += " " * (10 - len(char))
    print("{}\t{}".format(char, info))


# TODO: Fix -- this breaks under new API (e.g.: Call "Lift" instead of "SawyerLift")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="SawyerLift",
    )
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
    inp = input("\nPlease paste a camera xml tag below (e.g. <camera ... />) \nOR leave blank for an example:\n")

    if len(inp) == 0:
        if args.env != "SawyerLift":
            raise Exception("ERROR: env must be SawyerLift to run default example.")
        print("\nUsing an example tag corresponding to the frontview camera.")
        print("This xml tag was copied from robosuite/models/assets/arenas/table_arena.xml")
        inp = '<camera mode="fixed" name="frontview" pos="1.6 0 1.45" quat="0.56 0.43 0.43 0.56"/>'
    
    print("NOTE: using the following xml tag:\n")
    print("{}\n".format(inp))

    # remember the tag and infer some properties
    cam_tree = ET.fromstring(inp)
    CAMERA_NAME = cam_tree.get("name")
    initial_file_camera_pos = np.array(cam_tree.get("pos").split(" ")).astype(float)
    initial_file_camera_quat = T.convert_quat(np.array(cam_tree.get("quat").split(" ")).astype(float), to='xyzw')
    initial_file_camera_pose = T.make_pose(initial_file_camera_pos, T.quat2mat(initial_file_camera_quat))

    # make the environment
    env = robosuite.make(
        args.env,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )
    env.reset()
    initial_mjstate = env.sim.get_state().flatten()
    xml = env.model.get_xml()

    # add mocap body to camera to be able to move it around
    xml = modify_xml_for_camera_movement(xml, camera_name=CAMERA_NAME)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(initial_mjstate)
    env.sim.forward()

    camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
    env.viewer.set_camera(camera_id=camera_id)

    # remember difference between camera pose in initial tag
    # and absolute camera pose in world
    initial_world_camera_pos = np.array(env.sim.data.get_mocap_pos("cameramover"))
    initial_world_camera_quat = T.convert_quat(env.sim.data.get_mocap_quat("cameramover"), to='xyzw')
    initial_world_camera_pose = T.make_pose(initial_world_camera_pos, T.quat2mat(initial_world_camera_quat))
    world_in_file = initial_file_camera_pose.dot(T.pose_inv(initial_world_camera_pose))

    # register callbacks to handle key presses in the viewer
    key_handler = KeyboardHandler(env=env, camera_id=camera_id)
    env.viewer.add_keypress_callback("any", key_handler.on_press)
    env.viewer.add_keyup_callback("any", key_handler.on_release)
    env.viewer.add_keyrepeat_callback("any", key_handler.on_press)

    # just spin to let user interact with glfw window
    spin_count = 0
    while True:
        action = np.zeros(env.dof)
        obs, reward, done, _ = env.step(action)
        env.render()
        spin_count += 1
        if spin_count % 500 == 0:
            # convert from world coordinates to file coordinates (xml subtree)
            camera_pos = env.sim.data.get_mocap_pos("cameramover")
            camera_quat = T.convert_quat(env.sim.data.get_mocap_quat("cameramover"), to='xyzw')
            world_camera_pose = T.make_pose(camera_pos, T.quat2mat(camera_quat))
            file_camera_pose = world_in_file.dot(world_camera_pose)
            camera_pos, camera_quat = T.mat2pose(file_camera_pose)
            camera_quat = T.convert_quat(camera_quat, to='wxyz')

            print("\n\ncurrent camera tag you should copy")
            cam_tree.set("pos", "{} {} {}".format(camera_pos[0], camera_pos[1], camera_pos[2]))
            cam_tree.set("quat", "{} {} {} {}".format(camera_quat[0], camera_quat[1], camera_quat[2], camera_quat[3]))
            print(ET.tostring(cam_tree, encoding="utf8").decode("utf8"))




