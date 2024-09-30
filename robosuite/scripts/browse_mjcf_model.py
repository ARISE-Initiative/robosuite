"""Visualize MJCF models.

Loads MJCF XML models from file and renders it on screen.

Example:
    $ python browse_mjcf_model.py --filepath ../models/assets/arenas/table_arena.xml
"""

import argparse
import os

import mujoco
from mujoco import viewer
from pynput import keyboard

import robosuite as suite
from robosuite.utils.binding_utils import MjSim

DEFAULT_FREE_CAM = {
    "lookat": [0, 0, 0.7],
    "distance": 1.5,
    "azimuth": 180,
    "elevation": -20,
}

continue_running = True


def on_press(key):
    global continue_running
    try:
        if key == keyboard.Key.esc:
            continue_running = False
    except AttributeError:
        pass


if __name__ == "__main__":

    arena_file = os.path.join(suite.models.assets_root, "arenas/pegs_arena.xml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=arena_file)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.filepath)
    sim = MjSim(model)

    vwr = viewer.launch_passive(sim.model._model, sim.data._data, show_left_ui=False, show_right_ui=False)
    vwr.cam.lookat = DEFAULT_FREE_CAM["lookat"]
    vwr.cam.distance = DEFAULT_FREE_CAM["distance"]
    vwr.cam.azimuth = DEFAULT_FREE_CAM["azimuth"]
    vwr.cam.elevation = DEFAULT_FREE_CAM["elevation"]

    vwr.sync()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("Press 'ESC' to exit.")

    while continue_running:
        pass

    vwr.close()
    listener.stop()
    listener.join()
