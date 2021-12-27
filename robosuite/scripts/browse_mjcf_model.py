"""Visualize MJCF models.

Loads MJCF XML models from file and renders it on screen.

Example:
    $ python browse_arena_model.py --filepath ../models/assets/arenas/table_arena.xml
"""

import argparse
import os

from mujoco_py import MjSim, MjViewer, load_model_from_path

import robosuite as suite

if __name__ == "__main__":

    arena_file = os.path.join(suite.models.assets_root, "arenas/pegs_arena.xml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=arena_file)
    args = parser.parse_args()

    model = load_model_from_path(args.filepath)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    print("Press ESC to exit...")
    while True:
        sim.step()
        viewer.render()
