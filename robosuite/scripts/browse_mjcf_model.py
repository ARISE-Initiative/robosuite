"""Visualize MJCF models.

Loads MJCF XML models from file and renders it on screen.

Example:
    $ python browse_mjcf_model.py --filepath ../models/assets/arenas/table_arena.xml
"""

import argparse
import os

import mujoco

import robosuite as suite
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContext, MjSim

if __name__ == "__main__":

    arena_file = os.path.join(suite.models.assets_root, "arenas/pegs_arena.xml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=arena_file)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.filepath)
    sim = MjSim(model)
    render_context = MjRenderContext(sim)
    sim.add_render_context(render_context)
    viewer = OpenCVRenderer(sim)

    print("Press ESC to exit...")
    while True:
        sim.step()
        viewer.render()
