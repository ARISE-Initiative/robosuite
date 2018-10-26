"""Loads a raw mjcf file and saves a compiled mjcf file.

This avoids mujoco-py from complaining about .urdf extension.
Also allows assets to be compiled properly.

Example:
    $ python compile_mjcf_model.py source_mjcf.xml target_mjcf.xml
"""

import os
import sys
from shutil import copyfile
from mujoco_py import load_model_from_path


def print_usage():
    print("""python compile.py input_file output_file""")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print_usage()
        exit(0)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    input_folder = os.path.dirname(input_file)

    tempfile = os.path.join(input_folder, ".surreal_temp_model.xml")
    copyfile(input_file, tempfile)

    model = load_model_from_path(tempfile)
    xml_string = model.get_xml()
    with open(output_file, "w") as f:
        f.write(xml_string)

    os.remove(tempfile)
