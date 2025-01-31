# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="robosuite",
    packages=[package for package in find_packages() if package.startswith("robosuite")],
    install_requires=[
        "numpy>=1.13.3",
        "numba>=0.49.1",
        "scipy>=1.2.3",
        "mujoco>=3.2.3",
        "mink>=0.0.5",
        "Pillow",
        "opencv-python",
        "pynput",
        "termcolor",
        "pytest",
        "tqdm",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="robosuite: A Modular Simulation Framework and Benchmark for Robot Learning",
    author="Yuke Zhu",
    url="https://github.com/ARISE-Initiative/robosuite",
    author_email="yukez@cs.utexas.edu",
    version="1.5.2",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
