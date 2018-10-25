from setuptools import setup, find_packages


setup(
    name="robosuite",
    packages=[
        package for package in find_packages() if package.startswith("robosuite")
    ],
    install_requires=[
        "numpy>=1.13.3",
        "mujoco-py<1.50.2,>=1.50.1",
    ],
    description="Surreal Robotics Suite: Standardized and Accessible Robot Manipulation Benchmark in Physics Simulation",
    author="Yuke Zhu, Jiren Zhu, Ajay Mandlekar, Joan Creus-Costa, Anchit Gupta",
    url="https://github.com/StanfordVL/robosuite",
    author_email="yukez@cs.stanford.edu",
    version="0.1.0",
)
