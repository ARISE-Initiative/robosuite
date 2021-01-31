from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if '.png' not in x]
long_description = ''.join(lines)

setup(
    name="robosuite",
    packages=[
        package for package in find_packages() if package.startswith("robosuite")
    ],
    install_requires=[
        "numpy>=1.13.3",
        "numba>=0.49.1",
        "scipy>=1.2.3",
        "mujoco-py==2.0.2.9",
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description="robosuite: A Modular Simulation Framework and Benchmark for Robot Learning",
    author="Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín",
    url="https://github.com/ARISE-Initiative/robosuite",
    author_email="yukez@cs.utexas.edu",
    version="1.2.0",
    long_description=long_description,
    long_description_content_type='text/markdown'
)
