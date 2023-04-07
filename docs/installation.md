# Installation
**robosuite** officially supports macOS and Linux on Python 3. It can be run with an on-screen display for visualization or in a headless mode for model training, with or without a GPU.

The base installation requires the MuJoCo physics engine (with [mujoco](https://github.com/deepmind/mujoco), refer to link for troubleshooting the installation and further instructions) and [numpy](http://www.numpy.org/). To avoid interfering with system packages, it is recommended to install it under a virtual environment by first running `virtualenv -p python3 . && source bin/activate` or setting up a Conda environment by installing [Anaconda](https://www.anaconda.com/) and running `conda create -n robosuite python=3.8`.

### Install from pip

**Note**: for users looking to use the most up-to-date code and develop advanced features, it is recommended to install from source.

1. After setting up mujoco, robosuite can be installed with
   ```sh
   $ pip install robosuite
   ```

2. Test your installation with
   ```sh
   $ python -m robosuite.demos.demo_random_action
   ```

### Install from source

1. Clone the robosuite repository
   ```sh 
   $ git clone https://github.com/ARISE-Initiative/robosuite.git
   $ cd robosuite
   ```

2. Install the base requirements with
   ```sh
   $ pip3 install -r requirements.txt
   ```
   This will also install our library as an editable package, such that local changes will be reflected elsewhere without having to reinstall the package.

3. (Optional) We also provide add-on functionalities, such as [OpenAI Gym](https://github.com/openai/gym) [interfaces](source/robosuite.wrappers), [inverse kinematics controllers](source/robosuite.controllers) powered by [PyBullet](http://bulletphysics.org), and [teleoperation](source/robosuite.devices) with [SpaceMouse](https://www.3dconnexion.com/products/spacemouse.html) devices. To enable these additional features, please install the extra dependencies by running
   ```sh
   $ pip3 install -r requirements-extra.txt
   ```

4. Test your installation with
   ```sh
   $ python robosuite/demos/demo_random_action.py
   ```
