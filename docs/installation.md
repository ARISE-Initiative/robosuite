# Installation
**robosuite** officially supports Mac OS X and Linux on Python 3. It can be run with an on-screen display for visualization or in a headless mode for model training, with or without a GPU.

The base installation requires the MuJoCo physics engine (with [mujoco-py](https://github.com/nimrod-gileadi/mujoco-py), refer to link for troubleshooting the installation and further instructions) and [numpy](http://www.numpy.org/). To avoid interfering with system packages, it is recommended to install it under a virtual environment by first running `virtualenv -p python3 . && source bin/activate`.

For Linux, you will need to install some packages to build `mujoco-py` (sourced from [here](https://github.com/openai/mujoco-py/blob/master/Dockerfile), with a couple missing packages added). If using `apt`, the required installation command is:
   ```sh
   $ sudo apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
            libosmesa6-dev software-properties-common net-tools unzip vim \
            virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf
   ```
   Note that for older versions of Ubuntu (e.g., 14.04) there's no libglfw3 package, in which case you need to `export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin` before proceeding to the next step.

### Install from pip
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
$ git clone https://github.com/StanfordVL/robosuite.git
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