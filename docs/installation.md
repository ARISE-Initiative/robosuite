# Installation
robosuite officially supports Mac OS X and Linux on Python 3.5 or 3.7. It can be run with an on-screen display for visualization or in a headless mode for model training, with or without a GPU.

The base installation requires the MuJoCo physics engine (with [mujoco-py](https://github.com/openai/mujoco-py), refer to link for troubleshooting the installation and further instructions) and [numpy](http://www.numpy.org/). To avoid interfering with system packages, it is recommended to install it under a virtual environment by first running `virtualenv -p python3 . && source bin/activate`.

First download MuJoCo 2.0 ([Linux](https://www.roboti.us/download/mujoco200_linux.zip) and [Mac OS X](https://www.roboti.us/download/mujoco200_macos.zip)) and unzip its contents into `~/.mujoco/mujoco200`, and copy your MuJoCo license key `~/.mujoco/mjkey.txt`. You can obtain a license key from [here](https://www.roboti.us/license.html).
   - For Linux, you will need to install some packages to build `mujoco-py` (sourced from [here](https://github.com/openai/mujoco-py/blob/master/Dockerfile), with a couple missing packages added). If using `apt`, the required installation command is:
     ```sh
     $ sudo apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
             libosmesa6-dev software-properties-common net-tools unzip vim \
             virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf
     ```
     Note that for older versions of Ubuntu (e.g., 14.04) there's no libglfw3 package, in which case you need to `export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin` before proceeding to the next step.

### Install from pip
1. After setting up mujoco, robosuite can be installed with
```sh
    $ pip install robosuite
```

2. Test your installation with
```sh
    $ python -m robosuite.demo
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

3. (Optional) We also provide add-on functionalities, such as [OpenAI Gym](https://github.com/openai/gym) [interfaces](robosuite/wrappers/gym_wrapper.py), [inverse kinematics controllers](robosuite/wrappers/ik_wrapper.py) powered by [PyBullet](http://bulletphysics.org), and [teleoperation](robosuite/scripts/demo_spacemouse_ik_control.py) with [SpaceMouse](https://www.3dconnexion.com/products/spacemouse.html) devices (Mac OS X only). To enable these additional features, please install the extra dependencies by running
   ```sh
   $ pip3 install -r requirements-extra.txt
   ```

4. Test your installation with
```sh
    $ python robosuite/demo.py
```
