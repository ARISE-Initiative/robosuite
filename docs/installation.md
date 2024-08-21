# Installation
**robosuite** officially supports macOS and Linux on Python 3. It can be run with an on-screen display for visualization or in a headless mode for model training, with or without a GPU.

The base installation requires the MuJoCo physics engine (with [mujoco](https://github.com/deepmind/mujoco), refer to link for troubleshooting the installation and further instructions) and [numpy](http://www.numpy.org/). To avoid interfering with system packages, it is recommended to install it under a virtual environment by first running `virtualenv -p python3 . && source bin/activate` or setting up a Conda environment by installing [Anaconda](https://www.anaconda.com/) and running `conda create -n robosuite python=3.8`.

### Install from pip

**Note**: For users looking to use the most up-to-date code and develop customized features, it is recommended to install from source.

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

### Installing on Windows

It is common to run into issues when installing **robosuite** on a Windows machine. **robosuite** can be installed on Windows using the following steps.

1. Either follow step 1 from the section [Install from pip](#install-from-pip) or steps 1 and 2 in the section [Install from source](#install-from-source). During this process, you may run into some errors. Please refer to the steps below on how to fix these.

2. If you run into the error `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\tmp\\robosuite.log'`, create a directory called `tmp` under `C:\`.

3. You will also likely face the issue of `mujoco.dll not found`. If you are running in a conda environment (highly recommended), go to the location where your packages are installed (i.e. site-packages). If you are unsure where the MuJoCo package is located, open a new python shell and run the following.

   ```python
   import mujoco
   print(mujoco.__path__)
   ```

   If the MuJoCo package does not already exists, install it by running 

   ```sh
   $ pip install mujoco
   ```

   Within the MuJoCo package, there should be a file called `mujoco.dll`. If you installed robosuite using pip, copy and paste this file into `anaconda3\envs\{your env name}\Lib\site-packages\robosuite\utils `. If you installed robosuite from source, copy and paste this file directly into `robosuite\utils`. 

4. You may also get an `EGL` issue. If this happens, please go into `robosuite\utils\binding_utils.py` (either in site-packages or in cloned repository depending on whether you installed from pip or source) and change `"egl"` to `"wgl"` at line 43. It should look like this:

   ```python
    if _SYSTEM == "Darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    else:
        os.environ["MUJOCO_GL"] = "wgl"
   ```

5. Test your **robosuite** installation by running

   ```sh
   $ python robosuite/demos/demo_random_action.py
   ```