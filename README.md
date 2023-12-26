# robosuite for particle jamming gripper

#### Installations

1. Clone the repository

```
git clone https://github.com/DaebangStn/robosuite.git
```

2. There is an error when repository name and package name are the same.

```
mv ./robosuite ./pkg
```

3. Install packages 

(setuptools has changed the way it handles editable installs
so that you should use `--config-settings editable_mode=compat` to install)

```
pip install -e . --config-settings editable_mode=compat
pip install -r requirements-extra.txt
```

#### Adding novel mechanism gripper

1. Add gripper class in `robosuite/models/grippers/xxx.py` directory (inherit from `FlexGripperModel` class)
2. Append gripper name in `robosuite/models/grippers/__init__.py` file
3. Add gripper mjcf model in `robosuite/models/assets/grippers/xxx.xml` directory
4. (Optional) Append new control signal in `robosuite/devices/keyboards.py` file
    - _display_controls()
    - _reset_internal_state()
    - get_controller_state()
    - on_press() or on_release()


#### TODO
1. Add Deformable objects (maybe plugin..?)
   