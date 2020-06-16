"""
Subclasses for robosuite environments to define tasks that are compatible with teleoperation.
This is a convenient place to model different tasks we'd like to collect data on.

NOTE: most of these just override the intiial robot configuration to make it convenient for
teleoperation.
"""
import numpy as np

import robosuite
from robosuite.environments.sawyer_lift import *
from robosuite.environments.sawyer_pick_place import *
from robosuite.environments.sawyer_nut_assembly import *
from robosuite.models.tasks import UniformRandomSampler

DEFAULT_JPOS = np.array([0.00, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

### Lift Tasks ###

class SawyerLiftTeleop(SawyerLift):
    def _get_default_placement_initializer(self):
        # to maintain backwards compatibility, use a constant z-rotation for the cube
        self.placement_initializer = UniformRandomSampler(
            x_range=[-0.03, 0.03],
            y_range=[-0.03, 0.03],
            ensure_object_boundary_in_range=False,
            z_rotation=1.,
        )

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = (-0.03, 0.03, 3)
        y_bounds = (-0.03, 0.03, 3)
        z_rot_bounds = (1., 1., 1) # constant z-rotation
        return x_bounds, y_bounds, z_rot_bounds

    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

### Nut Assembly Tasks ###

class SawyerNutAssemblyTeleop(SawyerNutAssembly):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerNutAssemblySingleTeleop(SawyerNutAssemblySingle):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerNutAssemblySquareTeleop(SawyerNutAssemblySquare):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerNutAssemblyRoundTeleop(SawyerNutAssemblyRound):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

### Pick Place Tasks ###

class SawyerPickPlaceTeleop(SawyerPickPlace):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPickPlaceSingleTeleop(SawyerPickPlaceSingle):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPickPlaceMilkTeleop(SawyerPickPlaceMilk):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPickPlaceBreadTeleop(SawyerPickPlaceBread):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPickPlaceCerealTeleop(SawyerPickPlaceCereal):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPickPlaceCanTeleop(SawyerPickPlaceCan):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS
