from robosuite.controllers.interpolators.base_interpolator import Interpolator
import numpy as np
import robosuite.utils.transform_utils as T


class LinearInterpolator(Interpolator):
    """
    Simple class for implementing a linear interpolator.

    Abstracted to interpolate n-dimensions

    Args:
        max_delta: Maximum single change in dx allowed by the system.
                Note that this should be in units magnitude / step
        ndim: Number of dimensions to interpolate
        controller_freq: Frequency (Hz) of the controller
        policy_freq: Frequency (Hz) of the policy model
        ramp_ratio: Percentage of interpolation timesteps across which we will interpolate to a goal position.
            Note: Num total interpolation steps will be equal to np.floor(ramp_ratio * controller_freq / policy_freq)
                    i.e.: how many controller steps we get per action space update
        use_delta_goal: Whether to interpret inputs as delta goals from a current position or absolute values
        ori_interpolate: Whether this interpolator is interpolating angles (orientation) or not
    """
    def __init__(self,
                 max_delta,
                 ndim,
                 controller_freq,
                 policy_freq,
                 ramp_ratio=0.2,
                 use_delta_goal=False,
                 ori_interpolate=False,
                 ):

        self.max_delta = max_delta                                 # Maximum allowed change per interpolator step
        self.goal = None                                           # Requested goal
        self.start = None                                          # Start state
        self.dim = ndim                                            # Number of dimensions to interpolate
        self.order = 1                                             # Order of the interpolator (1 = linear)
        self.step = 0                                              # Current step of the interpolator
        self.total_steps = \
            np.ceil(ramp_ratio * controller_freq / policy_freq)    # Total num steps per interpolator action
        self.use_delta_goal = use_delta_goal                       # Whether to use delta or absolute goals (currently
                                                                   # not implemented yet- TODO)
        self.ori_interpolate = ori_interpolate                     # Whether this is interpolating orientation or not

    def set_goal(self, goal, start=None):
        """
        set_goal: Takes a requested goal and updates internal parameters for next interpolation step
        Args:
            goal: Requested goal (either absolute or delta value). Should be same dimension as self.dim
            start: Only relevant if "self.use_delta_goal" is set. This is the current state from which
                the goal will be taken relative to
        Returns:
            None
        """
        # First, check to make sure requested goal shape is the same as self.dim
        if goal.shape[0] != self.dim:
            print("Requested goal: {}".format(goal))
            raise ValueError("LinearInterpolator: Input size wrong for goal; got {}, needs to be {}!".format(
                goal.shape[0], self.dim))

        # Update goal and save start state
        self.goal = np.array(goal)
        self.start = start

        # Reset interpolation steps
        self.step = 0

    def get_interpolated_goal(self, x):
        """
        get_interpolated_goal: Takes the current state and provides the next step in interpolation given
            the remaining steps.

        NOTE: If this interpolator is for orientation, it is assumed to be receiving

        Args:
            x: Current state. Should be same dimension as self.dim
            NOTE: If this interpolator is for orientation, x is assumed to be the current relative rotation from
                the initial goal that was set. Otherwise, it is assumed to be an absolute value
        Returns:
            x_current: Next position in the interpolated trajectory
        """
        # First, check to make sure x in same shape as self.dim
        if x.shape[0] != self.dim:
            print("Current position: {}".format(x))
            raise ValueError("LinearInterpolator: Input size wrong; needs to be {}!".format(self.dim))

        # Also make sure goal has been set
        if self.goal is None:
            raise ValueError("LinearInterpolator: Goal has not been set yet!")

        # Calculate the desired next step based on remaining interpolation steps
        if self.ori_interpolate:
            # This is an orientation interpolation, so we interpolate linearly around a sphere instead
            goal = self.goal
            if self.dim == 3:
                # this is assumed to be euler (x,y,z), so we need to first map to quat
                x = T.mat2quat(T.euler2mat(x))
                goal = T.mat2quat(T.euler2mat(self.goal))

            # Interpolate to the next sequence
            x_current = T.quat_slerp(x, goal,
                                     fraction=(self.step + 1) / self.total_steps)
            # Check if dx is greater than max value; if it is; clamp and notify user
            dx, clipped = T.clip_rotation(T.quat_distance(x_current, x), self.max_delta)
            if clipped:
                print(
                    "LinearInterpolator: WARNING: Requested interpolation (ori) exceeds max speed;"
                    "clamping to {}.".format(dx))
            # Map back to euler if necessary
            x_current = dx
            if self.dim == 3:
                x_current = T.mat2euler(T.quat2mat(x_current))
        else:
            # This is a normal interpolation
            dx = (self.goal - x) / (self.total_steps - self.step)
            # Check if dx is greater than max value; if it is; clamp and notify user
            dx, clipped = T.clip_translation(dx, self.max_delta)
            if clipped:
                print("LinearInterpolator: WARNING: Requested interpolation "
                      "exceeds max speed; clamping to {}.".format(dx))
            x_current = x + dx

        # Increment step if there's still steps remaining based on ramp ratio
        if self.step < self.total_steps - 1:
            self.step += 1

        # Return the new interpolated step
        return x_current
