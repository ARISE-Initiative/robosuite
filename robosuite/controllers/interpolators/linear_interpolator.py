from robosuite.controllers.interpolators.base_interpolator import Interpolator
import numpy as np


class LinearInterpolator(Interpolator):
    '''
    Simple class for implementing a linear interpolator.

    Abstracted to interpolate n-dimensions

    Args:
        max_dx: Maximum single change in dx allowed by the system.
                Note that this should be in units distance / second
        ndim: Number of dimensions to interpolate
        controller_freq: Frequency (Hz) of the controller
        policy_freq: Frequency (Hz) of the policy model
        ramp_ratio: Percentage of interpolation timesteps across which we will interpolate to a goal position.
            Note: Num total interpolation steps will be equal to np.floor(ramp_ratio * controller_freq / policy_freq)
                    i.e.: how many controller steps we get per action space update
    '''
    def __init__(self, max_dx, ndim, controller_freq, policy_freq, ramp_ratio=0.2):
        self.max_dx = max_dx                                       # Maximum allowed change per interpolator step
        self.goal = None                                           # Requested goal
        self.dim = ndim                                            # Number of dimensions to interpolate
        self.order = 1                                             # Order of the interpolator (1 = linear)
        self.step = 0                                              # Current step of the interpolator
        self.total_steps = \
            np.floor(ramp_ratio * controller_freq / policy_freq)   # Total num steps per interpolator action

    '''
    set_goal: Takes a requested goal and updates internal parameters for next interpolation step
    Args:
        goal: Requested goal. Should be same dimension as self.dim
    Returns:
        None
    '''
    def set_goal(self, goal):
        # First, check to make sure requested goal shape is the same as self.dim
        if goal.shape[0] != self.dim:
            print("Requested goal: {}".format(goal))
            raise ValueError("LinearInterpolator: Input size wrong for goal; got {}, needs to be {}!".format(
                goal.shape[0], self.dim))

        # Update goal and reset interpolation step
        self.goal = np.array(goal)
        self.step = 0



    '''
    get_interpolated_goal: Takes the current position and provides the next step in interpolation given
        the remaining steps
    Args:
        x: Current position. Should be same dimension as self.dim
    Returns:
        x_current: Next position in the interpolated trajectory
    '''
    def get_interpolated_goal(self, x):
        # First, check to make sure x in same shape as self.dim
        if x.shape[0] != self.dim:
            print("Current position: {}".format(x))
            raise ValueError("LinearInterpolator: Input size wrong for pos; needs to be {}!".format(self.dim))

        # Also make sure goal has been set
        if self.goal is None:
            raise ValueError("LinearInterpolator: Goal has not been set yet!")

        # Calculate the desired next step based on remaining interpolation steps and increment step if necessary
        if self.step < self.total_steps:
            dx = (self.goal - x) / (self.total_steps - self.step)
            # Check if dx is greater than max value; if it is; clamp and notify user
            if np.any(abs(dx) > self.max_dx):
                dx = np.clip(dx, -self.max_dx, self.max_dx)
                print("LinearInterpolator: WARNING: Requested interpolation exceeds max speed; clamping to {}.".format(self.max_dx))
            x_current = x + dx
            self.step += 1
        else:
            # We've already completed all interpolation steps; return goal
            x_current = self.goal

        # Return the new interpolated step
        return x_current










