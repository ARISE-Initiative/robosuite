
## Adding Third Party Controllers

To use a third-party controller with robosuite, you'll need to:
1. Create a new class that subclasses one of the composite controllers in `robosuite/controllers/composite/composite_controller.py`.
2. Register the composite controller with the decorator `@register_composite_controller`.
3. Implement composite specific functionality that ultimately provides control input to the underlying `part_controller`'s.
4. Import the new class so that it gets added to robosuite's `REGISTERED_COMPOSITE_CONTROLLERS_DICT` via the `@register_composite_controller` decorator.
5. Provide controller specific configs and the new controller's `type` in a json file.

For the new composite controllers subclassing `WholeBody`, you'll mainly need to update `joint_action_policy`.

We provide an example of how to use a third-party `WholeBodyMinkIK` composite controller with robosuite, in the `robosuite/examples/third_party_controller/` directory. You can run the command `python teleop_mink.py` example script to see a third-party controller in action. Note: to run this specific example, you'll need to `pip install mink`.


Steps 1 and 2:

In `robosuite/examples/third_party_controller/mink_controller.py`:

```
@register_composite_controller
class WholeBodyMinkIK(WholeBody):
    name = "WHOLE_BODY_MINK_IK"
```

Step 3:

In `robosuite/examples/third_party_controller/mink_controller.py`, add logic specific to the new composite controller:

```
self.joint_action_policy = IKSolverMink(...)
```

Step 4:

In `teleop_mink.py`, we import:

```
from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK
```

Step 5:

In `robosuite/examples/third_party_controller/default_mink_ik_gr1.json`, we add configs specific to our new composite controller. and also set the `type` to
match the `name` specified in `WholeBodyMinkIK`:

```
{
    "type": "WHOLE_BODY_MINK_IK",  # set the correct type
    "composite_controller_specific_configs": {
            ...
    },
    ...
}
```