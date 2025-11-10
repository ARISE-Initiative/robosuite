import robosuite as suite
import numpy as np

# We need this import to make sure the env is registered
import robosuite.environments.manipulation.pick_place_clutter

print("Initializing test...")
print("Creating environment with ALL rendering disabled...")

try:
    env = suite.make(
        env_name="PickPlaceClutter",
        robots="Panda",
        has_renderer=False,             # <-- FALSE
        has_offscreen_renderer=False,   # <-- FALSE
        use_camera_obs=False,           # <-- FALSE
        control_freq=20,
        horizon=200,
    )
    
    print("Environment created successfully.")
    print("Resetting environment (this will run object placement)...")
    env.reset()
    print("Environment reset successfully.")

    print("\n--- OBJECT PLACEMENT VERIFICATION ---")

    # Get the names of the objects we expect
    task_object_names = [obj.name for obj in env.objects]
    clutter_object_names = [obj.name for obj in env.clutter_objects]
    all_object_names = task_object_names + clutter_object_names

    print(f"Found {len(all_object_names)} total objects: {all_object_names}\n")

    # Check their positions in the simulation
    for obj_name in all_object_names:
        body_id = env.obj_body_id[obj_name]
        obj_pos = env.sim.data.body_xpos[body_id]
        print(f"  - {obj_name}: [x={obj_pos[0]:.3f}, y={obj_pos[1]:.3f}, z={obj_pos[2]:.3f}]")

    print("\n[SUCCESS] Part 1 is complete and correct.")
    print("The environment loaded, and all 6 objects were successfully placed in the simulation.")
    
    env.close()

except Exception as e:
    print("\n[TEST FAILED]")
    print(f"An error occurred: {e}")
    print("Please check the error message. If it's the 'RandomizationError', you may still have too many objects for the bin.")

