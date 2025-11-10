
#!/usr/bin/env python3
"""
mimicgen_seed_hdf5_recorder.py  (policy-enabled)

Adds:
- --policy "module:function"  OR  --policy-file /path/to/py  --policy-func function_name
- --env-list  (prints available robosuite envs and exits)
- base_abs_pose.json auto-resolved at ./controllers/basic_abs_pose.json (relative to this file)

Everything else remains: writes MimicGen "core" HDF5.
"""

import argparse, os, json, glob, re, warnings, importlib, importlib.util, inspect
from pathlib import Path
import numpy as np
import h5py

# Optional: only needed if you want to record new data instead of convert-only
try:
    import robosuite as suite
    from robosuite.wrappers import DataCollectionWrapper
    HAVE_ROBOSUITE = True
except Exception:
    HAVE_ROBOSUITE = False


def quat_to_omega(q_prev_xyzw, q_curr_xyzw, dt):
    def to_wxyz(q):
        x,y,z,w = q
        return np.array([w,x,y,z], dtype=np.float64)

    qp = to_wxyz(q_prev_xyzw)
    qc = to_wxyz(q_curr_xyzw)
    r = np.array([
        qc[0]*qp[0] + qc[1]*(-qp[1]) + qc[2]*(-qp[2]) + qc[3]*(-qp[3]),
        qc[0]*(-qp[1]) + qc[1]*qp[0] + qc[2]*(-qp[3]) + qc[3]*qp[2],
        qc[0]*(-qp[2]) + qc[1]*qp[3] + qc[2]*qp[0] + qc[3]*(-qp[1]),
        qc[0]*(-qp[3]) + qc[1]*(-qp[2]) + qc[2]*qp[1] + qc[3]*qp[0],
    ], dtype=np.float64)
    angle = 2.0 * np.arccos(np.clip(r[0], -1.0, 1.0))
    s = np.sqrt(max(1e-12, 1.0 - r[0]*r[0]))
    axis = r[1:] / s if s > 1e-6 else np.array([0.0,0.0,0.0], dtype=np.float64)
    return (axis * angle) / max(1e-12, dt)


def stack_key(obs_list, key):
    return np.stack([o[key] for o in obs_list], axis=0)


def resize_if_needed(arr, H, W):
    if arr.shape[1] == H and arr.shape[2] == W:
        return arr
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python is required to resize images. Install with `pip install opencv-python`.")
    return np.stack([cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in arr], axis=0)


def load_episode_npzs(ep_dir):
    files = sorted(glob.glob(str(Path(ep_dir) / "state_*.npz")))
    if not files:
        raise FileNotFoundError(f"No 'state_*.npz' files found in {ep_dir}")
    obs_list, actions_list, xml_model_str = [], [], ""
    for fp in files:
        dic = np.load(fp, allow_pickle=True)
        act = dic.get("actions", dic.get("action"))
        if act is None:
            raise KeyError(f"No 'actions' key in {fp}. Keys: {list(dic.keys())}")
        actions_list.append(np.asarray(act, dtype=np.float32))
        if "obs" not in dic:
            raise KeyError(f"No 'obs' key in {fp}. Keys: {list(dic.keys())}")
        obs = dic["obs"].item() if hasattr(dic["obs"], "item") else dic["obs"]
        obs_list.append(obs)
        if "model_file" in dic:
            try:
                xml_model_str = str(dic["model_file"].item())
            except Exception:
                pass
    return obs_list, np.stack(actions_list, axis=0), xml_model_str


def episode_to_mg_core(h5_group, obs_list, actions, env_name, cameras, H, W, control_freq, xml_model_str):
    T = actions.shape[0]
    dt = 1.0 / float(control_freq)

    required = [
        "object-state",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "robot0_joint_pos",
        "robot0_joint_pos_cos",
        "robot0_joint_pos_sin",
        "robot0_joint_vel",
        "agentview_image",
    ]
    for k in required:
        if k not in obs_list[0]:
            raise KeyError(f"Missing observation key '{k}' in episode. Present keys: {sorted(obs_list[0].keys())}")

    object_vec     = stack_key(obs_list, "object-state").astype(np.float32)
    eef_pos        = stack_key(obs_list, "robot0_eef_pos").astype(np.float32)
    eef_quat       = stack_key(obs_list, "robot0_eef_quat").astype(np.float32)
    grip_qpos      = stack_key(obs_list, "robot0_gripper_qpos").astype(np.float32)
    grip_qvel      = stack_key(obs_list, "robot0_gripper_qvel").astype(np.float32)
    joint_pos      = stack_key(obs_list, "robot0_joint_pos").astype(np.float32)
    joint_pos_cos  = stack_key(obs_list, "robot0_joint_pos_cos").astype(np.float32)
    joint_pos_sin  = stack_key(obs_list, "robot0_joint_pos_sin").astype(np.float32)
    joint_vel      = stack_key(obs_list, "robot0_joint_vel").astype(np.float32)

    agent_imgs     = stack_key(obs_list, "agentview_image").astype(np.uint8)
    agent_imgs     = resize_if_needed(agent_imgs, H, W)

    if "robot0_eye_in_hand_image" in obs_list[0]:
        eye_imgs = stack_key(obs_list, "robot0_eye_in_hand_image").astype(np.uint8)
        eye_imgs = resize_if_needed(eye_imgs, H, W)
    else:
        warnings.warn("robot0_eye_in_hand_image is missing; duplicating agentview_image as a fallback.")
        eye_imgs = agent_imgs.copy()

    eef_vel_lin = np.zeros_like(eef_pos, dtype=np.float32)
    eef_vel_lin[1:] = (eef_pos[1:] - eef_pos[:-1]) / max(1e-12, dt)

    eef_vel_ang = np.zeros_like(eef_pos, dtype=np.float32)
    for t in range(1, T):
        eef_vel_ang[t] = quat_to_omega(eef_quat[t-1], eef_quat[t], dt).astype(np.float32)

    states_71 = np.concatenate([
        object_vec,
        eef_pos,
        eef_quat,
        eef_vel_ang,
        eef_vel_lin,
        grip_qpos,
    ], axis=1).astype(np.float32)

    h5_group.attrs["model_file"] = xml_model_str
    h5_group.attrs["num_samples"] = int(T)

    h5_group.create_dataset("actions", data=actions, compression="gzip")
    h5_group.create_dataset("states", data=states_71, compression="gzip")
    h5_group.create_dataset("dones", data=np.array([False]*(T-1) + [True], dtype=np.bool_), compression="gzip")
    h5_group.create_dataset("rewards", data=np.zeros((T,), dtype=np.float32), compression="gzip")

    og = h5_group.create_group("obs")
    og.create_dataset("agentview_image", data=agent_imgs, compression="gzip")
    og.create_dataset("robot0_eye_in_hand_image", data=eye_imgs, compression="gzip")
    og.create_dataset("object", data=object_vec, compression="gzip")
    og.create_dataset("robot0_eef_pos", data=eef_pos, compression="gzip")
    og.create_dataset("robot0_eef_quat", data=eef_quat, compression="gzip")
    og.create_dataset("robot0_eef_vel_ang", data=eef_vel_ang, compression="gzip")
    og.create_dataset("robot0_eef_vel_lin", data=eef_vel_lin, compression="gzip")
    og.create_dataset("robot0_gripper_qpos", data=grip_qpos, compression="gzip")
    og.create_dataset("robot0_gripper_qvel", data=grip_qvel, compression="gzip")
    og.create_dataset("robot0_joint_pos", data=joint_pos, compression="gzip")
    og.create_dataset("robot0_joint_pos_cos", data=joint_pos_cos, compression="gzip")
    og.create_dataset("robot0_joint_pos_sin", data=joint_pos_sin, compression="gzip")
    og.create_dataset("robot0_joint_vel", data=joint_vel, compression="gzip")


def convert_episodes_root_to_hdf5(episodes_root, out_h5, env_name, control_freq, cameras, H, W):
    root = Path(episodes_root)
    ep_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("ep_")])
    if not ep_dirs:
        raise FileNotFoundError(f"No ep_* subdirs found under {episodes_root}")

    with h5py.File(out_h5, "w") as f:
        g = f.create_group("data")
        total = 0
        for ep in ep_dirs:
            try:
                obs_list, actions, xml_model_str = load_episode_npzs(ep)
            except Exception as e:
                warnings.warn(f"Skipping {ep.name}: {e}")
                continue

            demo = g.create_group(f"demo_{total}")
            episode_to_mg_core(demo, obs_list, actions, env_name, cameras, H, W, control_freq, xml_model_str)
            total += 1

        g.attrs["env_args"] = json.dumps({
            "env_name": env_name,
            "camera_names": cameras,
            "camera_height": H,
            "camera_width": W,
            "control_freq": control_freq,
        })
        g.attrs["total"] = int(total)

    print(f"Wrote {out_h5} with {total} demos.")


def import_from_module_function(spec: str):
    """
    spec = "package.module:function"
    returns callable
    """
    mod_name, func_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def import_from_file(path: str, func: str):
    spec = importlib.util.spec_from_file_location("user_policy_module", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    fn = getattr(mod, func)
    if not callable(fn):
        raise TypeError(f"{func} in {path} is not callable")
    return fn


def resolve_base_pose_path():
    # ./controllers/basic_abs_pose.json relative to this file
    here = Path(__file__).resolve().parent
    candidate = here / "controllers" / "basic_abs_pose.json"
    return str(candidate)


def record_with_env(env_name, num_episodes, out_h5, control_freq, policy_fn=None):
    if not HAVE_ROBOSUITE:
        raise RuntimeError("robosuite not installed. Install robosuite or use --episodes_root to convert existing data.")

    # List envs if unknown
    if env_name not in getattr(suite, "ALL_ENVIRONMENTS", []):
        warnings.warn(f"--env '{env_name}' not found in robosuite.ALL_ENVIRONMENTS. "
                      f"Use --env-list to print available names.")

    env = suite.make(
        env_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=84,
        camera_widths=84,
        control_freq=control_freq,
    )

    tmp_root = Path(f"/tmp/mg_seed_{os.getpid()}")
    tmp_root.mkdir(parents=True, exist_ok=True)
    env = DataCollectionWrapper(env, str(tmp_root), record_video=False)

    # Prepare a context dict we pass to the policy (paths etc.)
    ctx = {
        "base_abs_pose_json": resolve_base_pose_path(),
        "env_name": env_name,
        "control_freq": control_freq,
    }

    for ep in range(num_episodes):
        obs = env.reset()
        steps = 0
        done = False

        # If your policy returns a fixed action sequence, it can also ignore obs and step.
        while not done and steps < int(control_freq * 120):  # up to ~120s
            if policy_fn is None:
                action = np.zeros(7, dtype=np.float32)
            else:
                # Call with a flexible signature.
                # Try (obs, step, ctx=...), then (obs, step), then (obs).
                try:
                    action = policy_fn(obs, steps, ctx=ctx)
                except TypeError:
                    try:
                        action = policy_fn(obs, steps)
                    except TypeError:
                        action = policy_fn(obs)
            action = np.asarray(action, dtype=np.float32).reshape(7)
            obs, reward, done, info = env.step(action)
            steps += 1

        env.reset_to(reset_states=None, save_success=True)
        print(f"Recorded episode {ep+1}/{num_episodes} to {tmp_root}")

    convert_episodes_root_to_hdf5(tmp_root, out_h5, env_name, control_freq,
                                  cameras=["agentview", "robot0_eye_in_hand"], H=84, W=84)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default=None, help="robosuite env name, e.g., PickPlace, Lift, NutAssembly, etc.")
    p.add_argument("--env-list", action="store_true", help="List available robosuite environments and exit")
    p.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    p.add_argument("--num_episodes", type=int, default=1, help="Num episodes to record (ignored if --episodes_root is set)")
    p.add_argument("--episodes_root", type=str, default="", help="If provided, convert ep_* dirs here into an HDF5 (no env needed)")
    p.add_argument("--control_freq", type=float, default=20.0, help="Control frequency (Hz) used for finite-diff velocities")
    # policy hooks
    p.add_argument("--policy", type=str, default="", help='Call policy as "module.submodule:funcname"')
    p.add_argument("--policy-file", type=str, default="", help="Path to a .py file containing the policy function")
    p.add_argument("--policy-func", type=str, default="", help="Function name inside --policy-file")
    args = p.parse_args()

    if args.env_list:
        if not HAVE_ROBOSUITE:
            print("robosuite not installed.")
            return
        print("Available robosuite environments:")
        try:
            print("\n".join(sorted(getattr(suite, "ALL_ENVIRONMENTS", []))))
        except Exception as e:
            print("Could not list environments:", e)
        return

    out_h5 = Path(args.output).expanduser().resolve()
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    # Resolve policy function if provided
    policy_fn = None
    if args.policy:
        policy_fn = import_from_module_function(args.policy)
    elif args.policy_file and args.policy_func:
        policy_fn = import_from_file(args.policy_file, args.policy_func)

    if args.episodes_root:
        # Convert-only mode
        if args.env is None:
            # default env name just for metadata
            args.env = "PickPlace"
        convert_episodes_root_to_hdf5(args.episodes_root, out_h5, args.env, args.control_freq,
                                      cameras=["agentview", "robot0_eye_in_hand"], H=84, W=84)
    else:
        if args.env is None:
            raise SystemExit("--env is required when recording from an environment (no --episodes_root given). "
                             "Run with --env-list to see valid options.")
        record_with_env(args.env, args.num_episodes, out_h5, args.control_freq, policy_fn=policy_fn)


if __name__ == "__main__":
    main()
