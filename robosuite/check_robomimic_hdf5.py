#!/usr/bin/env python3
"""
check_robomimic_hdf5.py

Quick validator for demonstration datasets intended for robomimic / MimicGen.
It inspects structure, attributes, shapes, and common pitfalls.

Usage:
  python check_robomimic_hdf5.py /path/to/dataset.hdf5 [--strict]

Exit codes:
  0 = passed basic checks
  1 = warnings only (non-fatal issues)
  2 = errors found (structural or consistency problems)
"""
import argparse
import json
import sys
from typing import Dict, List, Tuple, Any

import h5py
from collections import defaultdict


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def list_demos(data_grp) -> List[str]:
    return sorted([k for k in data_grp.keys() if k.startswith("demo_")])


def is_dataset(obj) -> bool:
    return isinstance(obj, h5py.Dataset)


def is_group(obj) -> bool:
    return isinstance(obj, h5py.Group)


def try_load_env_args(attrs: h5py.AttributeManager) -> Tuple[bool, Any]:
    if "env_args" not in attrs:
        return False, None
    val = attrs["env_args"]
    # env_args is often stored as JSON (bytes or str)
    if isinstance(val, (bytes, bytearray)):
        try:
            return True, json.loads(val.decode("utf-8"))
        except Exception:
            return True, val.decode("utf-8", errors="ignore")
    if isinstance(val, str):
        try:
            return True, json.loads(val)
        except Exception:
            return True, val
    # Sometimes it may be a pickled dict or other type; just return raw
    return True, val


def check_demo_group(demo: h5py.Group) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Returns (warnings, errors, info_dict)
    info_dict contains keys: T, has_obs, has_next_obs, has_states, has_actions, obs_keys
    """
    warnings, errors = [], []
    info = {"T": None, "has_obs": False, "has_next_obs": False, "has_states": False,
            "has_actions": False, "obs_keys": [], "shapes": {}}

    # Basic children commonly expected:
    # - actions [T, A]
    # - rewards [T] (optional for some pipelines)
    # - dones or terminals [T] (naming may vary)
    # - (optional) states [T, ...] if states-only dataset
    # - obs/<keys> [T, ...]; next_obs/<keys> [T, ...]
    children = set(demo.keys())
    actions = demo.get("actions", None)
    rewards = demo.get("rewards", None)
    dones = demo.get("dones", demo.get("terminals", None))  # accommodate 'terminals'
    states = demo.get("states", None)
    obs = demo.get("obs", None)
    next_obs = demo.get("next_obs", None)

    if actions is None:
        warnings.append("Missing 'actions' dataset (some tools require it).")
    else:
        info["has_actions"] = True
        if not is_dataset(actions) or actions.ndim < 1:
            errors.append("'actions' should be a dataset with shape [T, ...].")
        else:
            info["T"] = int(actions.shape[0])
            info["shapes"]["actions"] = tuple(actions.shape)

    if rewards is not None:
        if not is_dataset(rewards) or rewards.ndim != 1:
            warnings.append("'rewards' should be 1D [T].")
        else:
            if info["T"] is not None and rewards.shape[0] != info["T"]:
                warnings.append(f"'rewards' length {rewards.shape[0]} != T ({info['T']}).")
            info["shapes"]["rewards"] = tuple(rewards.shape)

    if dones is not None:
        if not is_dataset(dones) or dones.ndim != 1:
            warnings.append("'dones/terminals' should be 1D [T].")
        else:
            if info["T"] is not None and dones.shape[0] != info["T"]:
                warnings.append(f"'dones/terminals' length {dones.shape[0]} != T ({info['T']}).")
            info["shapes"]["dones_or_terminals"] = tuple(dones.shape)
    else:
        warnings.append("Missing 'dones' (or 'terminals') dataset.")

    if states is not None:
        info["has_states"] = True
        if not is_dataset(states) or states.ndim < 1:
            warnings.append("'states' should be a dataset with shape [T, ...].")
        else:
            if info["T"] is not None and states.shape[0] != info["T"]:
                warnings.append(f"'states' length {states.shape[0]} != T ({info['T']}).")
            info["shapes"]["states"] = tuple(states.shape)

    if obs is not None:
        if not is_group(obs):
            errors.append("'obs' should be a group containing per-key datasets.")
        else:
            info["has_obs"] = True
            obs_keys = sorted(list(obs.keys()))
            info["obs_keys"] = obs_keys
            for k in obs_keys:
                ds = obs.get(k, None)
                if ds is None or not is_dataset(ds) or ds.ndim < 1:
                    errors.append(f"obs/{k} should be a dataset with shape [T, ...].")
                    continue
                if info["T"] is not None and ds.shape[0] != info["T"]:
                    warnings.append(f"obs/{k} length {ds.shape[0]} != T ({info['T']}).")
                info["shapes"][f"obs/{k}"] = tuple(ds.shape)

    if next_obs is not None:
        if not is_group(next_obs):
            errors.append("'next_obs' should be a group containing per-key datasets.")
        else:
            info["has_next_obs"] = True
            for k in sorted(list(next_obs.keys())):
                ds = next_obs.get(k, None)
                if ds is None or not is_dataset(ds) or ds.ndim < 1:
                    errors.append(f"next_obs/{k} should be a dataset with shape [T, ...].")
                    continue
                if info["T"] is not None and ds.shape[0] != info["T"]:
                    warnings.append(f"next_obs/{k} length {ds.shape[0]} != T ({info['T']}).")
                info["shapes"][f"next_obs/{k}"] = tuple(ds.shape)

    # num_samples attribute is commonly set per demo
    if "num_samples" in demo.attrs:
        ns = int(demo.attrs["num_samples"])
        if info["T"] is not None and ns != info["T"]:
            warnings.append(f"attrs['num_samples'] ({ns}) != T ({info['T']}).")
    else:
        warnings.append("Missing demo attribute 'num_samples'.")

    return warnings, errors, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5_path")
    ap.add_argument("--strict", action="store_true",
                    help="Treat warnings as errors (exit code 2).")
    args = ap.parse_args()

    warn_total = 0
    err_total = 0

    with h5py.File(args.hdf5_path, "r") as f:
        # Basic structure
        if "data" not in f:
            eprint("[ERROR] Missing top-level 'data' group.")
            return 2
        data = f["data"]

        demos = list_demos(data)
        if not demos:
            eprint("[ERROR] No 'demo_*' groups found under /data.")
            return 2

        print(f"[OK] Found /data with {len(demos)} demos.")
        # env_args and total attrs (optional but recommended)
        has_env, env_args = try_load_env_args(data.attrs)
        if has_env:
            print("[OK] /data has env_args attribute.")
        else:
            print("[WARN] /data missing env_args attribute.")
            warn_total += 1

        if "total" in data.attrs:
            total_attr = int(data.attrs["total"])
            print(f"[OK] /data.attrs['total'] = {total_attr}")
        else:
            print("[WARN] /data missing 'total' attribute.")
            total_attr = None
            warn_total += 1

        # Per-demo checks
        sum_num_samples = 0
        any_obs_keys = set()
        shapes_mismatch = 0

        for di, dname in enumerate(demos):
            demo = data[dname]
            dwarns, derrs, info = check_demo_group(demo)

            if "num_samples" in demo.attrs:
                sum_num_samples += int(demo.attrs["num_samples"])
            elif info["T"] is not None:
                sum_num_samples += int(info["T"])

            warn_total += len(dwarns)
            err_total += len(derrs)

            print(f"\n--- {dname} ---")
            print(f"T: {info['T']}")
            print(f"has_actions={info['has_actions']}  has_states={info['has_states']}  has_obs={info['has_obs']}  has_next_obs={info['has_next_obs']}")
            if info['obs_keys']:
                print("obs keys:", ", ".join(info['obs_keys']))
                any_obs_keys.update(info['obs_keys'])
            for w in dwarns:
                print("[WARN]", w)
            for e in derrs:
                print("[ERROR]", e)

        # Compare 'total'
        if total_attr is not None and sum_num_samples != total_attr:
            print(f"\n[WARN] Sum of per-demo samples ({sum_num_samples}) != /data.attrs['total'] ({total_attr}).")
            warn_total += 1
        else:
            print(f"\n[OK] Sum of per-demo samples = {sum_num_samples} (matches /data.total)." if total_attr is not None else f"\n[INFO] Sum of per-demo samples = {sum_num_samples}.")

        # Optional: mask group validation
        if "mask" in f:
            mask = f["mask"]
            print("\n[INFO] Found /mask:")
            for mk in mask.keys():
                ds = mask[mk]
                if not is_dataset(ds):
                    print(f"[WARN] mask/{mk} is not a dataset.")
                    warn_total += 1
                    continue
                entries = []
                try:
                    entries = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in ds[()]]
                except Exception as exc:
                    print(f"[WARN] Could not read mask/{mk}: {exc}")
                    warn_total += 1
                    continue
                bad = [x for x in entries if x not in demos]
                if bad:
                    print(f"[ERROR] mask/{mk} references unknown demos: {bad[:5]}{'...' if len(bad)>5 else ''}")
                    err_total += 1
                else:
                    print(f"[OK] mask/{mk} has {len(entries)} entries, all refer to existing demos.")

        # Summary
        print("\n==== SUMMARY ====")
        if warn_total:
            print(f"Warnings: {warn_total}")
        if err_total:
            print(f"Errors: {err_total}")
        if err_total > 0 or (args.strict and warn_total > 0):
            return 2
        elif warn_total > 0:
            return 1
        else:
            print("All basic checks passed.")
            return 0


if __name__ == "__main__":
    code = main()
    sys.exit(code)
