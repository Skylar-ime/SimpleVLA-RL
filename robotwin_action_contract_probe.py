"""Verify RoboTwin action execution independently of model inference."""

import os
import sys

import numpy as np
import yaml

sys.path[:0] = [".", "./description/utils"]

from envs import CONFIGS_PATH
from envs.lift_pot import lift_pot


def make_args(config_name):
    with open(f"task_config/{config_name}.yml", encoding="utf-8") as file:
        args = yaml.safe_load(file)
    args.update(
        task_name="lift_pot",
        task_config=config_name,
        ckpt_setting="action_contract_probe",
        eval_mode=True,
        eval_video_log=False,
        render_freq=0,
    )
    with open(CONFIGS_PATH + "_embodiment_config.yml", encoding="utf-8") as file:
        embodiments = yaml.safe_load(file)
    selected = args["embodiment"]
    left_path = embodiments[selected[0]]["file_path"]
    right_path = left_path if len(selected) == 1 else embodiments[selected[1]]["file_path"]
    args["dual_arm_embodied"] = len(selected) == 1
    if len(selected) == 3:
        args["embodiment_dis"] = selected[2]
    for side, path in (("left", left_path), ("right", right_path)):
        args[f"{side}_robot_file"] = path
        with open(os.path.join(path, "config.yml"), encoding="utf-8") as file:
            args[f"{side}_embodiment_config"] = yaml.safe_load(file)
    with open(CONFIGS_PATH + "_camera_config.yml", encoding="utf-8") as file:
        cameras = yaml.safe_load(file)
    args["head_camera_h"] = cameras["D435"]["h"]
    args["head_camera_w"] = cameras["D435"]["w"]
    return args


def run(config_name, seed=100000):
    task = lift_pot()
    try:
        task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **make_args(config_name))
        before = np.asarray(task.get_obs()["joint_action"]["vector"], dtype=np.float64)
        action = np.repeat(before[None, :], 25, axis=0)
        task.take_action(action)
        after = np.asarray(task.get_obs()["joint_action"]["vector"], dtype=np.float64)
        print(
            "IDENTITY",
            config_name,
            "shape",
            tuple(before.shape),
            "step_count",
            task.take_action_cnt,
            "max_abs_delta",
            float(np.abs(after - before).max()),
            "finite",
            bool(np.isfinite(after).all()),
            flush=True,
        )
    finally:
        task.close_env(clear_cache=True)


for config_name in ("demo_randomized", "demo_randomized_aloha"):
    run(config_name)
print("PROBE_DONE", flush=True)
