"""Probe one published RoboTwin success seed in an unmodified worktree."""

import importlib
import os
import sys
import types

import yaml


ROOT = os.environ["ROBOTWIN_ROOT"]
TASK_NAME = "lift_pot"
SEEDS = [int(x) for x in os.environ.get("ROBOTWIN_SEEDS", os.environ.get("ROBOTWIN_SEED", "100006")).split(",")]

os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "description", "utils"))

from envs import CONFIGS_PATH


def resolve_asset_path(value):
    return value.replace("${ASSETS_PATH}", ROOT).replace("$ASSETS_PATH", ROOT)


def load_args():
    with open(os.path.join(ROOT, "task_config", "demo_randomized.yml"), encoding="utf-8") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        embodiments = yaml.load(f, Loader=yaml.FullLoader)
    for embodiment in embodiments.values():
        embodiment["file_path"] = resolve_asset_path(embodiment["file_path"])

    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        robot_file = embodiments[embodiment[0]]["file_path"]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiments[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiments[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"unsupported embodiment: {embodiment}")
    for side in ("left", "right"):
        with open(os.path.join(args[f"{side}_robot_file"], "config.yml"), encoding="utf-8") as f:
            args[f"{side}_embodiment_config"] = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        camera_configs = yaml.load(f, Loader=yaml.FullLoader)
    camera = camera_configs[args["camera"]["head_camera_type"]]
    args["head_camera_h"] = camera["h"]
    args["head_camera_w"] = camera["w"]
    args.update(
        task_name=TASK_NAME,
        task_config="demo_randomized",
        ckpt_setting="demo_randomized",
        eval_mode=True,
        eval_video_log=False,
        render_freq=0,
        instruction_type="unseen",
    )
    return args


import numpy as np

for seed in SEEDS:
    task = getattr(importlib.import_module(f"envs.{TASK_NAME}"), TASK_NAME)()
    args = load_args()
    print("CONFIG", args["embodiment"], args["camera"], flush=True)
    # Author seed collection binds the episode/output id and physics seed alike.
    task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **args)
    print("SETUP", seed, task.get_obs()["joint_action"]["vector"].shape, flush=True)
    original_together_move = task.together_move_to_pose

    def trace_together_move(self, left_target_pose, right_target_pose, *args, **kwargs):
        stage = len(self.left_joint_path) + 1
        print(
            f"PLAN_BEGIN stage={stage} left_pose={np.asarray(left_target_pose)[:3].round(4).tolist()} "
            f"right_pose={np.asarray(right_target_pose)[:3].round(4).tolist()}", flush=True,
        )
        result = original_together_move(left_target_pose, right_target_pose, *args, **kwargs)
        left_status = self.left_joint_path[-1].get("status") if self.left_joint_path else None
        right_status = self.right_joint_path[-1].get("status") if self.right_joint_path else None
        print(
            f"PLAN_END stage={stage} left_status={left_status} right_status={right_status} "
            f"plan_success={self.plan_success}", flush=True,
        )
        return result

    task.together_move_to_pose = types.MethodType(trace_together_move, task)
    original_take_dense_action = task.take_dense_action

    def trace_take_dense_action(self, control_seq, *args, **kwargs):
        def endpoint(component, field, fallback=None):
            if component is None:
                return fallback
            values = np.asarray(component[field])
            return values[0].round(4).tolist(), values[-1].round(4).tolist(), len(values)

        print(
            "EXPERT_CONTROL "
            f"left_arm={endpoint(control_seq['left_arm'], 'position')} "
            f"left_gripper={endpoint(control_seq['left_gripper'], 'result')} "
            f"right_arm={endpoint(control_seq['right_arm'], 'position')} "
            f"right_gripper={endpoint(control_seq['right_gripper'], 'result')}",
            flush=True,
        )
        return original_take_dense_action(control_seq, *args, **kwargs)

    task.take_dense_action = types.MethodType(trace_take_dense_action, task)
    task.play_once()
    pot_pose = task.pot.get_pose()
    left_end = np.asarray(task.robot.get_left_tcp_pose()[:3])
    right_end = np.asarray(task.robot.get_right_tcp_pose()[:3])
    left_grasp = np.asarray(task.pot.get_contact_point(0)[:3])
    right_grasp = np.asarray(task.pot.get_contact_point(1)[:3])
    print(
        f"EXPERT seed={seed} plan_success={task.plan_success} success={task.check_success()} "
        f"pot_z={pot_pose.p[2]:.4f} left_grasp_distance={np.linalg.norm(left_end-left_grasp):.4f} "
        f"right_grasp_distance={np.linalg.norm(right_end-right_grasp):.4f} "
        f"gripper_open=({task.robot.get_left_gripper_val():.4f},{task.robot.get_right_gripper_val():.4f})",
        flush=True,
    )
    task.close_env(clear_cache=True)
