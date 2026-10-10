"""Run one fixed-seed SimpleVLA/RoboTwin checkpoint diagnostic."""

import os
import sys
import time
import traceback

import numpy as np
import torch

# OpenVLA-OFT selects ALOHA action constants from process arguments.
sys.argv += ["platform_hint", "aloha"]

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/public/home/s2250018/wsy/VLA/RoboTwin-20250810")
CHECKPOINT = os.environ.get(
    "CHECKPOINT",
    "/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_sft_eval_l40",
)
SEED = int(os.environ.get("ROBOTWIN_SEED", "100063"))

os.chdir(ROBOTWIN_ROOT)
sys.path[:0] = [".", "./policy", "./policy/openvla-oft", "./description/utils"]

import yaml

from envs import CONFIGS_PATH
from envs.lift_pot import lift_pot
from generate_episode_instructions import generate_episode_descriptions
import deploy_policy
from deploy_policy import InferenceConfig, Model, encode_obs
from experiments.robot.openvla_utils import DEVICE, normalize_proprio, prepare_images_for_vla


def get_vla_action_with_proprio(cfg, vla, processor, obs, task_label, proprio_projector):
    """Discrete OpenVLA decoder with the SFT proprio projector attached."""
    with torch.inference_mode():
        images = prepare_images_for_vla([obs["full_image"]], cfg)
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = processor(prompt, images[0]).to(DEVICE, dtype=torch.bfloat16)
        proprio = normalize_proprio(obs["state"], vla.norm_stats[cfg.unnorm_key]["proprio"])
        action, _ = vla.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            proprio=proprio,
            proprio_projector=proprio_projector,
        )
    return [action[i] for i in range(len(action))]


def model_get_action_with_proprio(self, observation):
    obs = encode_obs(observation)
    return get_vla_action_with_proprio(
        self.cfg, self.vla, self.processor, obs, obs["instruction"], self.proprio_projector
    )


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_args():
    with open("task_config/demo_randomized.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args.update(
        task_name="lift_pot",
        task_config="demo_randomized",
        ckpt_setting="fixed_seed_diagnostic",
        policy_name="openvla-oft",
        eval_mode=True,
        eval_video_log=False,
        render_freq=0,
        save_data=False,
        collect_data=False,
    )
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiments = yaml.safe_load(f)
    emb = args["embodiment"]
    if len(emb) == 1:
        args["left_robot_file"] = embodiments[emb[0]]["file_path"]
        args["right_robot_file"] = args["left_robot_file"]
        args["dual_arm_embodied"] = True
    elif len(emb) == 3:
        args["left_robot_file"] = embodiments[emb[0]]["file_path"]
        args["right_robot_file"] = embodiments[emb[1]]["file_path"]
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"unsupported embodiment: {emb}")
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        cameras = yaml.safe_load(f)
    cam = cameras[args["camera"]["head_camera_type"]]
    args["head_camera_h"], args["head_camera_w"] = cam["h"], cam["w"]
    return args


def setup_task(args):
    task = lift_pot()
    task.setup_demo(now_ep_num=0, seed=SEED, is_test=True, **args)
    return task


def main():
    args = make_args()
    print(f"CONFIG embodiment={args['embodiment']} seed={SEED}", flush=True)
    t0 = time.time()
    expert = setup_task(args)
    try:
        episode_info = expert.play_once()
        print(
            f"EXPERT plan_success={expert.plan_success} check_success={expert.check_success()} "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
        if not (expert.plan_success and expert.check_success()):
            print("ABORT model: expert seed is not valid", flush=True)
            return 2
    finally:
        expert.close_env()

    cfg = InferenceConfig(
        pretrained_checkpoint=CHECKPOINT,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        use_proprio=True,
        num_images_in_input=1,
        center_crop=True,
        unnorm_key="lift_pot_1k",
        num_open_loop_steps=25,
        lora_rank=32,
    )
    print("LOADING checkpoint", flush=True)
    model = Model(cfg)
    model.get_action = model_get_action_with_proprio.__get__(model, Model)
    print("DECODER discrete_action_tokens + normalized_proprio + projector", flush=True)
    task = setup_task(args)
    try:
        result = generate_episode_descriptions("lift_pot", [episode_info["info"]], 1, seed=0)
        instruction = np.random.choice(result[0]["seen"])
        task.set_instruction(instruction=instruction)
        print(f"INSTRUCTION {instruction}", flush=True)
        while task.take_action_cnt < task.step_lim:
            observation = task.get_obs()
            observation["language"] = task.get_instruction()
            actions = np.asarray(model.get_action(observation))
            print(
                f"ACTION chunk_shape={actions.shape} finite={np.isfinite(actions).all()} "
                f"range=({actions.min():.4f},{actions.max():.4f})",
                flush=True,
            )
            if actions.shape != (25, 14) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid action contract: {actions.shape}")
            task.take_action(actions)
            print(
                f"STEP take_action_cnt={task.take_action_cnt} eval_success={task.eval_success}",
                flush=True,
            )
            if task.eval_success:
                break
        print(
            f"MODEL_RESULT eval_success={task.eval_success} check_success={task.check_success()} "
            f"plan_success={task.plan_success} take_action_cnt={task.take_action_cnt}",
            flush=True,
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        task.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
