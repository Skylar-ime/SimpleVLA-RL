"""Run RoboTwin lift_pot through SimpleVLA's own rollout preprocessing/decoder."""

import os
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

sys.argv += ["platform_hint", "aloha"]
ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/public/home/s2250018/wsy/VLA/RoboTwin-20250810")
CHECKPOINT = os.environ.get(
    "CHECKPOINT", "/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_sft_eval_l40"
)
SEED = int(os.environ.get("ROBOTWIN_SEED", "100063"))
PROPRIO_SOURCE = os.environ.get("SIMPLEVLA_PROPRIO_SOURCE", "drive_target")
DO_SAMPLE = os.environ.get("SIMPLEVLA_DO_SAMPLE", "true").lower() == "true"
TEMPERATURE = float(os.environ.get("SIMPLEVLA_TEMPERATURE", "1.6"))
MAX_ACTION_STEPS = int(os.environ.get("SIMPLEVLA_MAX_ACTION_STEPS", "200"))
os.chdir(ROBOTWIN_ROOT)
sys.path[:0] = [
    ".",
    "./policy",
    "./policy/openvla-oft",
    "./description/utils",
    "/public/home/s2250018/wsy/VLA/SimpleVLA-RL",
]

import yaml
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from envs import CONFIGS_PATH
from envs.lift_pot import lift_pot
from generate_episode_instructions import generate_episode_descriptions
from verl.utils.vla_utils.openvla_oft.configuration_prismatic import OpenVLAConfig
from verl.utils.vla_utils.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
from verl.utils.vla_utils.openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from verl.utils.vla_utils.openvla_oft.train_utils import find_checkpoint_file, load_component_state_dict
from verl.workers.rollout.rob_rollout import normalize_proprio


def embodiment_config(path):
    with open(os.path.join(path, "config.yml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_asset_path(value):
    return value.replace("${ASSETS_PATH}", ROBOTWIN_ROOT).replace("$ASSETS_PATH", ROBOTWIN_ROOT)


def make_args():
    with open("task_config/demo_randomized.yml", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args.update(
        task_name="lift_pot", task_config="demo_randomized", ckpt_setting="simplevla_contract",
        policy_name="openvla-oft", eval_mode=True, eval_video_log=False, render_freq=0,
        save_data=False, collect_data=False,
    )
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        embodiments = yaml.safe_load(f)
    for embodiment in embodiments.values():
        embodiment["file_path"] = resolve_asset_path(embodiment["file_path"])
    emb = args["embodiment"]
    args["left_robot_file"] = embodiments[emb[0]]["file_path"]
    if len(emb) == 1:
        args["right_robot_file"] = args["left_robot_file"]
        args["dual_arm_embodied"] = True
    else:
        args["right_robot_file"] = embodiments[emb[1]]["file_path"]
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    args["left_embodiment_config"] = embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = embodiment_config(args["right_robot_file"])
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        cameras = yaml.safe_load(f)
    cam = cameras[args["camera"]["head_camera_type"]]
    args["head_camera_h"], args["head_camera_w"] = cam["h"], cam["w"]
    return args


def setup(args):
    task = lift_pot()
    task.setup_demo(now_ep_num=SEED, seed=SEED, is_test=True, **args)
    return task


def load_model():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
    config.use_proprio = True
    config.proprio_dim = 14
    config = config.to_dict()
    config = OpenVLAConfig.from_dict(config)
    model = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, config=config, torch_dtype=torch.bfloat16, trust_remote_code=False
    )
    try:
        projector_path = find_checkpoint_file(CHECKPOINT, "proprio_projector")
    except (FileNotFoundError, AssertionError):
        projector_path = None
    if projector_path:
        expected = load_component_state_dict(projector_path)
        before = {k: v.detach().float().cpu().clone() for k, v in model.proprio_projector.state_dict().items()}
        model.load_proprio_projector_weights(CHECKPOINT)
        actual = {k: v.detach().float().cpu() for k, v in model.proprio_projector.state_dict().items()}
        max_abs_error = max(float((actual[k] - expected[k].float().cpu()).abs().max()) for k in expected)
        matches = all(torch.allclose(actual[k], expected[k].float().cpu(), atol=0.01, rtol=0.01) for k in expected)
        changed = any(not torch.equal(before[k], actual[k]) for k in actual)
        print(
            f"PROJECTOR source=separate_file path={projector_path} keys={len(expected)} changed={changed} allclose={matches} "
            f"max_abs_error={max_abs_error:.6f} fc1_norm={actual['fc1.weight'].norm().item():.6f} "
            f"fc2_norm={actual['fc2.weight'].norm().item():.6f}", flush=True,
        )
    else:
        actual = {k: v.detach().float().cpu() for k, v in model.proprio_projector.state_dict().items()}
        print(
            f"PROJECTOR source=model_safetensors fc1_norm={actual['fc1.weight'].norm().item():.6f} "
            f"fc2_norm={actual['fc2.weight'].norm().item():.6f}", flush=True,
        )
    model.vision_backbone.set_num_images_in_input(1)
    import json
    with open(os.path.join(CHECKPOINT, "dataset_statistics.json")) as f:
        model.norm_stats = json.load(f)
    model.cuda().eval()
    return model, processor


def process_input(model, processor, observation, instruction):
    from PIL import Image
    from verl.workers.rollout.rob_rollout import center_crop_image
    image = Image.fromarray(observation["observation"]["head_camera"]["rgb"]).convert("RGB")
    image = center_crop_image(image)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    feat = processor(prompt, image)
    input_ids = feat["input_ids"]
    attention_mask = feat["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat((input_ids, torch.tensor([[29871]])), dim=1)
        attention_mask = torch.cat((attention_mask, torch.tensor([[True]])), dim=1)
    input_ids = input_ids.transpose(0, 1).unsqueeze(0).squeeze(-1).cuda()
    attention_mask = attention_mask.transpose(0, 1).unsqueeze(0).squeeze(-1).cuda()
    padding_mask = input_ids.ne(processor.tokenizer.pad_token_id)
    sorted_indices = torch.argsort((~padding_mask).int(), dim=1, descending=True, stable=True)
    input_ids = torch.gather(input_ids, 1, sorted_indices)
    attention_mask = torch.gather(attention_mask, 1, sorted_indices)
    # Match RobHFRollout: processor output stays FP32; inference enters BF16 autocast.
    pixel_values = feat["pixel_values"].to("cuda")
    state = observation["joint_action"]["vector"]
    stats = model.norm_stats["lift_pot_1k"]["proprio"]
    proprio = normalize_proprio(state, stats)
    proprio = torch.from_numpy(proprio).unsqueeze(0).cuda()
    return input_ids, attention_mask, pixel_values, proprio


def assert_wrapper_input_equivalence(model, processor, observation, instruction):
    """Prove this single-episode probe uses RobHFRollout's actual input path."""
    from verl.workers.rollout.rob_rollout import RobHFRollout
    wrapper = RobHFRollout.__new__(RobHFRollout)
    wrapper.module = model
    wrapper.processor = processor
    wrapper.config = SimpleNamespace(
        use_proprio=True,
        task_suite_name="robotwin2_lift_pot",
        center_crop=True,
        vla="openvla-oft",
        num_images_in_input=1,
        unnorm_key="lift_pot_1k",
    )
    wrapped = wrapper.process_input(
        [{"full_image": observation["observation"]["head_camera"]["rgb"], "state": observation["joint_action"]["vector"]}],
        [instruction],
    )
    direct = process_input(model, processor, observation, instruction)
    pixel_diff = (wrapped["pixel_values"].float() - direct[2].float()).abs()
    checks = {
        "ids": torch.equal(wrapped["input_ids"], direct[0]),
        "mask": torch.equal(wrapped["attention_mask"], direct[1]),
        "pixels": torch.allclose(wrapped["pixel_values"], direct[2], atol=1e-6, rtol=1e-6),
        "proprio": torch.equal(wrapped["proprio"], direct[3]),
    }
    print(
        f"WRAPPER_INPUT_EQ={checks} pixel_max_abs_diff={pixel_diff.max().item():.6f} "
        f"pixel_mean_abs_diff={pixel_diff.mean().item():.6f}", flush=True,
    )
    if not all(checks.values()):
        raise AssertionError(checks)


def main():
    args = make_args()
    t0 = time.time()
    expert = setup(args)
    try:
        episode = expert.play_once()
        print(f"EXPERT plan={expert.plan_success} check={expert.check_success()} elapsed={time.time()-t0:.1f}", flush=True)
        if not (expert.plan_success and expert.check_success()):
            return 2
    finally:
        expert.close_env()
    print("LOADING SIMPLEVLA MODEL", flush=True)
    model, processor = load_model()
    task = setup(args)
    try:
        desc = generate_episode_descriptions("lift_pot", [episode["info"]], 1, seed=SEED)
        instruction = np.random.choice(desc[0]["seen"])
        task.set_instruction(instruction)
        print(f"INSTRUCTION {instruction} do_sample={DO_SAMPLE} temperature={TEMPERATURE}", flush=True)
        assert_wrapper_input_equivalence(model, processor, task.get_obs(), instruction)
        first_chunk = True
        while task.take_action_cnt < min(task.step_lim, MAX_ACTION_STEPS):
            obs = task.get_obs()
            if PROPRIO_SOURCE == "real_qpos":
                state = np.asarray(
                    task.robot.get_left_arm_real_jointState()
                    + task.robot.get_right_arm_real_jointState(),
                    dtype=np.float64,
                )
                obs["joint_action"]["vector"] = state
            elif PROPRIO_SOURCE != "drive_target":
                raise ValueError(f"Unknown SIMPLEVLA_PROPRIO_SOURCE={PROPRIO_SOURCE}")
            ids, mask, pixels, proprio = process_input(model, processor, obs, instruction)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                actions, _ = model.generate_action_verl(
                    input_ids=ids, pixel_values=pixels, proprio=proprio,
                    attention_mask=mask, padding_idx=processor.tokenizer.pad_token_id,
                    do_sample=DO_SAMPLE, unnorm_key="lift_pot_1k", temperature=TEMPERATURE,
                )
            if isinstance(actions, torch.Tensor):
                actions = actions[0].detach().cpu().numpy()
            else:
                actions = np.asarray(actions)[0]
            if first_chunk:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predict_actions, _ = model.predict_action(
                        input_ids=ids, pixel_values=pixels, proprio=proprio,
                        proprio_projector=model.proprio_projector,
                        attention_mask=mask, unnorm_key="lift_pot_1k",
                    )
                predict_actions = np.asarray(predict_actions)
                print(
                    f"DECODER_AB first_generate_shape={actions.shape} first_predict_shape={predict_actions.shape} "
                    f"max_abs_diff={np.abs(actions - predict_actions).max():.6f}",
                    flush=True,
                )
                print(
                    f"PROPRIO source={PROPRIO_SOURCE} raw_range=({np.asarray(obs['joint_action']['vector']).min():.6f},"
                    f"{np.asarray(obs['joint_action']['vector']).max():.6f}) "
                    f"norm_range=({proprio.min().item():.6f},{proprio.max().item():.6f})",
                    flush=True,
                )
                print(f"ACTION_FIRST_ROWS={np.array2string(actions[:3], precision=4, suppress_small=True)}", flush=True)
                first_chunk = False
            print(f"ACTION shape={actions.shape} finite={np.isfinite(actions).all()} range=({actions.min():.4f},{actions.max():.4f})", flush=True)
            if actions.shape != (25, 14):
                raise RuntimeError(actions.shape)
            task.take_action(actions)
            pot_z = task.pot.get_pose().p[2]
            left_tcp = np.asarray(task.robot.get_left_tcp_pose()[:3])
            right_tcp = np.asarray(task.robot.get_right_tcp_pose()[:3])
            left_contact = np.asarray(task.pot.get_contact_point(0)[:3])
            right_contact = np.asarray(task.pot.get_contact_point(1)[:3])
            print(
                f"STEP count={task.take_action_cnt} success={task.eval_success} pot_z={pot_z:.4f} "
                f"grasp_dist=({np.linalg.norm(left_tcp-left_contact):.4f},{np.linalg.norm(right_tcp-right_contact):.4f}) "
                f"gripper=({task.robot.get_left_gripper_val():.4f},{task.robot.get_right_gripper_val():.4f})",
                flush=True,
            )
            if task.eval_success:
                break
        print(f"RESULT eval={task.eval_success} check={task.check_success()} plan={task.plan_success} count={task.take_action_cnt}", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        task.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
