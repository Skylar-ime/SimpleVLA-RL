#!/usr/bin/env bash
set -euo pipefail

# Four-L40 single-task run. Keep author lift_pot contract; reduce per-update
# batch sizes so one model replica fits on each 48 GiB card.
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# This node's IB loopback path fails during local NCCL initialization. FSDP
# workers reside on one host, so use local shared-memory/P2P transports.
export NCCL_IB_DISABLE=1

source /public/home/s2250018/miniconda3/etc/profile.d/conda.sh
conda activate robotwin-pristine

PROJECT_NAME="SimpleVLA-RL-robotwin2-l40"
EXPERIMENT_NAME="twin2_lift_pot_l40_4gpu_sft_start_binary"
ROOT="/public/home/s2250018/wsy/VLA/SimpleVLA-RL-462f23d"
ROBOTWIN_ROOT="/public/home/s2250018/wsy/VLA/RoboTwin-462f23d"
SFT_MODEL_PATH="/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_sft_eval_l40"
OUTPUT_ROOT="/public/home/s2250018/wsy/VLA/outputs/lift_pot_l40_4gpu"

cd "$ROOT"
export SIMPLEVLA_ROBOTWIN2_PATH="$ROBOTWIN_ROOT"
export PYTHONPATH="$ROBOTWIN_ROOT:$ROOT:${PYTHONPATH:-}"
# Keep the author's 8-sample GRPO group. A group of 4 made the already
# low-success SFT policy frequently produce all-zero groups, which the
# binary-reward accuracy filter correctly discards.
# L40 memory: one per-rank PPO micro-batch plus activation checkpointing keeps
# actor backward below 48 GiB. FSDP divides this global value by 4 ranks, so
# pass 4 here to obtain the required per-rank value of 1.
python -u -m verl.trainer.main_ppo \
    data.task_suite_name=robotwin2_lift_pot \
    data.num_trials_per_task=1000 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path="$SFT_MODEL_PATH" \
    actor_rollout_ref.model.vla=openvla-oft \
    actor_rollout_ref.model.action_token_len=14 \
    actor_rollout_ref.model.action_chunks_len=25 \
    actor_rollout_ref.model.resume=False \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.twin2_task_config=demo_randomized \
    actor_rollout_ref.rollout.twin2_instruction_type=seen \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=True \
    actor_rollout_ref.rollout.val_micro_batch_size=4 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name="$EXPERIMENT_NAME" \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=robotwin2_lift_pot_1k \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=robotwin2_lift_pot \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint="$SFT_MODEL_PATH" \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger='[console]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_ROOT/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=4 \
    trainer.total_epochs=10 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env="$ROOT/align.json" \
    trainer.val_before_train=True \
    "$@"
