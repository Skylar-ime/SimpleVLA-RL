"""Measure whether model-generated reflection improves action selection."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sdpo_result_feedback.actions import ACTION_LABELS
from sdpo_result_feedback.env_adapter import InvalidScene, PandaPushOutcomePilot
from sdpo_result_feedback.evaluation import json_write
from sdpo_result_feedback.feedback import FeedbackRecord, causal_feedback
from sdpo_result_feedback.policy import PolicyConfig, QwenActionPolicy
from sdpo_result_feedback.scenes import create_directional_local_scene_splits
from sdpo_result_feedback.trainer import observation_transition


def reflection_prompt(transition, result: FeedbackRecord) -> str:
    return (
        "You control a robot arm. Read the original image and task.\n"
        f"Task: {transition.instruction}\n"
        "Actions: A=world +x, B=world -x, C=world +y, D=world -y, E=hold.\n"
        f"The attempted action was {transition.action_label}.\n"
        "Observed result from executing that action at the original state:\n"
        f"Contact: {result.contact}. Cube motion: {result.object_motion}. "
        f"Target distance change: {result.goal_progress}. Success: {result.success}.\n\n"
        "Give one short correction for choosing the next action at the ORIGINAL state. "
        "State what went wrong and which action should be tried next."
    )


@torch.no_grad()
def generate_reflection(policy: QwenActionPolicy, transition, result: FeedbackRecord) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": reflection_prompt(transition, result)},
    ]}]
    text = policy.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = policy.processor(
        text=[text],
        images=[Image.fromarray(transition.image).convert("RGB")],
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(policy.device) if torch.is_tensor(value) else value
              for key, value in inputs.items()}
    output = policy.model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
    )
    prompt_length = inputs["input_ids"].shape[1]
    return policy.processor.batch_decode(
        output[:, prompt_length:], skip_special_tokens=True
    )[0].strip()


def action_distribution(policy, transitions, records=None):
    chunks = []
    for start in range(0, len(transitions), 4):
        batch_records = None if records is None else records[start:start + 4]
        log_probs = policy.log_probs(transitions[start:start + 4],
            feedback=records is not None, feedback_records=batch_records, grad=False)
        chunks.append(log_probs.exp().cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {args.output}")
    args.output.mkdir(parents=True)

    root = Path(__file__).resolve().parents[1]
    splits = create_directional_local_scene_splits(
        root / "artifacts/directional_local_scenes_v1.json"
    )
    policy = QwenActionPolicy(PolicyConfig(seed=args.seed))
    env = PandaPushOutcomePilot(seed=args.seed, max_decisions=1)
    transitions = []
    counterfactual = []
    counterfactual_success = []
    try:
        for scene in splits["dev"][:args.scenes]:
            try:
                observation = env.reset(f"reflection_{scene.scene_id}", scene)
            except InvalidScene:
                continue
            snapshot = env.snapshot()
            try:
                env.restore(snapshot)
                original = observation_transition(env, observation)
                action = policy.sample(original)
                _, transition = env.step(action)
                values = []
                successes = []
                for label in ACTION_LABELS:
                    env.restore(snapshot)
                    _, result = env.step(label)
                    values.append(result.distance_before_m - result.distance_after_m)
                    successes.append(result.result.success)
                transitions.append(transition)
                counterfactual.append(values)
                counterfactual_success.append(successes)
            finally:
                env.remove_snapshot(snapshot)
    finally:
        env.close()

    reflections = []
    for index, transition in enumerate(transitions):
        text = generate_reflection(policy, transition, transition.result)
        reflections.append(text)
        print(json.dumps({"index": index, "reflection": text}, ensure_ascii=False), flush=True)

    student = action_distribution(policy, transitions)
    direct = action_distribution(policy, transitions, [t.result for t in transitions])
    rule_records = [causal_feedback(t.action_label, t.result) for t in transitions]
    rule = action_distribution(policy, transitions, rule_records)
    model_records = [replace(t.result, interpretation=text)
                     for t, text in zip(transitions, reflections)]
    model = action_distribution(policy, transitions, model_records)
    neutral_records = [FeedbackRecord(None, None, None, None) for _ in transitions]
    neutral = action_distribution(policy, transitions, neutral_records)
    values = np.asarray(counterfactual)
    success = np.asarray(counterfactual_success, dtype=float)
    base_progress = (student * values).sum(-1)
    base_success = (student * success).sum(-1)
    suggested_actions = []
    for reflection in reflections:
        match = re.search(r"Action:\s*([ABCDE])(?:\b|=)", reflection)
        suggested_actions.append(match.group(1) if match else None)

    def summarize(probs):
        progress_difference = (probs * values).sum(-1) - base_progress
        success_difference = (probs * success).sum(-1) - base_success
        return {
            "expected_progress_difference_vs_student_m": float(progress_difference.mean()),
            "expected_success_difference_vs_student": float(success_difference.mean()),
            "expected_progress_difference_vs_result_feedback_m": float(
                ((probs * values).sum(-1) - (direct * values).sum(-1)).mean()),
            "mean_total_variation_vs_student": float(.5 * np.abs(probs - student).sum(-1).mean()),
            "mean_total_variation_vs_result_feedback": float(.5 * np.abs(probs - direct).sum(-1).mean()),
            "argmax_actions": {
                label: int((probs.argmax(-1) == index).sum())
                for index, label in enumerate(ACTION_LABELS)
            },
        }

    report = {
        "summary": {
            "scenes": len(transitions),
            "counterfactual_interactions": len(transitions) * len(ACTION_LABELS),
            "groups": {
                "result": summarize(direct),
                "causal_rule": summarize(rule),
                "model_reflection": summarize(model),
                "neutral": summarize(neutral),
            },
            "note": "Reflection text is generated by the frozen policy; no training occurred.",
        },
        "reflections": reflections,
        "suggested_actions": suggested_actions,
        "suggested_action_success_rate": (
            sum(action is not None and success[index, ACTION_LABELS.index(action)] > 0
                for index, action in enumerate(suggested_actions)) / len(transitions)
            if transitions else None),
        "records": [t.as_json_record() for t in transitions],
        "counterfactual_progress_m": values.tolist(),
        "counterfactual_success": success.astype(bool).tolist(),
        "probabilities": {
            "student": student.tolist(),
            "result": direct.tolist(),
            "causal_rule": rule.tolist(),
            "model_reflection": model.tolist(),
            "neutral": neutral.tolist(),
        },
    }
    json_write(args.output / "report.json", report)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
