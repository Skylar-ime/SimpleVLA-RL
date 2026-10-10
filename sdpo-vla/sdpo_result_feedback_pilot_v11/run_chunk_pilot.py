"""Train and evaluate autoregressive chunk-level SDPO."""
from pathlib import Path
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import time

import numpy as np
import torch

from sdpo_result_feedback.chunk_context import CHUNK_CONDITIONS, build_chunk_context
from sdpo_result_feedback.chunk_evaluation import evaluate_chunk
from sdpo_result_feedback.policy import MODEL_PRESETS, PolicyConfig, QwenActionPolicy
from sdpo_result_feedback.scenes import (create_directional_local_scene_splits,
                                          create_directional_long_scene_splits)
from sdpo_result_feedback.trainer import ChunkRolloutCollector, TrainingConfig, sdpo_chunk_update
from sdpo_result_feedback.reproducibility import create_snapshot


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=sorted(MODEL_PRESETS), default='qwen3')
    parser.add_argument('--condition', choices=CHUNK_CONDITIONS, required=True)
    parser.add_argument('--transitions', type=int, default=64,
                        help='Number of chunks per optimizer update')
    parser.add_argument('--updates', type=int, default=12)
    parser.add_argument('--chunk-length', type=int, default=2)
    parser.add_argument('--grouped-candidates', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--microbatch-size', type=int, default=4)
    parser.add_argument('--dev-scenes', type=int, default=64)
    parser.add_argument('--scene-profile', choices=('directional_long', 'directional_local'),
                        default='directional_long')
    parser.add_argument('--expect-feedback-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite existing run: {args.output}')
    if args.chunk_length < 2 or args.transitions % args.grouped_candidates:
        raise SystemExit('Invalid chunk dimensions')
    if args.condition.startswith('chunk_solution') and args.expect_feedback_only:
        raise SystemExit('Solution conditions require success contexts, not feedback-only mode')
    args.output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; activate sdpo-vla-pilot first')

    protocol = ROOT / ('protocol_chunk_v13.json'
                       if args.condition.startswith('chunk_suffix_')
                       else 'protocol_chunk_v12.json')
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((ROOT / 'src').rglob('*.py'))}
    config = TrainingConfig(
        transitions_per_update=args.transitions,
        microbatch_size=args.microbatch_size,
        parallel_envs=1,
        max_decisions=args.chunk_length,
        seed=args.seed,
        chunk_length=args.chunk_length,
    )
    argument_record={key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    reproducibility=create_snapshot(ROOT,args.output,protocol,
                                    str(MODEL_PRESETS[args.backbone]),argument_record)
    write_json(args.output / 'run_config.json', {
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'protocol': protocol.name,
        'training': asdict(config),
        'sources': sources,
        'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'reproducibility': reproducibility,
        'started_unix': time.time(),
    })
    splits = (create_directional_local_scene_splits(
        ROOT / 'artifacts/directional_local_scenes_v1.json')
        if args.scene_profile == 'directional_local' else
        create_directional_long_scene_splits(
            ROOT / 'artifacts/directional_long_scenes_v1.json'))
    start = time.perf_counter()
    default_policy = PolicyConfig(model_id=str(MODEL_PRESETS[args.backbone]))
    policy = QwenActionPolicy(PolicyConfig(
        seed=args.seed, model_id=default_policy.model_id))
    write_json(args.output / 'policy_audit.json', policy.audit)
    collector = ChunkRolloutCollector(config, splits['train'], args.grouped_candidates)
    metrics = []
    reset_cursor = 0
    try:
        for update_index in range(args.updates):
            tick = time.perf_counter()
            chunks = collector.collect(policy)
            rollout_seconds = time.perf_counter() - tick
            path = args.output / f'update_{update_index + 1:03d}'
            path.mkdir()
            np.savez_compressed(path / 'pre_action_images.npz',
                                images=np.stack([chunk.image for chunk in chunks]))
            (path / 'chunks.jsonl').write_text(''.join(
                json.dumps({'image_index': index, **chunk.as_json_record()}, allow_nan=False) + '\n'
                for index, chunk in enumerate(chunks)))
            context = build_chunk_context(chunks, args.condition)
            context_audit = {}
            for key, value in context.items():
                if key in {'baseline_records', 'records'}:
                    continue
                if key in {'records_by_position', 'baseline_records_by_position'}:
                    context_audit[key] = [
                        [record.as_dict() for record in position_records]
                        for position_records in value
                    ] if value is not None else None
                    continue
                context_audit[key] = value.tolist() if isinstance(value, np.ndarray) else value
            write_json(path / 'chunk_context.json', context_audit)
            stats = sdpo_chunk_update(policy, chunks, args.microbatch_size, context)
            stats.update({
                'update': update_index + 1,
                'group': args.condition,
                'rollout_seconds': rollout_seconds,
                'grouped_rollout_count': context['group_count'],
                'chunk_progress_counts': dict(Counter(chunk.result.goal_progress for chunk in chunks)),
                'safety_failures': sum(chunk.safety_failure is not None for chunk in chunks),
                'successes': sum(chunk.result.success for chunk in chunks),
                'feedback_record_unique_count': len({json.dumps(record.as_dict(), sort_keys=True)
                                                    for record in context['records']}),
                'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            })
            write_json(path / 'update.json', stats)
            with (args.output / 'reset_audit.jsonl').open('a') as handle:
                for row in collector.resets[reset_cursor:]:
                    handle.write(json.dumps(row) + '\n')
            reset_cursor = len(collector.resets)
            policy.save(path / 'checkpoint')
            metrics.append({key: value for key, value in stats.items()
                            if key not in {'teacher_chunk_log_probs',
                                           'teacher_privileged_chunk_log_probs',
                                           'teacher_noop_chunk_log_probs',
                                           'teacher_residual_chunk_log_probs',
                                           'valid_action_mask'}})
            print(json.dumps(metrics[-1]), flush=True)
    finally:
        collector.close()
    evaluation = evaluate_chunk(policy, splits['dev'][:args.dev_scenes], args.chunk_length)
    write_json(args.output / 'dev_final.json', evaluation)
    write_json(args.output / 'summary.json', {
        'updates': metrics,
        'evaluation': evaluation['summary'],
        'total_train_chunks': args.updates * args.transitions,
        'total_train_primitive_actions': args.updates * args.transitions * args.chunk_length,
        'elapsed_seconds': time.perf_counter() - start,
    })
    write_json(args.output / 'completed.json', {
        'status': 'complete', 'mode': 'train', 'ended_unix': time.time(),
        'elapsed_seconds': time.perf_counter() - start,
    })
    print('RUN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
