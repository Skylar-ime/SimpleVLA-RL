"""Reproducible stage-A diagnostic and bounded stage-B online SDPO runs."""
from pathlib import Path
import argparse
from dataclasses import asdict
from collections import Counter
import hashlib
import json
import time
import numpy as np
import torch
from sdpo_result_feedback.policy import PolicyConfig,QwenActionPolicy
from sdpo_result_feedback.trainer import TrainingConfig,RolloutCollector,sdpo_update
from sdpo_result_feedback.scenes import create_directional_calibrated_scene_splits,create_directional_long_scene_splits,create_directional_scene_splits,create_mechanism_scene_splits,create_scene_splits
from sdpo_result_feedback.feedback import select_feedback
from sdpo_result_feedback.evaluation import evaluate,feedback_probe,json_write

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['probe','train','eval'],default='probe')
    parser.add_argument('--feedback-mode',choices=['result','result_interpreted','causal','causal_shuffled','binary','shuffled','hard_shuffled','neutral','no_feedback'],default='result')
    parser.add_argument('--transitions',type=int,default=256)
    parser.add_argument('--updates',type=int,default=10)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--microbatch-size',type=int,default=4)
    parser.add_argument('--parallel-envs',type=int,default=8)
    parser.add_argument('--dev-scenes',type=int,default=64)
    parser.add_argument('--scene-profile',choices=['standard','mechanism_near_contact','directional_push','directional_long','directional_calibrated'],default='standard')
    parser.add_argument('--one-step-rollouts',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite an existing run: {args.output}')
    args.output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; activate sdpo-vla-pilot first')
    if args.transitions<1 or args.updates<1 or not 1<=args.dev_scenes<=128:
        raise SystemExit('Invalid experiment dimensions')
    eval_max_decisions=16 if args.scene_profile=='mechanism_near_contact' else 12 if args.scene_profile in {'directional_push','directional_long','directional_calibrated'} else 32
    rollout_max_decisions=1 if args.one_step_rollouts else eval_max_decisions
    config=TrainingConfig(args.transitions,args.microbatch_size,args.parallel_envs,rollout_max_decisions,args.seed)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'src').rglob('*.py'))}
    json_write(args.output/'run_config.json',{'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'training':asdict(config),'sources':sources,'started_unix':time.time(),'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()})
    # Test scene parameters are created/frozen here, but never evaluated in development.
    splits=(create_mechanism_scene_splits(ROOT/'artifacts/mechanism_scenes.json')
            if args.scene_profile=='mechanism_near_contact'
            else create_directional_calibrated_scene_splits(ROOT/'artifacts/directional_calibrated_scenes_v1.json')
            if args.scene_profile=='directional_calibrated'
            else create_directional_long_scene_splits(ROOT/'artifacts/directional_long_scenes_v1.json')
            if args.scene_profile=='directional_long'
            else create_directional_scene_splits(ROOT/'artifacts/directional_scenes_v2.json')
            if args.scene_profile=='directional_push'
            else create_scene_splits(ROOT/'artifacts/scenes.json'))
    start=time.perf_counter()
    policy=QwenActionPolicy(PolicyConfig(seed=args.seed))
    json_write(args.output/'policy_audit.json',policy.audit)
    print('MODEL_READY',round(time.perf_counter()-start,2),flush=True)
    if args.mode=='probe':
        report=feedback_probe(policy,splits['dev'][:args.dev_scenes],args.output/'probe',args.microbatch_size,args.seed)
        print(json.dumps(report['summary']),flush=True)
    elif args.mode=='eval':
        report=evaluate(policy,splits['dev'][:args.dev_scenes],args.output/'evaluation.json',args.microbatch_size,
                        max_decisions=eval_max_decisions)
        print(json.dumps(report['summary']),flush=True)
    else:
        collector=RolloutCollector(config,splits['train'])
        metrics=[];reset_cursor=0
        try:
            for update_index in range(args.updates):
                tick=time.perf_counter()
                transitions=collector.collect(policy)
                rollout_seconds=time.perf_counter()-tick
                path=args.output/f'update_{update_index+1:03d}'
                path.mkdir()
                # Persist exact pre-action RGB and real outcomes before any model update.
                np.savez_compressed(path/'pre_action_images.npz',images=np.stack([t.image for t in transitions]))
                with (path/'transitions.jsonl').open('w') as f:
                    for i,t in enumerate(transitions):
                        f.write(json.dumps({'image_index':i,**t.as_json_record()},allow_nan=False)+'\n')
                records,valid,donors=select_feedback(transitions,args.feedback_mode,args.seed+update_index)
                stats=sdpo_update(policy,transitions,args.feedback_mode,args.microbatch_size,args.seed+update_index)
                stats.update({'update':update_index+1,'group':args.feedback_mode,'rollout_seconds':rollout_seconds,
                    'contact_count':sum(t.result.contact for t in transitions),
                    'motion_count':sum(t.result.object_motion=='detected' for t in transitions),
                    'progress_counts':dict(Counter(t.result.goal_progress for t in transitions)),
                    'feedback_changed_count':sum(record!=t.result for record,t in zip(records,transitions)),
                    'unmatched_donor_count':sum(d is None for d in donors),
                    'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
                json_write(path/'update.json',stats)
                with (args.output/'reset_audit.jsonl').open('a') as f:
                    for row in collector.resets[reset_cursor:]: f.write(json.dumps(row)+'\n')
                reset_cursor=len(collector.resets)
                policy.save(path/'checkpoint')
                compact={k:v for k,v in stats.items() if k not in {'teacher_log_probs','donor_indices','valid_mask'}}
                metrics.append(compact)
                print(json.dumps(compact),flush=True)
        finally:
            collector.close()
        final=evaluate(policy,splits['dev'][:args.dev_scenes],args.output/'dev_final.json',args.microbatch_size,
                       max_decisions=eval_max_decisions)
        json_write(args.output/'summary.json',{'updates':metrics,'evaluation':final['summary'],
            'total_train_interactions':args.updates*args.transitions,'elapsed_seconds':time.perf_counter()-start,
            'training_episodes':collector.completed,'note':'Development pilot; not final test or a causal effectiveness claim.'})
    json_write(args.output/'completed.json',{'status':'complete','elapsed_seconds':time.perf_counter()-start,
        'ended_unix':time.time(),'mode':args.mode})
    print('RUN_COMPLETE',flush=True)


if __name__=='__main__': main()
