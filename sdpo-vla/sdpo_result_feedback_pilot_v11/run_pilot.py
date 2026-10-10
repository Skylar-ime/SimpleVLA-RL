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
from sdpo_result_feedback.policy import MODEL_PRESETS,PolicyConfig,QwenActionPolicy
from sdpo_result_feedback.trainer import GroupedRolloutCollector,TrainingConfig,RolloutCollector,sdpo_update
from sdpo_result_feedback.scenes import create_directional_calibrated_scene_splits,create_directional_local_scene_splits,create_directional_long_scene_splits,create_directional_scene_splits,create_mechanism_scene_splits,create_scene_splits
from sdpo_result_feedback.feedback import select_feedback
from sdpo_result_feedback.evaluation import evaluate,feedback_probe,json_write
from sdpo_result_feedback.sdpo_context import CONDITIONS,build_sdpo_context
from sdpo_result_feedback.reproducibility import create_snapshot

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['probe','train','eval'],default='probe')
    parser.add_argument('--feedback-mode',choices=CONDITIONS,default='sdpo_result')
    parser.add_argument('--transitions',type=int,default=256)
    parser.add_argument('--updates',type=int,default=10)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--microbatch-size',type=int,default=4)
    parser.add_argument('--eval-microbatch-size',type=int)
    parser.add_argument('--parallel-envs',type=int,default=8)
    parser.add_argument('--dev-scenes',type=int,default=64)
    parser.add_argument('--scene-profile',choices=['standard','mechanism_near_contact','directional_push','directional_long','directional_calibrated','directional_local'],default='standard')
    parser.add_argument('--one-step-rollouts',action='store_true')
    parser.add_argument('--grouped-candidates',type=int,default=0)
    parser.add_argument('--expect-feedback-only',action='store_true')
    parser.add_argument('--protocol-file',type=Path)
    parser.add_argument('--model-id',type=str)
    parser.add_argument('--model-revision',type=str)
    parser.add_argument('--backbone',choices=sorted(MODEL_PRESETS),default='qwen3')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite an existing run: {args.output}')
    args.output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; activate sdpo-vla-pilot first')
    if args.transitions<1 or args.updates<1 or not 1<=args.dev_scenes<=128:
        raise SystemExit('Invalid experiment dimensions')
    eval_microbatch_size=args.eval_microbatch_size or args.microbatch_size
    if eval_microbatch_size<1:
        raise SystemExit('Invalid evaluation microbatch size')
    grouped_condition=bool(args.grouped_candidates)
    feedback_mode=args.feedback_mode.removeprefix('sdpo_')
    if grouped_condition != bool(args.grouped_candidates):
        raise SystemExit('Grouped SDPO conditions require --grouped-candidates, and other conditions forbid it')
    if grouped_condition and not args.one_step_rollouts:
        raise SystemExit('Grouped SDPO requires --one-step-rollouts to preserve one initial state per candidate group')
    if args.expect_feedback_only and args.scene_profile!='directional_long':
        raise SystemExit('--expect-feedback-only requires the directional_long scene profile')
    eval_max_decisions=16 if args.scene_profile=='mechanism_near_contact' else 12 if args.scene_profile in {'directional_push','directional_long','directional_calibrated','directional_local'} else 32
    rollout_max_decisions=1 if args.one_step_rollouts else eval_max_decisions
    config=TrainingConfig(args.transitions,args.microbatch_size,args.parallel_envs,rollout_max_decisions,args.seed)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'src').rglob('*.py'))}
    protocol_path=(args.protocol_file if args.protocol_file is not None else
                   ROOT/'protocol_v12.json' if args.feedback_mode.startswith('sdpo_advantage')
                   else ROOT/'protocol_v9.json' if args.feedback_mode=='sdpo_directional'
                   else ROOT/'protocol_v8.json' if args.expect_feedback_only else ROOT/'protocol.json')
    protocol_path=protocol_path.resolve()
    argument_record={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    model_id=args.model_id or str(MODEL_PRESETS[args.backbone])
    reproducibility=create_snapshot(ROOT,args.output,protocol_path,model_id,argument_record)
    json_write(args.output/'run_config.json',{'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'training':asdict(config),'sources':sources,'started_unix':time.time(),
        'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'reproducibility':reproducibility})
    # Test scene parameters are created/frozen here, but never evaluated in development.
    splits=(create_directional_local_scene_splits(ROOT/'artifacts/directional_local_scenes_v1.json')
            if args.scene_profile=='directional_local'
            else create_mechanism_scene_splits(ROOT/'artifacts/mechanism_scenes.json')
            if args.scene_profile=='mechanism_near_contact'
            else create_directional_calibrated_scene_splits(ROOT/'artifacts/directional_calibrated_scenes_v1.json')
            if args.scene_profile=='directional_calibrated'
            else create_directional_long_scene_splits(ROOT/'artifacts/directional_long_scenes_v1.json')
            if args.scene_profile=='directional_long'
            else create_directional_scene_splits(ROOT/'artifacts/directional_scenes_v2.json')
            if args.scene_profile=='directional_push'
            else create_scene_splits(ROOT/'artifacts/scenes.json'))
    start=time.perf_counter()
    default_policy=PolicyConfig(model_id=str(MODEL_PRESETS[args.backbone]))
    policy=QwenActionPolicy(PolicyConfig(seed=args.seed,
        model_id=args.model_id or default_policy.model_id,
        revision=args.model_revision if args.model_revision is not None
        else default_policy.revision if args.model_id is None else None))
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
        collector=(GroupedRolloutCollector(config,splits['train'],args.grouped_candidates)
                   if grouped_condition else RolloutCollector(config,splits['train']))
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
                context=(build_sdpo_context(transitions,args.feedback_mode,args.seed+update_index)
                         if grouped_condition else None)
                if args.expect_feedback_only and (context['solution_context_count']!=0
                        or any(t.result.success for t in transitions)):
                    raise RuntimeError('Feedback-only protocol received an actual same-state success')
                records,valid,donors=((context['records'],context['valid_mask'],context['donor_indices'])
                                      if context is not None else select_feedback(transitions,feedback_mode,args.seed+update_index))
                reflection_records=None
                if args.feedback_mode=='sdpo_reflection':
                    reflection_records=[policy.reflection_feedback(t) for t in transitions]
                    records=reflection_records
                    (path/'reflection_feedback.jsonl').write_text(''.join(
                        json.dumps({'index':i,'scene_id':t.scene_id,'action':t.action_label,
                                    'result':t.result.as_dict(),'reflection':record.interpretation},
                                   ensure_ascii=False,allow_nan=False)+'\n'
                        for i,(t,record) in enumerate(zip(transitions,reflection_records))))
                stats=sdpo_update(policy,transitions,feedback_mode,args.microbatch_size,args.seed+update_index,
                                  teacher_context=context,reflection_records=reflection_records)
                if context is not None:
                    json_write(path/'teacher_context.json',{
                        'condition':context['condition'],'solution_actions':context['solution_actions'],
                        'solution_donor_indices':context['solution_donor_indices'],
                        'include_outcome_feedback':context['include_outcome_feedback'],
                        'valid_mask':context['valid_mask'].tolist(),'donor_indices':context['donor_indices'],
                        'records':[record.as_dict() for record in context['records']],
                        'group_count':context['group_count'],'witness_count':context['witness_count'],
                        'groups_with_witness':context['groups_with_witness'],
                        'solution_context_count':context['solution_context_count']})
                stats.update({'update':update_index+1,'group':args.feedback_mode,'rollout_seconds':rollout_seconds,
                    'contact_count':sum(t.result.contact for t in transitions),
                    'motion_count':sum(t.result.object_motion=='detected' for t in transitions),
                    'progress_counts':dict(Counter(t.result.goal_progress for t in transitions)),
                    'feedback_changed_count':sum(record!=t.result for record,t in zip(records,transitions)),
                    'unmatched_donor_count':sum(d is None for d in donors),
                    'grouped_rollout_count':context['group_count'] if context is not None else None,
                    'group_progress_witness_count':context['witness_count'] if context is not None else None,
                    'groups_with_progress_witness':context['groups_with_witness'] if context is not None else None,
                    'solution_context_count':context['solution_context_count'] if context is not None else None,
                    'subtract_noop_baseline':context.get('subtract_noop_baseline',False) if context is not None else False,
                    'feedback_used_count':sum(context['include_outcome_feedback']) if context is not None else None,
                    'feedback_record_unique_count':len({json.dumps(record.as_dict(),sort_keys=True)
                                                        for record in context['records']}) if context is not None else None,
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
        final=evaluate(policy,splits['dev'][:args.dev_scenes],args.output/'dev_final.json',eval_microbatch_size,
                       max_decisions=eval_max_decisions)
        json_write(args.output/'summary.json',{'updates':metrics,'evaluation':final['summary'],
            'total_train_interactions':args.updates*args.transitions,'elapsed_seconds':time.perf_counter()-start,
            'training_episodes':collector.completed,'note':'Development pilot; not final test or a causal effectiveness claim.'})
    json_write(args.output/'completed.json',{'status':'complete','elapsed_seconds':time.perf_counter()-start,
        'ended_unix':time.time(),'mode':args.mode})
    print('RUN_COMPLETE',flush=True)


if __name__=='__main__': main()
