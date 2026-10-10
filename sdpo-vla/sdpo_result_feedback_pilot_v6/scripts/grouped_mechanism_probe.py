"""Check grouped self-distillation context before expensive online training."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from sdpo_result_feedback.policy import PolicyConfig,QwenActionPolicy
from sdpo_result_feedback.scenes import create_directional_calibrated_scene_splits
from sdpo_result_feedback.trainer import GroupedRolloutCollector,TrainingConfig,build_group_teacher_context
from sdpo_result_feedback.evaluation import json_write


def batched_teacher(policy,transitions,context,batch_size):
    output=np.full((len(transitions),5),np.nan,dtype=np.float32)
    valid=np.asarray(context['valid_mask'],dtype=bool)
    with torch.no_grad():
        for start in range(0,len(transitions),batch_size):
            indices=np.flatnonzero(valid[start:start+batch_size])+start
            if not len(indices):
                continue
            records=[context['records'][index] for index in indices]
            solutions=[context['solution_actions'][index] for index in indices]
            values=policy.teacher_log_probs([transitions[index] for index in indices],feedback=True,
                feedback_records=records,solution_actions=solutions,
                include_outcome_feedback=context['include_outcome_feedback']).exp().cpu().numpy()
            output[indices]=values
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--groups',type=int,default=64)
    parser.add_argument('--candidates',type=int,default=8)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    if args.groups<8 or args.candidates<2:
        raise SystemExit('Probe dimensions too small')
    args.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1]
    splits=create_directional_calibrated_scene_splits(root/'artifacts/directional_calibrated_scenes_v1.json')
    policy=QwenActionPolicy(PolicyConfig(seed=args.seed))
    collector=GroupedRolloutCollector(TrainingConfig(args.groups*args.candidates,4,8,1,args.seed),
                                      splits['train'],args.candidates)
    start=time.perf_counter()
    try:
        transitions=collector.collect(policy)
    finally:
        collector.close()
    contexts={name:build_group_teacher_context(transitions,name,args.seed)
              for name in ['solution_only','solution_plus_binary','solution_plus_causal',
                           'solution_plus_causal_shuffled']}
    probabilities={name:batched_teacher(policy,transitions,context,args.batch_size)
                   for name,context in contexts.items()}
    groups={}
    for index,transition in enumerate(transitions):
        groups.setdefault(transition.rollout_group_id,[]).append(index)
    labels=('A','B','C','D','E')
    metrics={}
    for name,context in contexts.items():
        valid=np.asarray(context['valid_mask'],dtype=bool)
        argmax=np.nanargmax(np.where(np.isnan(probabilities[name]),-np.inf,probabilities[name]),axis=1)
        solution=np.asarray([labels.index(action) if action is not None else -1
                             for action in context['solution_actions']])
        solution_mask=valid & (solution>=0)
        predicted_progress=[]
        for index,transition in enumerate(transitions):
            if not valid[index]:
                continue
            same_group=groups[transition.rollout_group_id]
            candidates=[candidate for candidate in same_group if transitions[candidate].action_label==labels[argmax[index]]]
            if candidates:
                predicted_progress.append(transitions[candidates[0]].result.progress_m)
        metrics[name]={
            'valid':int(valid.sum()),
            'solution_contexts':int(solution_mask.sum()),
            'teacher_solution_argmax_accuracy':float((argmax[solution_mask]==solution[solution_mask]).mean()) if solution_mask.any() else None,
            'teacher_argmax_counts':{label:int((argmax[valid]==i).sum()) for i,label in enumerate(labels)},
            'mean_predicted_group_progress_m':float(np.mean(predicted_progress)) if predicted_progress else None,
            'mean_entropy_nats':float(np.mean(-np.nansum(probabilities[name]*np.log(np.clip(probabilities[name],1e-30,1)),axis=1)[valid])),
        }
    pair_metrics={}
    for left,right in [('solution_plus_causal','solution_only'),
                       ('solution_plus_causal','solution_plus_binary'),
                       ('solution_plus_causal','solution_plus_causal_shuffled')]:
        shared=np.asarray(contexts[left]['valid_mask'],dtype=bool)&np.asarray(contexts[right]['valid_mask'],dtype=bool)
        tv=.5*np.abs(probabilities[left]-probabilities[right]).sum(-1)
        pair_metrics[f'{left}_minus_{right}']={
            'shared_valid':int(shared.sum()),
            'mean_total_variation':float(np.nanmean(tv[shared])) if shared.any() else None,
            'mean_progress_delta_m':float(metrics[left]['mean_predicted_group_progress_m']-metrics[right]['mean_predicted_group_progress_m'])
                if metrics[left]['mean_predicted_group_progress_m'] is not None and metrics[right]['mean_predicted_group_progress_m'] is not None else None,
        }
    report={'status':'development_grouped_mechanism_probe','groups':args.groups,'candidates_per_group':args.candidates,
            'transitions':len(transitions),'elapsed_seconds':time.perf_counter()-start,'metrics':metrics,
            'pair_metrics':pair_metrics,
            'gate':{
                'has_self_generated_solutions':metrics['solution_plus_causal']['solution_contexts']>0,
                'rich_changes_solution_teacher':pair_metrics['solution_plus_causal_minus_solution_only']['mean_total_variation']>0.01,
                'rich_beats_binary_progress':pair_metrics['solution_plus_causal_minus_solution_plus_binary']['mean_progress_delta_m']>0,
                'rich_beats_shuffled_progress':pair_metrics['solution_plus_causal_minus_solution_plus_causal_shuffled']['mean_progress_delta_m']>0,
            },
            'note':'Mechanism probe uses only self-generated same-state candidates and measured outcomes; no expert action labels.'}
    report['gate']['status']='passed' if all(report['gate'].values()) else 'failed'
    json_write(args.output/'report.json',report)
    print(json.dumps(report,indent=2))
    if report['gate']['status']!='passed':
        raise SystemExit('Grouped mechanism gate failed')


if __name__=='__main__':
    main()
