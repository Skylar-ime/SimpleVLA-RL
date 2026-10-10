"""Read-only verification of saved experiment artifacts; never rewrite measured records."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json
import numpy as np
import torch
from safetensors import safe_open


def outcome_verdict(result):
    if result['goal_progress']=='closer': return 'ACCEPT'
    if result['goal_progress']=='farther' or not result['contact']: return 'REJECT'
    return 'RECONSIDER'


def audit(directory):
    output={};errors=[]
    groups=[name for name in ['result','result_interpreted','causal','causal_shuffled','binary','shuffled','hard_shuffled','neutral','no_feedback',
                              'solution_only','solution_plus_causal','solution_plus_causal_shuffled','solution_plus_binary']
            if (directory/name/'completed.json').exists()]
    first_batches={}
    for group in groups:
        run=directory/group
        if not (run/'completed.json').exists():
            raise RuntimeError(f'Run has not completed: {run}')
        config=json.loads((run/'run_config.json').read_text())
        for name,digest in config['sources'].items():
            actual=hashlib.sha256((directory.parents[1]/name).read_bytes()).hexdigest()
            if actual!=digest: errors.append(f'{group}: source drift {name}')
        total=0;valid=0;donor_errors=[];episode_ids=set();counts=Counter();fields=Counter();contacts=0;changed=0;verdict_changed=0
        first_contacts={};frozen_metadata=None;trainable_changed=None;steps=[];manifest=[]
        for index in range(1,config['arguments']['updates']+1):
            batch=run/f'update_{index:03d}'
            rows=[json.loads(line) for line in (batch/'transitions.jsonl').read_text().splitlines()]
            update=json.loads((batch/'update.json').read_text())
            grouped=group.startswith('solution_')
            context=json.loads((batch/'teacher_context.json').read_text()) if grouped else None
            images=np.load(batch/'pre_action_images.npz')['images']
            if images.shape!=(len(rows),448,448,3) or images.dtype!=np.uint8:
                errors.append(f'{group}/{index}: image shape/dtype')
            if index==1:
                first_batches[group]=hashlib.sha256((batch/'transitions.jsonl').read_bytes()).hexdigest()
            expected_count=int(config['arguments']['transitions'])
            if len(rows)!=expected_count or update['executed']!=len(rows) or update['optimizer_steps']!=1:
                errors.append(f'{group}/{index}: interaction/optimizer budget')
            probabilities=np.exp(np.asarray(update['teacher_log_probs']))
            if probabilities.shape!=(len(rows),5) or not np.allclose(probabilities.sum(1),1,atol=1e-6):
                errors.append(f'{group}/{index}: invalid teacher probabilities')
            expected_valid=(context['valid_mask'] if grouped
                            else [t['safety_failure'] is None and not t['result']['success'] for t in rows])
            if update['valid_mask']!=expected_valid or update['valid']!=sum(expected_valid):
                errors.append(f'{group}/{index}: valid transition mask mismatch')
            if grouped:
                if len(context['solution_actions'])!=len(rows) or len(context['records'])!=len(rows):
                    errors.append(f'{group}/{index}: grouped context length mismatch')
                grouped_rows={}
                for row_index,row in enumerate(rows):
                    group_id=row.get('rollout_group_id')
                    if group_id is None or row.get('rollout_candidate_index') is None:
                        errors.append(f'{group}/{index}/{row_index}: missing grouped rollout identity')
                        continue
                    grouped_rows.setdefault(group_id,[]).append(row_index)
                if len(grouped_rows)!=context['group_count']:
                    errors.append(f'{group}/{index}: grouped rollout count mismatch')
                for row_index,row in enumerate(rows):
                    members=grouped_rows.get(row.get('rollout_group_id'),[])
                    witnesses=[candidate for candidate in members
                               if rows[candidate]['safety_failure'] is None
                               and not rows[candidate]['result']['success']
                               and rows[candidate]['result']['goal_progress']=='closer']
                    witness=witnesses[0] if witnesses else None
                    expected_solution=None if witness is None or witness==row_index else rows[witness]['action_label']
                    if context['solution_actions'][row_index]!=expected_solution:
                        errors.append(f'{group}/{index}/{row_index}: non-self grouped solution')
            for i,t in enumerate(rows):
                if not t['scene_id'].startswith('train_') or t['image_index']!=i:
                    errors.append(f'{group}/{index}/{i}: split or image alignment')
                if abs(t['measurements']['simulated_time_s']-(t['decision_index']+1)*.16)>1e-8:
                    errors.append(f'{group}/{index}/{i}: physical time alignment')
                episode_ids.add(t['episode_id']);counts[t['action_label']]+=1
                fields[tuple((key,t['result'][key]) for key in sorted(t['result']))]+=1
                if t['result']['contact']:
                    contacts+=1;first_contacts.setdefault(t['episode_id'],t['decision_index'])
                if group in {'shuffled','causal_shuffled','hard_shuffled','solution_plus_causal_shuffled'}:
                    donor=update['donor_indices'][i]
                    if donor is not None:
                        d=rows[donor]
                        invalid=d['episode_id']==t['episode_id'] or d['safety_failure']
                        if group in {'causal_shuffled','solution_plus_causal_shuffled'}:
                            invalid=invalid or d['result']['success']!=t['result']['success'] or outcome_verdict(d['result'])==outcome_verdict(t['result'])
                            verdict_changed+=int(outcome_verdict(d['result'])!=outcome_verdict(t['result']))
                        elif group=='shuffled':
                            invalid=invalid or d['action_label']!=t['action_label'] or d['result']['success']!=t['result']['success']
                        else:
                            invalid=invalid or d['result']==t['result']
                        if invalid:
                            donor_errors.append([index,i,donor])
                        changed+=int(d['result']!=t['result'])
            # This file was produced by this experiment. weights_only avoids pickle objects.
            state=torch.load(batch/'checkpoint/training_state.pt',map_location='cpu',weights_only=True)
            optimizer_steps={int(v['step']) for v in state['optimizer']['state'].values()}
            if optimizer_steps!={index}:errors.append(f'{group}/{index}: optimizer step counters {optimizer_steps}')
            steps.append(sorted(optimizer_steps))
            with safe_open(batch/'checkpoint/adapter_model.safetensors',framework='pt',device='cpu') as f:
                keys=list(f.keys())
                if not all('.language_model.layers.' in k and 'lora_' in k for k in keys):
                    errors.append(f'{group}/{index}: non-language LoRA tensor')
                if not any(bool(f.get_tensor(k).abs().sum()>0) for k in keys if 'lora_B' in k):
                    errors.append(f'{group}/{index}: zero LoRA B after optimization')
            manifest.append({'update':index,'pre_action_images':str(batch/'pre_action_images.npz'),
                'transition_records':str(batch/'transitions.jsonl'),'split':'train',
                'policy_revision_before_update':f'round_{index-1:04d}',
                'feedback_schema_version':'result-v1','feedback_source':'simulator_measurements',
                'observation_time_rule':'decision_index * 0.16 seconds within episode',
                'result_time_rule':'(decision_index + 1) * 0.16 seconds within episode',
                'action_world_delta_rule':'measurements.requested_action * 0.05 meters, z target fixed at 0.025 m',
                'feedback_applies_to_decision':'decision_index',
                'image_ref_rule':'pre_action_images.npz images[image_index]',
                'teacher_donor_ref_rule':'update.json donor_indices[row] indexes this same transitions.jsonl',
                'image_sha256':hashlib.sha256((batch/'pre_action_images.npz').read_bytes()).hexdigest(),
                'transitions_sha256':hashlib.sha256((batch/'transitions.jsonl').read_bytes()).hexdigest()})
            total+=len(rows);valid+=sum(expected_valid)
        if donor_errors:errors.append(f'{group}: {len(donor_errors)} invalid feedback donors')
        summary=json.loads((run/'summary.json').read_text())
        evaluation=json.loads((run/'dev_final.json').read_text())
        if len(evaluation['episodes'])!=64 or any(not t['scene_id'].startswith('dev_') for t in evaluation['episodes']):
            errors.append(f'{group}: evaluation scene count/split')
        output[group]={'executed_transitions':total,'valid_transitions':valid,'episode_count':len(episode_ids),
            'contact_transitions':contacts,'episodes_with_contact':len(first_contacts),
            'first_contact_decision_mean':float(np.mean(list(first_contacts.values()))) if first_contacts else None,
            'action_counts':dict(counts),'feedback_tuple_counts':[{'tuple':[list(item) for item in k],'count':v} for k,v in fields.items()],
            'optimizer_steps_by_checkpoint':steps,'donor_errors':donor_errors,
            'changed_matched_feedback_count':changed if group in {'shuffled','causal_shuffled','hard_shuffled','solution_plus_causal_shuffled'} else None,
            'causal_shuffled_verdict_changed_count':verdict_changed if group in {'causal_shuffled','solution_plus_causal_shuffled'} else None,
            'train_episode_successes':sum(t['success'] for t in summary['training_episodes']),
            'dataset_manifest':manifest}
    first_equal=len(set(first_batches.values()))==1
    if not first_equal:errors.append('First online rollout batch differs across same-seed initial policies')
    return {'status':'passed' if not errors else 'failed','errors':errors,
        'first_rollout_batches_identical':first_equal,'first_rollout_sha256':first_batches,'groups':output,
        'limitations':['Checks cover this saved single-seed development experiment, not a multi-seed final result.',
                       'Run configuration pins source hashes. Raw records remain unchanged.',
                       'No claim of exact physics resume from optimizer checkpoints.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    report=audit(args.directory.resolve())
    (args.directory/'artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2))
    if report['errors']:raise SystemExit(1)
