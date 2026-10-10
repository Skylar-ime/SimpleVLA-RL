from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


CONDITIONS = ('sdpo_result', 'sdpo_binary', 'sdpo_shuffled', 'sdpo_neutral', 'sdpo_causal')
RESULT_FIELDS = ('contact', 'object_motion', 'goal_progress', 'success', 'progress_m',
                 'object_motion_m', 'distance_before_m', 'distance_after_m')


def read_json(path):
    return json.loads(path.read_text())


def audit(root, conditions=CONDITIONS):
    errors=[]
    first_hashes={}
    coverage={}
    for condition in conditions:
        run=root/condition
        if not (run/'completed.json').exists():
            errors.append(f'{condition}: incomplete')
            continue
        config=read_json(run/'run_config.json')
        arguments=config['arguments']
        if arguments.get('scene_profile')!='directional_long':
            errors.append(f'{condition}: wrong scene profile')
        if not arguments.get('expect_feedback_only',False):
            errors.append(f'{condition}: feedback-only guard disabled')
        if int(arguments.get('grouped_candidates',0))!=8:
            errors.append(f'{condition}: wrong candidate count')
        updates=int(arguments['updates'])
        expected=int(arguments['transitions'])
        solution_count=0
        feedback_count=0
        record_counts=[]
        for update_index in range(1,updates+1):
            batch=run/f'update_{update_index:03d}'
            rows=[json.loads(line) for line in (batch/'transitions.jsonl').read_text().splitlines()]
            context=read_json(batch/'teacher_context.json')
            update=read_json(batch/'update.json')
            images=np.load(batch/'pre_action_images.npz')['images']
            if len(rows)!=expected or images.shape[0]!=expected:
                errors.append(f'{condition}/{update_index}: transition budget')
            if update['optimizer_steps']!=1 or update['executed']!=expected:
                errors.append(f'{condition}/{update_index}: optimizer budget')
            if context['solution_context_count']!=0:
                errors.append(f'{condition}/{update_index}: unexpected solution context')
            if any(row['result']['success'] for row in rows):
                errors.append(f'{condition}/{update_index}: unexpected task success')
            if any(context['solution_actions']):
                errors.append(f'{condition}/{update_index}: nonempty solution action')
            if context['include_outcome_feedback']!=[True]*expected:
                errors.append(f'{condition}/{update_index}: feedback mask')
            if context['valid_mask']!=[True]*expected:
                errors.append(f'{condition}/{update_index}: valid mask')
            groups=defaultdict(list)
            for index,row in enumerate(rows):
                groups[row['rollout_group_id']].append(index)
            if len(groups)!=expected//8 or any(len(members)!=8 for members in groups.values()):
                errors.append(f'{condition}/{update_index}: group structure')
            for members in groups.values():
                first=members[0]
                for index in members[1:]:
                    if not np.array_equal(images[index],images[first]):
                        errors.append(f'{condition}/{update_index}: RGB state mismatch')
                    if rows[index]['proprio']!=rows[first]['proprio']:
                        errors.append(f'{condition}/{update_index}: proprio state mismatch')
            probabilities=np.exp(np.asarray(update['teacher_log_probs'],dtype=np.float64))
            if probabilities.shape!=(expected,5) or not np.allclose(probabilities.sum(1),1.0,atol=1e-6):
                errors.append(f'{condition}/{update_index}: teacher distribution')
            solution_count+=context['solution_context_count']
            feedback_count+=sum(context['include_outcome_feedback'])
            record_counts.append(update.get('feedback_record_unique_count'))
        first_rows=[json.loads(line) for line in
                    (run/'update_001'/'transitions.jsonl').read_text().splitlines()]
        for row in first_rows:
            row['result']={field:row['result'].get(field) for field in RESULT_FIELDS}
        first_hashes[condition]=hashlib.sha256(
            json.dumps(first_rows,sort_keys=True).encode()).hexdigest()
        audit_config=read_json(run/'policy_audit.json')
        if audit_config.get('teacher')!='separate_lora_ema' or audit_config.get('teacher_update_rate')!=0.05:
            errors.append(f'{condition}: teacher configuration')
        if audit_config.get('distillation')!='js':
            errors.append(f'{condition}: non-JSD distillation')
        coverage[condition]={'solution_contexts':solution_count,'feedback_used_rows':feedback_count,
                             'feedback_record_unique_counts':record_counts}
    if len(first_hashes)==len(conditions) and len(set(first_hashes.values()))!=1:
        errors.append('first rollout batches differ across conditions')
    result={'status':'passed' if not errors else 'failed','errors':errors,
            'first_rollout_hashes':first_hashes,'coverage':coverage}
    (root/'feedback_only_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--conditions',nargs='+',choices=CONDITIONS+('sdpo_directional','sdpo_wrong_directional'),
                        default=CONDITIONS)
    args=parser.parse_args()
    audit(args.directory.resolve(),args.conditions)
