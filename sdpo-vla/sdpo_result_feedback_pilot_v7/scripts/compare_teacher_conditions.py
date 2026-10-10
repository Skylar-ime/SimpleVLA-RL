from pathlib import Path
import argparse
import json

import numpy as np


CONDITIONS=('sdpo_result','sdpo_binary','sdpo_shuffled','sdpo_neutral','sdpo_causal')


def load_update(root,condition,update_index):
    path=root/condition/f'update_{update_index:03d}'
    update=json.loads((path/'update.json').read_text())
    context=json.loads((path/'teacher_context.json').read_text())
    return np.exp(np.asarray(update['teacher_log_probs'],dtype=np.float64)),context


def compare(root,update_index):
    distributions={}
    contexts={}
    for condition in CONDITIONS:
        distributions[condition],contexts[condition]=load_update(root,condition,update_index)
    result={}
    base=distributions['sdpo_result']
    base_solution=np.asarray([action is not None for action in contexts['sdpo_result']['solution_actions']])
    for condition in CONDITIONS[1:]:
        other=distributions[condition]
        tv=np.abs(base-other).sum(axis=1)/2
        changed=base.argmax(axis=1)!=other.argmax(axis=1)
        result[condition]={
            'mean_teacher_tv_all':float(tv.mean()),
            'mean_teacher_tv_solution_rows':float(tv[base_solution].mean()) if base_solution.any() else None,
            'mean_teacher_tv_feedback_rows':float(tv[~base_solution].mean()) if (~base_solution).any() else None,
            'teacher_argmax_changed_all':int(changed.sum()),
            'teacher_argmax_changed_feedback_rows':int(changed[~base_solution].sum()),
            'feedback_row_count':int((~base_solution).sum()),
        }
    output={'update':update_index,'reference':'sdpo_result','comparisons':result}
    (root/f'teacher_condition_comparison_update_{update_index:03d}.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--update',type=int,default=1)
    args=parser.parse_args()
    compare(args.directory.resolve(),args.update)
