"""Summarize completed development runs without opening the final test split."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.stats import beta


def bootstrap_interval(values,rng,repetitions=10000):
    values=np.asarray(values,dtype=float)
    if not len(values): return None
    estimates=rng.choice(values,size=(repetitions,len(values)),replace=True).mean(1)
    return np.quantile(estimates,[.025,.975]).tolist()


def binomial_interval(successes, n, confidence=.95):
    """Two-sided exact Clopper-Pearson interval, including all-zero samples."""
    if n<1: return None
    alpha=1-confidence
    lower=0. if successes==0 else float(beta.ppf(alpha/2,successes,n-successes+1))
    upper=1. if successes==n else float(beta.ppf(1-alpha/2,successes+1,n-successes))
    return [lower,upper]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    groups=['frozen']+[name for name in ['result','result_interpreted','causal','causal_shuffled','binary','shuffled','hard_shuffled','neutral','no_feedback']
                       if (args.directory/name/'completed.json').exists()]
    if 'result' not in groups:
        raise SystemExit('Run requires a completed result group')
    reports={};summary={};rng=np.random.default_rng(4171)
    for group in groups:
        path=args.directory/group
        if not (path/'completed.json').exists():
            raise SystemExit(f'Run incomplete: {path}')
        report=json.loads((path/('evaluation.json' if group=='frozen' else 'dev_final.json')).read_text())
        reports[group]={e['scene_id']:e for e in report['episodes']}
        summary[group]=report['summary']
        summary[group]['exact95_success_interval']=binomial_interval(
            summary[group]['successes'],summary[group]['scenes_valid'])
        if group!='frozen':
            updates=json.loads((path/'summary.json').read_text())['updates']
            total=sum(u['executed'] for u in updates)
            summary[group]['train_interactions']=total
            summary[group]['training_contact_fraction']=sum(u['contact_count'] for u in updates)/total
            summary[group]['training_motion_fraction']=sum(u['motion_count'] for u in updates)/total
            summary[group]['feedback_changed_fraction']=sum(u['feedback_changed_count'] for u in updates)/total
            summary[group]['unmatched_donor_fraction']=sum(u['unmatched_donor_count'] for u in updates)/total
            summary[group]['mean_update_seconds']=float(np.mean([u['seconds'] for u in updates]))
            summary[group]['mean_rollout_seconds']=float(np.mean([u['rollout_seconds'] for u in updates]))
            for metric in ['teacher_oracle_accuracy','teacher_repeat_positive_accuracy','teacher_reject_negative_accuracy']:
                values=[u[metric] for u in updates if u.get(metric) is not None]
                summary[group][f'mean_{metric}']=float(np.mean(values)) if values else None
    paired={}
    comparators=[name for name in groups if name!='result']
    for comparator in comparators:
        names=sorted(set(reports['result'])&set(reports[comparator]))
        valid=[n for n in names if reports['result'][n]['status']!='invalid_reset' and reports[comparator][n]['status']!='invalid_reset']
        diff=[int(reports['result'][n]['success'])-int(reports[comparator][n]['success']) for n in valid]
        progress=[reports['result'][n]['progress_m']-reports[comparator][n]['progress_m'] for n in valid]
        successes_a=sum(int(reports['result'][n]['success']) for n in valid)
        successes_b=sum(int(reports[comparator][n]['success']) for n in valid)
        interval_a=binomial_interval(successes_a,len(valid),confidence=.975)
        interval_b=binomial_interval(successes_b,len(valid),confidence=.975)
        conservative=[interval_a[0]-interval_b[1],interval_a[1]-interval_b[0]] if valid else None
        paired[f'result_minus_{comparator}']={'paired_scenes':len(valid),
            'bootstrap_degenerate':bool(diff and len(set(diff))==1),
            'conservative95_difference_interval':conservative,
            'interval_note':'Bonferroni combination of 97.5% exact binomial marginal intervals; conservative and does not exploit pairing. Scene bootstrap degeneracy is not evidence of zero uncertainty.',
            'success_difference':float(np.mean(diff)) if diff else None,
            'paired_bootstrap95_success_difference':bootstrap_interval(diff,rng),
            'progress_difference_m':float(np.mean(progress)) if progress else None,
            'paired_bootstrap95_progress_difference_m':bootstrap_interval(progress,rng)}
    report={'status':'development_only_single_seed','groups':summary,'paired_comparisons':paired,
        'limitations':['No final-test evaluation.','One training seed; scene bootstrap does not capture training-seed uncertainty.',
                      'All-zero observed successes yield a degenerate bootstrap; use the nonzero exact or conservative intervals, not a false certainty claim.',
                      'Local feedback and success rates are not by themselves proof of the SDPO mechanism.']}
    path=args.directory/'comparison.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
