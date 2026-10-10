"""Fail closed unless frozen-teacher rich feedback beats its paired shuffled control."""
from pathlib import Path
import argparse
import json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('report',type=Path)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    groups=report['summary']['groups']
    causal=groups['causal']
    shuffled=groups['causal_shuffled']
    checks={
        'causal_expected_progress_positive':causal['expected_progress_difference_m']>0.,
        'causal_progress_exceeds_shuffled':causal['expected_progress_difference_m']>shuffled['expected_progress_difference_m'],
        'causal_repeat_accuracy_exceeds_shuffled':causal['teacher_repeat_previous_accuracy']>shuffled['teacher_repeat_previous_accuracy'],
        'causal_oracle_accuracy_exceeds_shuffled':causal['teacher_oracle_accuracy']>shuffled['teacher_oracle_accuracy'],
        'shuffled_all_rows_matched':shuffled['matching']['matched']==report['summary']['scenes'],
        'shuffled_all_outcomes_changed':shuffled['matching']['changed']==report['summary']['scenes'],
        'all_scenes_directionally_informative':report['summary']['informative_progress_scenes']==report['summary']['scenes'],
    }
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,
        'causal':causal,'causal_shuffled':shuffled}
    output=args.report.parent/'mechanism_gate.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='passed':
        raise SystemExit('Mechanism gate failed; refusing expensive training.')


if __name__=='__main__':
    main()
