from pathlib import Path
import argparse
import json
import numpy as np


def read(path):
    return json.loads(path.read_text())


def audit(root: Path):
    errors=[]
    runs=([root/'sdpo_directional_residual',root/'sdpo_wrong_directional_residual']
          if (root/'sdpo_directional_residual').exists() else [root])
    for run in runs:
        if not (run/'completed.json').exists():
            errors.append(f'{run}: missing completed.json')
        paths=sorted(run.glob('update_*/update.json'))
        if not paths:
            errors.append(f'{run}: missing update files')
        for path in paths:
            update=read(path)
            if not update['subtract_noop_baseline']:
                errors.append(f'{path}: baseline flag false')
            privileged=np.asarray(update['teacher_privileged_log_probs'],dtype=float)
            baseline=np.asarray(update['teacher_noop_baseline_log_probs'],dtype=float)
            residual=np.asarray(update['teacher_residual_log_probs'],dtype=float)
            corrected=np.asarray(update['teacher_log_probs'],dtype=float)
            expected=(update['executed'],5)
            if privileged.shape != expected or baseline.shape != expected or residual.shape != expected:
                errors.append(f'{path}: teacher tensor shape mismatch')
                continue
            if corrected.shape != expected or not np.allclose(np.exp(corrected).sum(1),1.,atol=1e-6):
                errors.append(f'{path}: corrected target is not normalized')
            if not np.allclose(residual,privileged-baseline,atol=1e-6):
                errors.append(f'{path}: residual does not equal privileged-baseline')
    result={'status':'passed' if not errors else 'failed','errors':errors}
    (root/'v11_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    audit(parser.parse_args().directory.resolve())
