import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    'sdpo_result', 'sdpo_advantage',
    'sdpo_advantage_shuffled', 'sdpo_advantage_reversed',
)


@pytest.mark.parametrize('seed_arguments', [
    [], ['12', '13', '14', '15', '16'], ['12 13 14 15 16'],
])
def test_multiseed_runner_runs_all_seeds(tmp_path, seed_arguments):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    fake_python = tmp_path / 'fake_python'
    fake_python.write_text(
        '#!/usr/bin/env python3\n'
        'import json\n'
        'from pathlib import Path\n'
        'import sys\n'
        'arguments = sys.argv[1:]\n'
        'if "--output" in arguments:\n'
        '    output = Path(arguments[arguments.index("--output") + 1])\n'
        '    output.mkdir()\n'
        '    seed = arguments[arguments.index("--seed") + 1]\n'
        '    (output / "completed.json").write_text(json.dumps({"seed": seed}))\n'
        'else:\n'
        '    output = Path(arguments[-1])\n'
        '    (output / "v12_audit.json").write_text("{}")\n'
    )
    fake_python.chmod(0o755)
    runner = scripts / 'run_v12_multiseed.sh'
    runner.write_text((ROOT / 'scripts/run_v12_multiseed.sh').read_text().replace(
        '/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python', str(fake_python)))
    output_root = tmp_path / 'runs'
    command = ['bash', str(runner), 'qwen3', str(output_root), *seed_arguments]
    subprocess.run(command, check=True, capture_output=True, text=True)
    for seed in range(12, 17):
        for condition in CONDITIONS:
            run = output_root / f'seed_{seed}' / condition
            assert json.loads((run / 'completed.json').read_text()) == {'seed': str(seed)}
            assert (run / 'v12_audit.json').is_file()
    assert len(list(output_root.glob('seed_*/*/completed.json'))) == 20
    resumed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert resumed.stdout.count('Skipping audited run:') == 20
