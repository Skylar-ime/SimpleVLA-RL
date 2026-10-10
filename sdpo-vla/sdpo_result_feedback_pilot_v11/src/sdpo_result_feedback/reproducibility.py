from __future__ import annotations

from importlib import metadata
from pathlib import Path
import hashlib
import json
import platform
import shutil
import sys
import time


SNAPSHOT_FILES = ('run_pilot.py', 'run_chunk_pilot.py')
SNAPSHOT_DIRS = ('src', 'scripts', 'tests')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_paths(root: Path, protocol_path: Path) -> list[Path]:
    paths = [root / name for name in SNAPSHOT_FILES]
    paths.append(protocol_path)
    paths.extend(root.glob('protocol*.json'))
    for directory in SNAPSHOT_DIRS:
        paths.extend((root / directory).rglob('*.py'))
        if directory == 'scripts':
            paths.extend((root / directory).rglob('*.sh'))
    paths.extend(root.glob('artifacts/*_scenes*.json'))
    return sorted({path for path in paths if path.is_file()})


def _environment() -> dict:
    packages = {dist.metadata['Name']: dist.version
                for dist in metadata.distributions()
                if dist.metadata.get('Name')}
    try:
        import torch
        torch_info = {'version': torch.__version__, 'cuda': torch.version.cuda,
                      'cuda_available': torch.cuda.is_available()}
    except ImportError:
        torch_info = None
    return {
        'python': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'packages': dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        'torch': torch_info,
    }


def create_snapshot(root: Path, output: Path, protocol_path: Path,
                    model_id: str | None, arguments: dict) -> dict:
    snapshot = output / 'reproducibility'
    source_root = snapshot / 'source'
    source_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in _snapshot_paths(root, protocol_path):
        relative = path.relative_to(root)
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest.append({'path': str(relative), 'sha256': _sha256(path),
                         'bytes': path.stat().st_size})
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (snapshot / 'source_manifest.json').write_text(
        json.dumps({'files': manifest, 'sha256': manifest_hash}, indent=2) + '\n')
    (snapshot / 'environment.json').write_text(
        json.dumps(_environment(), indent=2) + '\n')
    model_path = Path(model_id) if model_id and Path(model_id).is_dir() else None
    model_files = []
    if model_path is not None:
        for path in sorted(model_path.rglob('*')):
            if path.is_file():
                model_files.append({'path': str(path.relative_to(model_path)),
                                    'bytes': path.stat().st_size})
    (snapshot / 'model_manifest.json').write_text(
        json.dumps({'model_id': model_id, 'files': model_files}, indent=2) + '\n')
    return {
        'format': 'sdpo-reproducibility-v1',
        'created_unix': time.time(),
        'source_manifest': 'reproducibility/source_manifest.json',
        'source_manifest_sha256': manifest_hash,
        'environment': 'reproducibility/environment.json',
        'model_manifest': 'reproducibility/model_manifest.json',
        'protocol': str(protocol_path.relative_to(root)),
        'arguments_sha256': hashlib.sha256(
            json.dumps(arguments, sort_keys=True).encode()).hexdigest(),
    }
