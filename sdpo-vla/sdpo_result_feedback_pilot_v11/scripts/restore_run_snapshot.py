from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import shutil


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Restore exact source snapshot from one run')
    parser.add_argument('run', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    source = args.run / 'reproducibility/source'
    manifest_path = args.run / 'reproducibility/source_manifest.json'
    if not source.is_dir() or not manifest_path.is_file():
        raise SystemExit('Run has no reproducibility source snapshot')
    if args.destination.exists() and any(args.destination.iterdir()):
        raise SystemExit(f'Refusing to overwrite non-empty destination: {args.destination}')
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    for row in manifest['files']:
        path = source / row['path']
        if digest(path) != row['sha256']:
            raise SystemExit(f'Snapshot hash mismatch: {path}')
        target = args.destination / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(f"Restored {len(manifest['files'])} files to {args.destination}")


if __name__ == '__main__':
    main()

