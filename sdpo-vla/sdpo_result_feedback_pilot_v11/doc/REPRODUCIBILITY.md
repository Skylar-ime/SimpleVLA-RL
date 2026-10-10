# Reproducibility

Every new `run_pilot.py` and `run_chunk_pilot.py` run stores a self-contained record under `run/reproducibility/`:

- `source/`: exact Python and shell sources, protocol files, tests, runners, and scene manifests available at run start.
- `source_manifest.json`: SHA-256 and byte size for every snapshot file.
- `environment.json`: Python, platform, PyTorch/CUDA, and installed package versions.
- `model_manifest.json`: model path and file names/sizes. Model weights remain outside the run directory.
- `run_config.json`: CLI arguments, training settings, protocol hash, and snapshot manifest hash.

Restore one run's exact source into an empty directory:

```bash
python scripts/restore_run_snapshot.py artifacts/<run> /tmp/sdpo-restore
```

Install the recorded environment separately, point the restored code at the recorded model path, then rerun the command recorded in `run_config.json`. Existing runs created before this mechanism only have source hashes and cannot be fully restored unless their source is recovered separately.
