from huggingface_hub import snapshot_download
from pathlib import Path

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
    local_dir=Path(__file__).resolve().parent / "models/Qwen2.5-VL-3B-Instruct",
    local_dir_use_symlinks=False,
)
print("DOWNLOAD_COMPLETE", flush=True)
