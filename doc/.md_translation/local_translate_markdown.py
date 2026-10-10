from __future__ import annotations

import argparse
import importlib.util
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, MarianMTModel


SCRIPT = Path(__file__).with_name("translate_markdown.py")
SPEC = importlib.util.spec_from_file_location("translate_markdown", SCRIPT)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)


class LocalBatchTranslator:
    def __init__(self, model_path: Path, batch_size: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = MarianMTModel(AutoConfig.from_pretrained(model_path))
        state = torch.load(model_path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self.batch_size = batch_size
        self.requests: queue.Queue = queue.Queue()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def translate(self, text: str) -> str:
        if BASE.LINE_MARKER in text:
            pieces = text.split(BASE.LINE_MARKER)
            return BASE.LINE_MARKER.join(self._translate_many(pieces))
        return self._translate_many([text])[0]

    def _translate_many(self, texts: list[str]) -> list[str]:
        pending = []
        for text in texts:
            event = threading.Event()
            result = {}
            self.requests.put((text, event, result))
            pending.append((event, result))
        values = []
        for event, result in pending:
            event.wait()
            if "error" in result:
                raise result["error"]
            values.append(result["text"])
        return values

    def _worker(self) -> None:
        while True:
            first = self.requests.get()
            batch = [first]
            deadline = threading.Event()
            while len(batch) < self.batch_size:
                try:
                    batch.append(self.requests.get(timeout=0.01))
                except queue.Empty:
                    break
            try:
                texts = [item[0] for item in batch]
                encoded = self.tokenizer(texts, return_tensors="pt", padding=True,
                                         truncation=True, max_length=512).to(self.device)
                with torch.inference_mode():
                    outputs = self.model.generate(**encoded, max_new_tokens=512,
                                                  num_beams=1, do_sample=False)
                translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for (_, event, result), text in zip(batch, translated):
                    result["text"] = text
                    event.set()
            except Exception as error:
                for _, event, result in batch:
                    result["error"] = error
                    event.set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/tmp/opus-en-zh"))
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    translator = LocalBatchTranslator(args.model.resolve())
    BASE.translate_text = translator.translate
    paths = sorted(set(args.root.rglob("*.md")) | set(args.root.rglob("*.MD")))
    changed = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(BASE.process, path, True): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                _, did_change = future.result()
                changed += int(did_change)
                print(f"{'changed' if did_change else 'unchanged'} {path}", flush=True)
            except Exception as error:
                failed.append(path)
                print(f"FAILED {path}: {error}", flush=True)
    print(f"processed={len(paths)} changed={changed} failed={len(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
