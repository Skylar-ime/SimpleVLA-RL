from __future__ import annotations

import argparse
import hashlib
import importlib.util
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


SCRIPT = Path(__file__).with_name("translate_markdown.py")
SPEC = importlib.util.spec_from_file_location("translate_markdown", SCRIPT)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)


PROMPT = (
    "Translate this plain Markdown text into Simplified Chinese. Output only "
    "the translation. Do not add explanations, headings, examples, or Markdown.\n\n"
)
MARKUP_TOKEN = re.compile(r"(`+[^`]*`+|https?://[^\s)>]+|<[^>]+>|"
                          r"\]\([^)]*\)|[\[\]{}|()]|\*+|_+|~~)")


class QwenBatchTranslator:
    def __init__(self, model_path: Path, batch_size: int = 16):
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda"},
            local_files_only=True, attn_implementation="sdpa")
        self.model.eval()
        self.batch_size = batch_size
        self.requests: queue.Queue = queue.Queue()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def translate(self, text: str) -> str:
        return self._translate_piece(text)

    def _translate_piece(self, text: str) -> str:
        if not text.strip() or not any(char.isascii() and char.isalpha() for char in text):
            return text
        if re.fullmatch(r"\s*(?:<[^>]+>\s*)+", text):
            return text
        if "<" in text and ">" in text:
            parts = re.split(r"(<[^>]+>)", text)
            translatable = [part for part in parts
                            if part and not re.fullmatch(r"<[^>]+>", part)
                            and any(char.isascii() and char.isalpha() for char in part)]
            values = iter(self._translate_many(translatable))
            return "".join(part if not part or re.fullmatch(r"<[^>]+>", part)
                           else next(values) if part in translatable else part
                           for part in parts)
        return self._translate_many([text])[0]

    def _translate_many(self, texts: list[str]) -> list[str]:
        pending = []
        for text in texts:
            if not text.strip() or not any(char.isascii() and char.isalpha() for char in text):
                pending.append((None, {"text": text}))
                continue
            event = threading.Event()
            result = {}
            self.requests.put((text, event, result))
            pending.append((event, result))
        values = []
        for event, result in pending:
            if event is not None:
                event.wait()
            if "error" in result:
                raise result["error"]
            values.append(result["text"])
        return values

    def _worker(self) -> None:
        while True:
            first = self.requests.get()
            batch = [first]
            while len(batch) < self.batch_size:
                try:
                    batch.append(self.requests.get(timeout=0.01))
                except queue.Empty:
                    break
            try:
                messages = []
                for text, _, _ in batch:
                    messages.append([{"role": "user", "content": [{"type": "text", "text": PROMPT + text}]}])
                prompts = [self.processor.apply_chat_template(message, tokenize=False,
                                                               add_generation_prompt=True)
                           for message in messages]
                encoded = self.processor(text=prompts, return_tensors="pt", padding=True).to("cuda")
                with torch.inference_mode():
                    outputs = self.model.generate(**encoded, max_new_tokens=128, do_sample=False)
                start = encoded["input_ids"].shape[1]
                translated = self.processor.batch_decode(outputs[:, start:], skip_special_tokens=True)
                for (source, event, result), text in zip(batch, translated):
                    text = text.strip()
                    text = text.replace("当然，请提供您希望翻译的Markdown文本。", "")
                    text = re.sub(
                        r"# Markdown to Simplified Chinese Translation\n+## (?:Preserve Markdown syntax, code, URLs, identifiers, numbers, and special tokens|保留Markdown语法、代码、URLs、标识符、数字和特殊标记)\n+(?:Output translation only, with no explanation\.|输出翻译，不提供解释。)\n*",
                        "", text)
                    if ("Markdown Prose" in text or "Markdown Syntax" in text
                            or "Markdown语法" in text or "Markdown 语法" in text):
                        text = source
                    if "\n" in text:
                        text = source
                    result["text"] = text.strip()
                    event.set()
            except Exception as error:
                for _, event, result in batch:
                    result["error"] = error
                    event.set()


def translate_safe_document(source: str, translator: QwenBatchTranslator) -> str:
    lines = source.splitlines(keepends=True)
    output = []
    in_fence = False
    in_front_matter = bool(lines and BASE.FRONT_MATTER.match(lines[0].rstrip("\r\n")))
    for line in lines:
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped):]
        if in_front_matter:
            output.append(line)
            if len(output) > 1 and BASE.FRONT_MATTER.match(stripped):
                in_front_matter = False
            continue
        if BASE.FENCE.match(stripped):
            in_fence = not in_fence
            output.append(line)
            continue
        if in_fence:
            output.append(line)
            continue
        if not re.search(r"[A-Za-z]", stripped):
            output.append(line)
            continue
        match = re.match(r"^(\s{0,3}(?:#{1,6}\s+|(?:[-+*]|\d+[.)])\s+))", stripped)
        prefix = match.group(1) if match else ""
        body = stripped[len(prefix):]
        fragments = MARKUP_TOKEN.split(body)
        translated_fragments = []
        for index, fragment in enumerate(fragments):
            if index % 2 or not re.search(r"[A-Za-z]", fragment):
                translated_fragments.append(fragment)
            else:
                translated_fragments.append(translator.translate(fragment))
        translated = "".join(translated_fragments).strip()
        if not translated or "Markdown to Simplified" in translated:
            translated = body
        output.append(prefix + translated + newline)
    return "".join(output)


def process_safe(path: Path, translator: QwenBatchTranslator) -> tuple[Path, bool]:
    original = path.read_text(encoding="utf-8")
    translated = translate_safe_document(original, translator)
    changed = translated != original
    if changed:
        temporary = path.with_suffix(path.suffix + ".zh.tmp")
        temporary.write_text(translated, encoding="utf-8")
        temporary.replace(path)
    return path, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", type=Path,
                        default=Path("sdpo_result_feedback_pilot/models/Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    translator = QwenBatchTranslator(args.model.resolve())
    paths = sorted(set(args.root.rglob("*.md")) | set(args.root.rglob("*.MD")))
    changed = 0
    failed = []
    groups = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups.setdefault(digest, []).append(path)
    representatives = [(members[0], digest) for digest, members in groups.items()]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_safe, path, translator): (path, digest)
                   for path, digest in representatives}
        for future in as_completed(futures):
            path, digest = futures[future]
            try:
                _, did_change = future.result()
                changed += int(did_change)
                print(f"{'changed' if did_change else 'unchanged'} {path}", flush=True)
                translated = path.read_bytes()
                for duplicate in groups[digest]:
                    if duplicate == path:
                        continue
                    original = duplicate.read_bytes()
                    if original != translated:
                        temporary = duplicate.with_suffix(duplicate.suffix + ".zh.tmp")
                        temporary.write_bytes(translated)
                        temporary.replace(duplicate)
                        changed += 1
                        print(f"changed {duplicate}", flush=True)
            except Exception as error:
                failed.extend(groups[digest])
                print(f"FAILED {path}: {error}", flush=True)
    print(f"processed={len(paths)} unique={len(representatives)} changed={changed} failed={len(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
