from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


LINE_MARKER = "MDLINEBREAKTOKEN"
PROTECTED = re.compile(r"(`+[^`]*`+|https?://[^\s)>]+|<[^>]+>|\]\([^)]*\)|"
                       r"\*\*|__|~~|\\\||"
                       r"\b[\w./:@+-]+\.(?:py|sh|yaml|yml|json|toml|md|txt|csv|h5|pt|safetensors)\b)")
FENCE = re.compile(r"^\s*(```|~~~)")
FRONT_MATTER = re.compile(r"^---\s*$")
API = "https://translate.googleapis.com/translate_a/single"
MAX_CHARS = 2800
CACHE: dict[str, str] = {}
CACHE_LOCK = threading.Lock()


def translate_text(text: str) -> str:
    if not text.strip() or not re.search(r"[A-Za-z]", text):
        return text
    with CACHE_LOCK:
        cached = CACHE.get(text)
    if cached is not None:
        return cached
    query = urlencode({"client": "at", "sl": "auto", "tl": "zh-CN", "dt": "t", "q": text})
    request = Request(f"{API}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            translated = "".join(part[0] for part in payload[0] if part and part[0])
            with CACHE_LOCK:
                CACHE[text] = translated
            return translated
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("translation failed")


def protect_prose(text: str) -> tuple[str, dict[str, str]]:
    protected = {}

    def replace(match: re.Match[str]) -> str:
        token = f"MDTOKEN{len(protected)}X"
        protected[token] = match.group(0)
        return token

    return PROTECTED.sub(replace, text), protected


def translate_prose(text: str) -> str:
    prepared, protected = protect_prose(text)
    translated = translate_text(prepared)
    for token, original in protected.items():
        translated = translated.replace(token, original)
    return translated


def translate_lines(lines: list[str]) -> list[str]:
    endings = [line[len(line.rstrip("\r\n")):] for line in lines]
    bodies = [line.rstrip("\r\n") for line in lines]
    prefixes = []
    prepared = []
    for body in bodies:
        if re.match(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", body):
            prefixes.append("")
            prepared.append(body)
            continue
        prefix = ""
        match = re.match(r"^(\s{0,3}(?:#{1,6}\s+|(?:[-+*]|\d+[.)])\s+|(?:>\s*)+))", body)
        if match:
            prefix = match.group(1)
            body = body[len(prefix):]
        prefixes.append(prefix)
        prepared.append(body)
    translated = translate_prose(LINE_MARKER.join(prepared))
    result = translated.split(LINE_MARKER)
    if len(result) != len(bodies):
        result = [translate_prose(body) for body in prepared]
    if len(result) != len(bodies):
        raise RuntimeError(f"line count changed during translation: {len(lines)} -> {len(result)}")
    return [prefix + body + ending for prefix, body, ending in zip(prefixes, result, endings)]


def translate_lines_chunked(lines: list[str]) -> list[str]:
    result = []
    chunk = []
    size = 0

    def flush() -> None:
        nonlocal chunk, size
        if chunk:
            result.extend(translate_lines(chunk))
            chunk = []
            size = 0

    for line in lines:
        if chunk and size + len(line) > MAX_CHARS:
            flush()
        chunk.append(line)
        size += len(line)
    flush()
    return result


def translate_document(source: str) -> str:
    lines = source.splitlines(keepends=True)
    output = []
    prose = []
    in_fence = False
    front_matter = bool(lines and FRONT_MATTER.match(lines[0].rstrip("\r\n")))
    in_front_matter = front_matter
    for line in lines:
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped):]
        if in_front_matter:
            if prose:
                output.extend(translate_lines_chunked(prose))
                prose = []
            output.append(line)
            if len(output) > 1 and FRONT_MATTER.match(stripped):
                in_front_matter = False
            continue
        if FENCE.match(stripped):
            if prose:
                output.extend(translate_lines_chunked(prose))
                prose = []
            in_fence = not in_fence
            output.append(line)
            continue
        if in_fence:
            output.append(line)
        else:
            prose.append(line)
    if prose:
        output.extend(translate_lines_chunked(prose))
    return "".join(output)


def process(path: Path, write: bool) -> tuple[Path, bool]:
    original = path.read_text(encoding="utf-8")
    translated = translate_document(original)
    changed = translated != original
    if write and changed:
        temporary = path.with_suffix(path.suffix + ".zh.tmp")
        temporary.write_text(translated, encoding="utf-8")
        temporary.replace(path)
    return path, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--files", nargs="*", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = args.files or sorted(set(args.root.rglob("*.md")) | set(args.root.rglob("*.MD")))
    paths = [path if path.is_absolute() else args.root / path for path in paths]
    changed = []
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, path, not args.dry_run): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result, did_change = future.result()
                if did_change:
                    changed.append(result)
                print(f"{'changed' if did_change else 'unchanged'} {result}", flush=True)
            except Exception as error:
                failed.append((path, repr(error)))
                print(f"FAILED {path}: {error}", flush=True)
    print(f"processed={len(paths)} changed={len(changed)} failed={len(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
