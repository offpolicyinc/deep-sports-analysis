"""Ask Claude (Opus) about images, with the answer held to a JSON schema and cached on disk.

Auth: ANTHROPIC_API_KEY. The cache key is a hash of model, effort, prompt,
schema and image bytes, so a rerun only pays for what changed; `<key>.usage.json` beside each answer keeps
the seconds and tokens of the call.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

MODEL = "claude-opus-5-5"
_client = None


def client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(max_retries=6)
    return _client


def _key(model: str, effort: str, prompt: str, schema: dict, images: list[Path]) -> str:
    h = hashlib.sha256()
    for part in (model, effort, prompt, json.dumps(schema, sort_keys=True)):
        h.update(part.encode())
        h.update(b"\0")
    for p in images:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:24]


def _api_schema(v):
    """The schema without array lengths other than 0 or 1, which the API refuses; `_lengths_ok` checks them."""
    if isinstance(v, dict):
        return {k: _api_schema(x) for k, x in v.items() if not (k in ("minItems", "maxItems") and x not in (0, 1))}
    if isinstance(v, list):
        return [_api_schema(x) for x in v]
    return v


def _lengths_ok(v, schema: dict) -> bool:
    if schema.get("type") == "array":
        return (schema.get("minItems", 0) <= len(v) <= schema.get("maxItems", len(v))
                and all(_lengths_ok(x, schema.get("items", {})) for x in v))
    if schema.get("type") == "object" and isinstance(v, dict):
        return all(_lengths_ok(v[k], sub) for k, sub in schema.get("properties", {}).items() if k in v)
    return True


def ask(prompt: str, images: list[Path], schema: dict, cache_dir: str | Path, model: str = MODEL,
        effort: str = "medium", retries: int = 2) -> tuple[dict, float]:
    """(answer matching `schema`, seconds spent; 0 when cached)."""
    cache = Path(cache_dir) / f"{_key(model, effort, prompt, schema, images)}.json"
    if cache.exists():
        return json.loads(cache.read_text()), 0.0
    cache.parent.mkdir(parents=True, exist_ok=True)
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    content = [{"type": "image", "source": {"type": "base64", "media_type": media[Path(p).suffix.lower()],
                                            "data": base64.b64encode(Path(p).read_bytes()).decode()}}
               for p in images] + [{"type": "text", "text": prompt}]
    t0, err = time.time(), None
    for _ in range(retries + 1):
        with client().messages.stream(
                model=model, max_tokens=32000, thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": _api_schema(schema)}, "effort": effort},
                messages=[{"role": "user", "content": content}]) as stream:
            r = stream.get_final_message()
        text = "".join(b.text for b in r.content if b.type == "text")
        try:
            answer = json.loads(text)
        except json.JSONDecodeError as e:
            err = e
            continue
        if not _lengths_ok(answer, schema):
            err = f"array lengths off the schema: {text[:500]}"
            continue
        sec = time.time() - t0
        cache.write_text(json.dumps(answer))
        cache.with_suffix(".usage.json").write_text(json.dumps(
            {"key": cache.stem, "model": model, "effort": effort, "sec": sec,
             "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}))
        return answer, sec
    raise RuntimeError(f"{model} gave no answer: {err}")
