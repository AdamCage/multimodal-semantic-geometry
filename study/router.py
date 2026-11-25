"""RouterAI-only calls with content-addressed request and response records."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .acquire import save_json, sha

BASE = "https://routerai.ru/api/v1/"
LOCKS = [threading.Lock() for _ in range(64)]


class RequestError(RuntimeError):
    pass


def request(root: Path, endpoint: str, body: dict, *, context: str) -> dict:
    if endpoint not in {"embeddings", "chat/completions"}:
        raise ValueError("unsupported RouterAI endpoint")
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    identity = sha(endpoint.encode() + b"\0" + raw)
    with LOCKS[int(identity[:2], 16) % len(LOCKS)]:
        return _request(root, endpoint, body["model"], raw, identity, context)


def _request(root: Path, endpoint: str, model: str, raw: bytes, identity: str, context: str) -> dict:
    path = root / "api" / f"{identity}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf8"))
        if "response" not in record:
            raise RequestError(f"prior request has no usable response: {context} {identity[:12]}")
        return record["response"]
    key = os.environ["ROUTERAI_API_KEY"]
    # A pending write survives interruption and prevents silently rebilling an unknown outcome.
    record = dict(endpoint=endpoint, request_sha256=identity, model=model,
                  context=context, attempts=[])
    save_json(path, record)
    params = urllib.request.Request(BASE + endpoint, data=raw, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json",
    })
    for attempt in range(3):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(params, timeout=95) as response:
                data = json.load(response)
                status = response.status
            if not isinstance(data, dict):
                raise ValueError("API returned a non-object")
            record["attempts"].append(dict(status=status, seconds=round(time.monotonic() - started, 3)))
            record["response"] = data
            save_json(path, record)
            return data
        except urllib.error.HTTPError as exc:
            record["attempts"].append(dict(status=exc.code, seconds=round(time.monotonic() - started, 3)))
            # Error messages can echo submitted media or user content. Keep a digest only.
            record["error_body_sha256"] = sha(exc.read(1_000_000))
            save_json(path, record)
            if exc.code != 429 or attempt == 2:
                raise RequestError(f"RouterAI HTTP {exc.code}: {context} {identity[:12]}") from None
            retry = exc.headers.get("Retry-After", "")
            time.sleep(min(30, int(retry)) if retry.isdecimal() else 2 ** (attempt + 1))
        except (OSError, ValueError) as exc:
            record["attempts"].append(dict(error=type(exc).__name__, seconds=round(time.monotonic() - started, 3)))
            save_json(path, record)
            raise RequestError(f"RouterAI outcome unknown: {context} {identity[:12]}") from None
    raise RuntimeError("unreachable retry state")


def media_part(path: Path) -> dict:
    return media_bytes(path.read_bytes(), path.suffix)


def media_bytes(data: bytes, suffix: str) -> dict:
    payload = base64.b64encode(data).decode("ascii")
    if suffix == ".jpg":
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}}
    if suffix == ".mp4":
        return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{payload}"}}
    if suffix == ".wav":
        return {"type": "input_audio", "input_audio": {"data": payload, "format": "wav"}}
    raise ValueError(f"unsupported media extension {suffix}")


def embedding(root: Path, model: str, inputs: list, *, context: str, provider: str | None = None) -> tuple[list, str]:
    body = {"model": model, "input": inputs, "encoding_format": "float"}
    if provider is not None:
        body["provider"] = {"only": [provider], "allow_fallbacks": False}
    response = request(root, "embeddings", body, context=context)
    if provider is not None and str(response.get("provider", "")).casefold() != provider.casefold():
        raise ValueError(f"embedding response did not verify the pinned provider: {context}")
    items = sorted(response["data"], key=lambda row: row["index"])
    if [item["index"] for item in items] != list(range(len(inputs))) or not response.get("model"):
        raise ValueError(f"embedding response has invalid indices or model: {context}")
    return [item["embedding"] for item in items], response["model"]


def chat(root: Path, model: str, content: str | list, *, context: str,
         system: str = "", max_tokens: int = 250, temperature: float | None = 0) -> tuple[str, dict]:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content})
    params = dict(model=model, max_tokens=max_tokens, messages=messages)
    if temperature is not None:
        params["temperature"] = temperature
    response = request(root, "chat/completions", params, context=context)
    choice = response["choices"][0]
    result = choice["message"].get("content")
    if choice.get("finish_reason") != "stop" or not isinstance(result, str) or not result.strip():
        raise ValueError(f"no complete response: {context} ({choice.get('finish_reason')})")
    return result.strip(), {key: response.get(key) for key in ("id", "model", "usage")}


def parse_json(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.partition("\n")[2][:-3].strip()
    return json.loads(text)
