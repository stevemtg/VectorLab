"""Loopback-only model bridge. No hosted service or model downloads required.

Runtime paths and origins are environment-driven so the same app runs from the
source tree (web workflow) and from an Electron-packaged install (desktop app):
  VECTOR_LAB_DATA_DIR     writable per-user vector storage (packaged builds set this)
  VECTOR_LAB_RUNTIME_DIR  directory holding the pinned b11146 llama.cpp libraries
  VECTOR_LAB_ORIGINS      browser origins (desktop token mode replaces the web defaults)
  VECTOR_LAB_TOKEN        desktop per-launch bearer token, required for every API request
  VECTOR_LAB_CONTROL_TOKEN main-process-only capability for folder import
  VECTOR_LAB_MIGRATE_FROM optional directory of vectors exported from an older source-tree install
  VECTOR_LAB_PORT         TCP port the packaged bridge executable listens on
"""
from __future__ import annotations

import json
import asyncio
import inspect
import math
import os
from pathlib import Path
import re
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager

import anyio
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.native import NativeModel, runtime_available

ROOT = Path(__file__).resolve().parents[1]
FROZEN = bool(getattr(sys, "frozen", False))
TOKEN = os.environ.get("VECTOR_LAB_TOKEN", "")
CONTROL_TOKEN = os.environ.get("VECTOR_LAB_CONTROL_TOKEN", "")
if FROZEN and (not TOKEN or not CONTROL_TOKEN):
    raise RuntimeError("The packaged bridge requires desktop authentication tokens.")
MODEL_ROOT = Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama/models"))).resolve()
# A packaged bridge must never write next to its (read-only, code-signed) bundle.
if FROZEN and not os.environ.get("VECTOR_LAB_DATA_DIR"):
    raise RuntimeError("The packaged bridge requires VECTOR_LAB_DATA_DIR to point at writable per-user storage.")
DATA = Path(os.environ.get("VECTOR_LAB_DATA_DIR", str(ROOT / "data" / "vectors"))).resolve()
DATA.mkdir(parents=True, exist_ok=True)
# Platform-appropriate native runtime directory (win-x64 ships in the installer today;
# macOS/Linux runtimes must be built at the same pinned b11146 ABI by the maintainer).
RUNTIME = Path(os.environ.get("VECTOR_LAB_RUNTIME_DIR", str(ROOT / ".runtime" / "llama"))).resolve()
OLLAMA = "http://127.0.0.1:11434"
ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173", "https://vector-lab-steering.beige-cloud-9219.chatgpt.site"]
if TOKEN:
    ORIGINS = []
for extra_origin in os.environ.get("VECTOR_LAB_ORIGINS", "").split(","):
    if extra_origin and extra_origin not in ORIGINS:
        ORIGINS.append(extra_origin)


def migrate_vectors(source):
    """Copy exported vectors from an older install into the active data dir.

    Called only for an explicitly configured migration or a main-process folder
    dialog. Validate pairs, never follow links outside that folder, preserve the
    exact checkpoint digest, and publish JSON last so readers see complete pairs.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise HTTPException(404, "The vector export folder could not be found.")
    imported = 0
    for file in sorted(source.glob("*.json")):
        try:
            array_file = file.with_suffix(".npy")
            if not file.resolve().is_relative_to(source) or not array_file.resolve().is_relative_to(source):
                continue
            if file.stat().st_size > 1024 * 1024 or array_file.stat().st_size > 128 * 1024 * 1024:
                continue
            record = json.loads(file.read_text(encoding="utf-8"))
            vector_id = str(uuid.UUID(str(record.get("id"))))
            if file.stem != vector_id or not valid_import_record(record):
                continue
            # mmap bounds allocations even for a malicious array header.
            array = np.load(array_file, allow_pickle=False, mmap_mode="r")
            if array.shape != (record["layers"], record["dimensions"]) or array.dtype.kind not in "fi" or not np.isfinite(array).all():
                continue
        except (ValueError, TypeError, AttributeError, KeyError, OSError, EOFError):
            continue
        if (DATA / f"{vector_id}.json").exists() or (DATA / f"{vector_id}.npy").exists():
            continue
        record["id"] = vector_id
        save_vector(record, array)
        imported += 1
    return imported


def valid_import_record(record):
    if not isinstance(record, dict) or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(record.get("model_digest", ""))):
        return False
    if not all(isinstance(record.get(key), str) and 0 < len(record[key]) <= limit for key, limit in
               [("name", 32), ("model", 200), ("positive", 4000), ("negative", 4000), ("method", 500)]):
        return False
    if not all(type(record.get(key)) is int and 0 < record[key] <= limit for key, limit in [("layers", 1024), ("dimensions", 65536)]):
        return False
    for key in ("created_at", "difference_norm"):
        if not isinstance(record.get(key), (float, int)) or not math.isfinite(record[key]) or record[key] < 0:
            return False
    norms = record.get("layer_norms")
    if not isinstance(norms, list) or len(norms) != record["layers"] or not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in norms):
        return False
    if record.get("mode", "mean") not in {"mean", "final_token"}:
        return False
    if "best_layer" in record and (type(record["best_layer"]) is not int or not 1 <= record["best_layer"] <= record["layers"]):
        return False
    # Reject nonfinite optional measurements as well, without inventing legacy ones.
    try:
        json.dumps(record, allow_nan=False)
    except (ValueError, TypeError):
        return False
    return True


def save_vector(record, array):
    identifier = str(uuid.UUID(record["id"]))
    staging = DATA / f".{identifier}-{uuid.uuid4()}"
    array_temp, json_temp = staging.with_suffix(".npy"), staging.with_suffix(".json")
    try:
        np.save(array_temp, array, allow_pickle=False)
        json_temp.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        array_temp.replace(DATA / f"{identifier}.npy")
        json_temp.replace(DATA / f"{identifier}.json")
    finally:
        array_temp.unlink(missing_ok=True)
        json_temp.unlink(missing_ok=True)


# First desktop launch can import vectors exported by an older source-tree install.
if os.environ.get("VECTOR_LAB_MIGRATE_FROM") and TOKEN:
    try:
        migrate_vectors(os.environ["VECTOR_LAB_MIGRATE_FROM"])
    except HTTPException as error:
        print(f"Vector Lab: vector migration from the previous install was skipped: {error.detail}", file=sys.stderr)

@asynccontextmanager
async def lifespan(_app):
    yield
    stop()
    def close_native():
        global native
        with lock:
            if native:
                native.close()
                native = None
    await anyio.to_thread.run_sync(close_native)


app = FastAPI(title="Vector Lab local model bridge", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-Vector-Lab", "Authorization"], allow_private_network=True)
lock = threading.Lock()
cancel = threading.Event()
native: NativeModel | None = None
ollama_task: tuple | None = None
connection: dict = {"engine": None, "model": None, "digest": None, "layers": None, "dimensions": None}


class ModelStreamingResponse(StreamingResponse):
    """Close the synchronous generator even when the browser aborts a stream."""

    def __init__(self, content, close_stream, cancel_stream, **kwargs):
        super().__init__(content, **kwargs)
        self.close_stream = close_stream
        self.cancel_stream = cancel_stream

    async def __call__(self, scope, receive, send):
        async def watch_disconnect():
            message = await receive()
            if message["type"] == "http.disconnect":
                self.cancel_stream()
            return message
        try:
            await super().__call__(scope, watch_disconnect, send)
        finally:
            self.cancel_stream()
            with anyio.CancelScope(shield=True):
                if inspect.iscoroutinefunction(self.close_stream):
                    await self.close_stream()
                else:
                    await anyio.to_thread.run_sync(self.close_stream)


@app.middleware("http")
async def local_only(request: Request, call_next):
    if not re.fullmatch(r"(?:127\.0\.0\.1|localhost)(?::[0-9]{1,5})?", request.headers.get("host", "")):
        return JSONResponse({"detail": "Local bridge requires a loopback host."}, 403)
    origin = request.headers.get("origin")
    if origin and origin not in ORIGINS:
        return JSONResponse({"detail": "This browser origin is not allowed."}, 403)
    if request.method not in {"GET", "OPTIONS"} and request.headers.get("x-vector-lab") != "1":
        return JSONResponse({"detail": "Missing workspace request header."}, 403)
    # OPTIONS is a browser preflight, carries no data, and has no bearer token.
    if TOKEN and request.method != "OPTIONS" and not secrets.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {TOKEN}".encode()):
        return JSONResponse({"detail": "Desktop bridge authentication required."}, 401)
    if request.url.path == "/api/vectors/import" and (not TOKEN or not CONTROL_TOKEN or not secrets.compare_digest(request.headers.get("x-vector-lab-control", "").encode(), CONTROL_TOKEN.encode())):
        return JSONResponse({"detail": "Vector import requires the desktop folder dialog."}, 403)
    return await call_next(request)


class LocalRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OLLAMA_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}), LocalRedirectHandler())


def ollama(path: str, body=None, timeout=10):
    request = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    try:
        return OLLAMA_HTTP.open(request, timeout=timeout)
    except (OSError, urllib.error.URLError) as error:
        raise HTTPException(503, f"Ollama is unavailable at {OLLAMA}. Start the installed Ollama service; install model weights separately before working offline. Details: {error}")


def model_catalog():
    with ollama("/api/tags") as response:
        result = json.load(response)
    return [{"name": m["name"], "size": m["size"], "family": m.get("details", {}).get("family", "unknown"), "parameter_size": m.get("details", {}).get("parameter_size", ""), "digest": m["digest"]} for m in result["models"]]


def resolve_model(name):
    # The request can select only an installed Ollama model, never an arbitrary file.
    models = model_catalog()
    model = next((m for m in models if m["name"] == name), None)
    if not model:
        raise HTTPException(404, "Select an installed model from the model list.")
    return resolve_model_files(model)


def resolve_model_files(model):
    name = model["name"]
    parts = name.split("/")
    if "." in parts[0] and len(parts) >= 3:
        registry, namespace, tail = parts[0], "/".join(parts[1:-1]), parts[-1]
    else:
        registry, namespace, tail = "registry.ollama.ai", "/".join(parts[:-1]) or "library", parts[-1]
    model_name, _, tag = tail.partition(":")
    manifest = (MODEL_ROOT / "manifests" / registry / namespace / model_name / (tag or "latest")).resolve()
    if not manifest.is_relative_to(MODEL_ROOT / "manifests") or not manifest.is_file():
        raise HTTPException(404, f"Native steering could not find this model's manifest under {MODEL_ROOT}. Set OLLAMA_MODELS to the same readable model directory used by Ollama, then restart Vector Lab. Standard Ollama chat does not require local file access.")
    data = json.loads(manifest.read_text())
    layers = [v for v in data["layers"] if v["mediaType"] == "application/vnd.ollama.image.model"]
    if len(layers) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", layers[0]["digest"]):
        raise HTTPException(422, "Native steering currently requires one GGUF model blob.")
    digest = layers[0]["digest"]
    path = MODEL_ROOT / "blobs" / digest.replace(":", "-")
    if not path.is_file():
        raise HTTPException(404, "The installed model blob is missing.")
    return model, path, digest


def acquire():
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "The model is busy. Wait for the current operation or stop generation.")


def vectors_for(digest):
    result = []
    for file in DATA.glob("*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            identifier = str(uuid.UUID(data["id"]))
            if identifier != file.stem or not isinstance(data.get("created_at"), (int, float)):
                continue
            if data["model_digest"] == digest and (DATA / f'{identifier}.npy').is_file():
                result.append(data)
        except (ValueError, TypeError, AttributeError, KeyError, OSError):
            continue
    return sorted(result, key=lambda v: v["created_at"])


@app.get("/api/models")
def models():
    return {"models": model_catalog(), "default_model": "qwen2.5-coder:1.5b"}


@app.get("/api/status")
def status():
    return {"connected": connection["model"] is not None, "busy": lock.locked(), **connection, "vectors": vectors_for(connection["digest"]), "native_available": runtime_available(RUNTIME)}


@app.get("/api/health")
def health():
    # No Ollama call, GGUF loading, or directory scanning on the readiness path.
    return {"ready": True, "instance": os.environ.get("VECTOR_LAB_INSTANCE_ID", "")}


class Connect(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    engine: str = Field(pattern="^(native|ollama)$")


@app.post("/api/connect")
def connect(body: Connect):
    global native, connection
    acquire()
    try:
        if body.engine == "native":
            model, path, digest = resolve_model(body.model)
            if native and connection["digest"] == digest:
                connection = {**connection, "engine": "native", "model": body.model}
            else:
                if native:
                    native.close()
                    native = None
                connection = {"engine": None, "model": None, "digest": None, "layers": None, "dimensions": None}
                native = NativeModel(RUNTIME, path)
                connection = {"engine": "native", "model": body.model, "digest": digest, "layers": native.layers, "dimensions": native.dimensions}
        else:
            # Standard Ollama chat also works for service installs whose weights
            # are not readable by this desktop user, or multi-blob model formats.
            model = next((m for m in model_catalog() if m["name"] == body.model), None)
            if not model:
                raise HTTPException(404, "Select an installed model from the model list.")
            digest = model["digest"]
            try:
                # Keep native vectors visible for the same checkpoint when its
                # manifest is readable; access to weights is still optional here.
                _, _, digest = resolve_model_files(model)
            except (HTTPException, OSError, ValueError, KeyError):
                pass
            with ollama("/api/show", {"model": body.model}) as response:
                metadata = json.load(response)
            if "completion" not in metadata.get("capabilities", ["completion"]):
                raise HTTPException(422, "This model does not support chat completion.")
            if native:
                native.close()
                native = None
            info = metadata.get("model_info", {})
            architecture = info.get("general.architecture", "")
            connection = {"engine": "ollama", "model": body.model, "digest": digest, "layers": info.get(f"{architecture}.block_count"), "dimensions": info.get(f"{architecture}.embedding_length"), "thinking": "thinking" in metadata.get("capabilities", [])}
        return {**connection, "connected": True, "vectors": vectors_for(digest), "device": "CPU" if body.engine == "native" else "Ollama managed"}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(422, str(error))
    finally:
        lock.release()


class Extract(BaseModel):
    name: str = Field(min_length=1, max_length=32)
    positive: str = Field(min_length=1, max_length=4000)
    negative: str = Field(min_length=1, max_length=4000)
    mode: str = Field(default="mean", pattern="^(mean|final_token)$")


@app.post("/api/extract")
def extract(body: Extract):
    acquire()
    try:
        if connection["engine"] != "native" or native is None:
            raise HTTPException(409, "Connect a model with Native steering to extract real activation vectors.")
        name, positive, negative = body.name.strip(), body.positive.strip(), body.negative.strip()
        positives = [line.strip() for line in positive.splitlines() if line.strip()]
        negatives = [line.strip() for line in negative.splitlines() if line.strip()]
        if not name or not positives or not negatives or [line.casefold() for line in positives] == [line.casefold() for line in negatives]:
            raise HTTPException(422, "Provide a name and two different nonempty contrastive prompts.")
        if len(positives) > 8 or len(negatives) > 8:
            raise HTTPException(422, "Each contrastive field holds at most 8 non-empty prompt lines.")
        existing = vectors_for(connection["digest"])
        if any(v["name"].casefold() == name.casefold() for v in existing):
            raise HTTPException(409, "A vector with this name already exists for this model.")
        if len(existing) >= 16:
            raise HTTPException(422, "Remove a vector before adding more than 16.")
        direction, stats = native.extract(positives, negatives, body.mode)
        identifier = str(uuid.uuid4())
        record = {"id": identifier, "name": name, "positive": positive, "negative": negative, "model_digest": connection["digest"], "model": connection["model"], "created_at": time.time(), **stats}
        save_vector(record, direction)
        return record
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(422, str(error))
    finally:
        if native:
            native.capture_enabled = False
        lock.release()


@app.delete("/api/vectors/{identifier}")
def delete_vector(identifier: str):
    try:
        uuid.UUID(identifier)
    except ValueError:
        raise HTTPException(422, "Invalid vector identifier.")
    acquire()
    try:
        match = next((v for v in vectors_for(connection["digest"]) if v["id"] == identifier), None)
        if not match:
            raise HTTPException(404, "Vector not found for the active model.")
        (DATA / f"{identifier}.json").unlink()
        (DATA / f"{identifier}.npy").unlink(missing_ok=True)
        return {"deleted": identifier}
    finally:
        lock.release()


class VectorImportRequest(BaseModel):
    source: str = Field(min_length=1, max_length=4096)


@app.post("/api/vectors/import", status_code=201)
def import_vectors(body: VectorImportRequest):
    if not TOKEN or not CONTROL_TOKEN:
        raise HTTPException(403, "Vector import is enabled only for desktop installs.")
    acquire()
    try:
        return {"imported": migrate_vectors(body.source)}
    finally:
        lock.release()


class Coefficient(BaseModel):
    id: str
    value: float = Field(ge=-5, le=5, allow_inf_nan=False)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str = Field(min_length=1, max_length=16000)


class Chat(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=60)
    coefficients: list[Coefficient] = Field(default_factory=list, max_length=16)


@app.post("/api/stop")
def stop():
    cancel.set()
    if native:
        native.cancel.set()
    active = ollama_task
    if active:
        loop, task = active
        if not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)
    return {"stop_requested": True}


@app.post("/api/chat")
def chat(body: Chat):
    acquire()
    cancel.clear()
    messages = [m.model_dump() for m in body.messages]
    try:
        if not connection["model"] or body.model != connection["model"]:
            raise HTTPException(409, "Connect the selected model before sending a message.")
        if connection["engine"] == "ollama" and body.coefficients:
            raise HTTPException(422, "Ollama's chat API cannot inject activation vectors. Select Native steering.")
        coefficients = [(v.id, v.value) for v in body.coefficients]
        if len(set(key for key, _ in coefficients)) != len(coefficients):
            raise HTTPException(422, "Duplicate vector IDs are not allowed.")
        directions = {}
        records = {v["id"]: v for v in vectors_for(connection["digest"])}
        for key, value in coefficients:
            if key not in records or not math.isfinite(value):
                raise HTTPException(422, "Every coefficient must reference a vector extracted for this exact model.")
            directions[key] = {**records[key], "direction": np.load(DATA / f"{key}.npy", allow_pickle=False)}
        if native and connection["engine"] == "native":
            native.cancel.clear()
            native.set_steering(directions, coefficients)
    except (ValueError, OSError) as error:
        lock.release()
        raise HTTPException(422, f"Unable to apply the saved vector: {error}")
    except Exception:
        lock.release()
        raise

    released = False

    def release():
        nonlocal released
        if not released:
            released = True
            lock.release()

    def stream():
        try:
            yield json.dumps({"type": "start", "model": connection["model"], "engine": connection["engine"], "coefficients": [v.model_dump() for v in body.coefficients]}) + "\n"
            if connection["engine"] == "native" and native:
                for event in native.generate(messages):
                    yield json.dumps(event, allow_nan=False) + "\n"
        except GeneratorExit:
            cancel.set()
            if native:
                native.cancel.set()
            raise
        except Exception as error:
            yield json.dumps({"type": "error", "error": str(error.detail if isinstance(error, HTTPException) else error), "cancelled": cancel.is_set()}) + "\n"
        finally:
            release()

    iterator = stream()

    def cancel_stream():
        if not released:
            stop()

    def close_stream():
        try:
            iterator.close()
        finally:
            # A disconnect before the first next() never enters stream's finally.
            release()

    if connection["engine"] == "ollama":
        async def ollama_stream():
            global ollama_task
            # Async I/O can be cancelled while waiting for headers or tokens,
            # including on Windows; no blocked synchronous reader is left behind.
            current = (asyncio.get_running_loop(), asyncio.current_task())
            ollama_task = current
            try:
                yield json.dumps({"type": "start", "model": connection["model"], "engine": "ollama", "coefficients": []}) + "\n"
                payload = {"model": connection["model"], "messages": messages, "stream": True, "options": {"num_predict": -1, "num_ctx": 2048}}
                if connection.get("thinking"):
                    payload["think"] = False
                # Environment proxies must never send local prompts off-device.
                async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(None, connect=10)) as client:
                    async with client.stream("POST", OLLAMA + "/api/chat", json=payload) as response:
                        if response.status_code != 200:
                            raise RuntimeError(f"Ollama rejected chat (HTTP {response.status_code}).")
                        async for line in response.aiter_lines():
                            if cancel.is_set():
                                yield json.dumps({"type": "done", "cancelled": True}) + "\n"
                                return
                            if not line.strip():
                                continue
                            event = json.loads(line)
                            if event.get("error"):
                                raise RuntimeError(event["error"])
                            text = event.get("message", {}).get("content", "")
                            if text:
                                yield json.dumps({"type": "token", "text": text}) + "\n"
                            if event.get("done"):
                                tokens, duration = event.get("eval_count", 0), event.get("eval_duration", 0) / 1e9
                                yield json.dumps({"type": "done", "tokens": tokens, "prompt_tokens": event.get("prompt_eval_count"), "seconds": event.get("total_duration", 0) / 1e9, "tokens_per_second": tokens / max(.001, duration), "layer_norms": None, "alignments": {}, "injection_norm": None}) + "\n"
            except asyncio.CancelledError:
                yield json.dumps({"type": "done", "cancelled": True}) + "\n"
            except Exception as error:
                yield json.dumps({"type": "error", "error": str(error), "cancelled": cancel.is_set()}) + "\n"
            finally:
                if ollama_task is current:
                    ollama_task = None
                release()

        async_iterator = ollama_stream()
        async def close_async_stream():
            try:
                await async_iterator.aclose()
            finally:
                release()
        return ModelStreamingResponse(async_iterator, close_async_stream, cancel_stream, media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    return ModelStreamingResponse(iterator, close_stream, cancel_stream, media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
