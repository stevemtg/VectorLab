"""Loopback-only model bridge. No hosted service or model downloads required."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.native import NativeModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama/models"))).resolve()
DATA = ROOT / "data/vectors"
DATA.mkdir(parents=True, exist_ok=True)
OLLAMA = "http://127.0.0.1:11434"
ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173", "https://vector-lab-steering.beige-cloud-9219.chatgpt.site"]
app = FastAPI(title="Vector Lab local model bridge", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-Vector-Lab"], allow_private_network=True)
lock = threading.Lock()
cancel = threading.Event()
native: NativeModel | None = None
connection: dict = {"engine": None, "model": None, "digest": None, "layers": None, "dimensions": None}


@app.middleware("http")
async def local_only(request: Request, call_next):
    if request.headers.get("host", "").split(":")[0] not in {"127.0.0.1", "localhost"}:
        return JSONResponse({"detail": "Local bridge requires a loopback host."}, 403)
    origin = request.headers.get("origin")
    if origin and origin not in ORIGINS:
        return JSONResponse({"detail": "This browser origin is not allowed."}, 403)
    if request.method not in {"GET", "OPTIONS"} and request.headers.get("x-vector-lab") != "1":
        return JSONResponse({"detail": "Missing workspace request header."}, 403)
    return await call_next(request)


def ollama(path: str, body=None, timeout=10):
    request = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except (OSError, urllib.error.URLError) as error:
        raise HTTPException(503, f"Ollama is unavailable at {OLLAMA}: {error}")


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
    parts = name.split("/")
    if "." in parts[0] and len(parts) >= 3:
        registry, namespace, tail = parts[0], "/".join(parts[1:-1]), parts[-1]
    else:
        registry, namespace, tail = "registry.ollama.ai", "/".join(parts[:-1]) or "library", parts[-1]
    model_name, _, tag = tail.partition(":")
    manifest = (MODEL_ROOT / "manifests" / registry / namespace / model_name / (tag or "latest")).resolve()
    if not manifest.is_relative_to(MODEL_ROOT / "manifests") or not manifest.is_file():
        raise HTTPException(404, "The local model manifest could not be resolved.")
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
            if data["model_digest"] == digest and (DATA / f'{data["id"]}.npy').is_file():
                result.append(data)
        except (ValueError, KeyError, OSError):
            continue
    return sorted(result, key=lambda v: v["created_at"])


@app.get("/api/models")
def models():
    return {"models": model_catalog(), "default_model": "qwen2.5-coder:1.5b"}


@app.get("/api/status")
def status():
    return {"connected": connection["model"] is not None, "busy": lock.locked(), **connection, "vectors": vectors_for(connection["digest"]), "native_available": (ROOT / ".runtime/llama/llama.dll").is_file()}


class Connect(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    engine: str = Field(pattern="^(native|ollama)$")


@app.post("/api/connect")
def connect(body: Connect):
    global native, connection
    acquire()
    try:
        model, path, digest = resolve_model(body.model)
        if body.engine == "native":
            if native and connection["digest"] == digest:
                connection = {**connection, "engine": "native", "model": body.model}
            else:
                if native:
                    native.close()
                    native = None
                connection = {"engine": None, "model": None, "digest": None, "layers": None, "dimensions": None}
                native = NativeModel(ROOT / ".runtime/llama", path)
                connection = {"engine": "native", "model": body.model, "digest": digest, "layers": native.layers, "dimensions": native.dimensions}
        else:
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


@app.post("/api/extract")
def extract(body: Extract):
    acquire()
    try:
        if connection["engine"] != "native" or native is None:
            raise HTTPException(409, "Connect a model with Native steering to extract real activation vectors.")
        name, positive, negative = body.name.strip(), body.positive.strip(), body.negative.strip()
        if not name or not positive or not negative or positive.casefold() == negative.casefold():
            raise HTTPException(422, "Provide a name and two different nonempty contrastive prompts.")
        existing = vectors_for(connection["digest"])
        if any(v["name"].casefold() == name.casefold() for v in existing):
            raise HTTPException(409, "A vector with this name already exists for this model.")
        if len(existing) >= 16:
            raise HTTPException(422, "Remove a vector before adding more than 16.")
        direction, stats = native.extract(positive, negative)
        identifier = str(uuid.uuid4())
        record = {"id": identifier, "name": name, "positive": positive, "negative": negative, "model_digest": connection["digest"], "model": connection["model"], "created_at": time.time(), **stats}
        np.save(DATA / f"{identifier}.npy", direction, allow_pickle=False)
        (DATA / f"{identifier}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
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
        valid = {v["id"] for v in vectors_for(connection["digest"])}
        for key, value in coefficients:
            if key not in valid or not math.isfinite(value):
                raise HTTPException(422, "Every coefficient must reference a vector extracted for this exact model.")
            directions[key] = np.load(DATA / f"{key}.npy", allow_pickle=False)
        if native and connection["engine"] == "native":
            native.cancel.clear()
            native.set_steering(directions, coefficients)
    except Exception:
        lock.release()
        raise

    def stream():
        try:
            yield json.dumps({"type": "start", "model": connection["model"], "engine": connection["engine"], "coefficients": [v.model_dump() for v in body.coefficients]}) + "\n"
            if connection["engine"] == "native" and native:
                for event in native.generate(messages):
                    yield json.dumps(event, allow_nan=False) + "\n"
            else:
                payload = {"model": connection["model"], "messages": messages, "stream": True, "options": {"num_predict": -1, "num_ctx": 2048}}
                if connection.get("thinking"):
                    payload["think"] = False
                with ollama("/api/chat", payload, timeout=None) as response:
                    for line in response:
                        if cancel.is_set():
                            yield json.dumps({"type": "done", "cancelled": True}) + "\n"
                            break
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
        except GeneratorExit:
            cancel.set()
            if native:
                native.cancel.set()
            raise
        except Exception as error:
            yield json.dumps({"type": "error", "error": str(error.detail if isinstance(error, HTTPException) else error), "cancelled": cancel.is_set()}) + "\n"
        finally:
            lock.release()

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
