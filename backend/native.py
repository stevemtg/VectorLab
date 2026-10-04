"""Pinned llama.cpp b11146 CPU ABI; real residual capture and additive steering.

The structures below mirror b11146/include/llama.h. Do not substitute an
arbitrary llama.dll: native ABI changes require updating these bindings.
"""
from __future__ import annotations

import codecs
import ctypes as C
import os
from pathlib import Path
import re
import threading
import time

import numpy as np

P = C.c_void_p
I = C.c_int32
U = C.c_uint32
F = C.c_float
B = C.c_bool
S = C.c_size_t
FP = C.POINTER(F)
IP = C.POINTER(I)
EvalCallback = C.CFUNCTYPE(B, P, B, P)
AbortCallback = C.CFUNCTYPE(B, P)
LogCallback = C.CFUNCTYPE(None, I, C.c_char_p, P)


class ModelParams(C.Structure):
    _fields_ = [("devices", P), ("tensor_buft_overrides", P), ("n_gpu_layers", I), ("split_mode", I), ("load_mode", I), ("lazy_mode", I), ("main_gpu", I), ("tensor_split", P), ("progress_callback", P), ("progress_callback_user_data", P), ("kv_overrides", P)] + [(n, B) for n in ["vocab_only", "check_tensors", "use_extra_bufts", "no_host", "no_alloc", "load_mtp"]]


class ContextParams(C.Structure):
    _fields_ = [(n, U) for n in ["n_ctx", "n_batch", "n_ubatch", "n_seq_max", "n_rs_seq", "n_outputs_max", "n_outputs_max_per_seq"]] + [(n, I) for n in ["n_threads", "n_threads_batch", "ctx_type", "rope_scaling_type", "pooling_type", "attention_type", "flash_attn_type"]] + [(n, F) for n in ["rope_freq_base", "rope_freq_scale", "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow"]] + [("yarn_orig_ctx", U), ("defrag_thold", F), ("cb_eval", EvalCallback), ("cb_eval_user_data", P), ("type_k", I), ("type_v", I), ("abort_callback", AbortCallback), ("abort_callback_data", P)] + [(n, B) for n in ["embeddings", "offload_kqv", "no_perf", "op_offload", "swa_full", "kv_unified"]] + [("samplers", P), ("n_samplers", S), ("ctx_other", P)]


class Batch(C.Structure):
    _fields_ = [("n_tokens", I), ("token", IP), ("embd", FP), ("pos", IP), ("n_seq_id", IP), ("seq_id", C.POINTER(IP)), ("logits", C.POINTER(C.c_int8))]


class ChatMessage(C.Structure):
    _fields_ = [("role", C.c_char_p), ("content", C.c_char_p)]


class NativeModel:
    def __init__(self, runtime: Path, model_path: Path, emit=lambda event: None):
        if os.name != "nt":
            raise RuntimeError("The installed native runtime is the pinned Windows x64 CPU build.")
        self.dll_directory = os.add_dll_directory(str(runtime))
        self.ggml_base = C.CDLL(str(runtime / "ggml-base.dll"))
        self.ggml = C.CDLL(str(runtime / "ggml.dll"))
        self.lib = C.CDLL(str(runtime / "llama.dll"))
        self.emit = emit
        self.model = self.ctx = None
        self.cancel = threading.Event()
        self.capture: dict[int, np.ndarray] = {}
        self.capture_enabled = False
        self.capture_error = None
        self.last_metrics = {}
        self.directions: dict[str, np.ndarray] = {}
        self.active: list[tuple[str, float]] = []
        self._bind()
        self.log_callback = LogCallback(lambda level, message, _: None)
        self.lib.llama_log_set(self.log_callback, None)
        self.ggml.ggml_backend_load_all_from_path(str(runtime).encode())
        self.lib.llama_backend_init()
        params = self.lib.llama_model_default_params()
        params.n_gpu_layers = 0
        params.load_mode = 1
        self.model = self.lib.llama_model_load_from_file(str(model_path).encode(), params)
        if not self.model:
            raise RuntimeError("llama.cpp could not load this GGUF. Choose a supported local text model.")
        self.layers = self.lib.llama_model_n_layer(self.model)
        self.dimensions = self.lib.llama_model_n_embd(self.model)
        self.vocab = self.lib.llama_model_get_vocab(self.model)
        self.eval_callback = EvalCallback(self._capture_tensor)
        self.abort_callback = AbortCallback(lambda _: self.cancel.is_set())
        ctx = self.lib.llama_context_default_params()
        ctx.n_ctx = 2048
        ctx.n_batch = ctx.n_ubatch = 512
        ctx.n_threads = ctx.n_threads_batch = min(8, max(1, (os.cpu_count() or 4) // 2))
        ctx.cb_eval = self.eval_callback
        ctx.abort_callback = self.abort_callback
        ctx.offload_kqv = False
        self.ctx = self.lib.llama_init_from_model(self.model, ctx)
        if not self.ctx:
            self.close()
            raise RuntimeError("Unable to allocate a native model context.")

    def _bind(self):
        def bind(lib, name, result, args):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = result, args
        for name, result, args in [
            ("llama_log_set", None, [LogCallback, P]),
            ("llama_backend_init", None, []),
            ("llama_model_default_params", ModelParams, []),
            ("llama_context_default_params", ContextParams, []),
            ("llama_model_load_from_file", P, [C.c_char_p, ModelParams]),
            ("llama_model_free", None, [P]), ("llama_free", None, [P]),
            ("llama_init_from_model", P, [P, ContextParams]),
            ("llama_model_n_layer", I, [P]), ("llama_model_n_embd", I, [P]),
            ("llama_model_get_vocab", P, [P]), ("llama_get_memory", P, [P]),
            ("llama_memory_clear", None, [P, B]),
            ("llama_tokenize", I, [P, C.c_char_p, I, IP, I, B, B]),
            ("llama_batch_get_one", Batch, [IP, I]),
            ("llama_decode", I, [P, Batch]),
            ("llama_synchronize", None, [P]),
            ("llama_set_adapter_cvec", I, [P, FP, S, I, I, I]),
            ("llama_sampler_init_greedy", P, []),
            ("llama_sampler_sample", I, [P, P, I]),
            ("llama_sampler_free", None, [P]),
            ("llama_vocab_is_eog", B, [P, I]),
            ("llama_token_to_piece", I, [P, I, P, I, I, B]),
            ("llama_model_chat_template", C.c_char_p, [P, C.c_char_p]),
            ("llama_chat_apply_template", I, [C.c_char_p, C.POINTER(ChatMessage), S, B, P, I]),
        ]:
            bind(self.lib, name, result, args)
        bind(self.ggml, "ggml_backend_load_all_from_path", None, [C.c_char_p])
        bind(self.ggml_base, "ggml_get_name", C.c_char_p, [P])
        bind(self.ggml_base, "ggml_nelements", C.c_int64, [P])
        bind(self.ggml_base, "ggml_nbytes", S, [P])
        self.tensor_get = None
        for lib in [self.ggml_base, self.ggml]:
            if hasattr(lib, "ggml_backend_tensor_get"):
                bind(lib, "ggml_backend_tensor_get", None, [P, P, S, S])
                self.tensor_get = lib.ggml_backend_tensor_get
                break
        if self.tensor_get is None:
            raise RuntimeError("Runtime does not expose tensor capture.")

    def _capture_tensor(self, tensor, ask, _):
        try:
            name = self.ggml_base.ggml_get_name(tensor).decode()
            match = re.fullmatch(r"l_out-(\d+)", name)
            wanted = bool(match) and self.capture_enabled
            if ask:
                return wanted
            if not wanted:
                return True
            count = self.ggml_base.ggml_nelements(tensor)
            nbytes = self.ggml_base.ggml_nbytes(tensor)
            if count <= 0 or count % self.dimensions or nbytes != count * 4:
                raise RuntimeError("Unexpected residual tensor layout; capture refused.")
            data = np.empty(count, dtype=np.float32)
            self.tensor_get(tensor, data.ctypes.data, 0, nbytes)
            self.capture[int(match[1])] = data.reshape(-1, self.dimensions).mean(axis=0)
            return not self.cancel.is_set()
        except Exception as error:
            self.capture_error = str(error)
            return False

    def clear(self):
        self.lib.llama_memory_clear(self.lib.llama_get_memory(self.ctx), True)
        self.capture.clear()
        self.capture_error = None

    def tokenize(self, text):
        data = text.encode("utf-8")
        capacity = len(data) + 16
        tokens = (I * capacity)()
        count = self.lib.llama_tokenize(self.vocab, data, len(data), tokens, capacity, True, True)
        if count < 0:
            raise ValueError("Tokenization failed.")
        return list(tokens[:count])

    def decode(self, tokens):
        array = (I * len(tokens))(*tokens)
        result = self.lib.llama_decode(self.ctx, self.lib.llama_batch_get_one(array, len(tokens)))
        self.lib.llama_synchronize(self.ctx)
        if self.capture_error:
            raise RuntimeError(self.capture_error)
        if result:
            raise RuntimeError("Generation cancelled." if self.cancel.is_set() else f"Native evaluation failed with code {result}.")

    def set_steering(self, directions: dict[str, np.ndarray], coefficients: list[tuple[str, float]]):
        mixed = np.zeros((self.layers - 1, self.dimensions), dtype=np.float32)
        for key, value in coefficients:
            direction = directions[key]
            if direction.shape != mixed.shape:
                raise ValueError("Vector dimensions do not match this model.")
            mixed += np.float32(value) * direction
        self.mixed = np.ascontiguousarray(mixed)
        result = self.lib.llama_set_adapter_cvec(self.ctx, self.mixed.ctypes.data_as(FP), self.mixed.size, self.dimensions, 1, self.layers - 1)
        if result:
            raise RuntimeError("Native runtime rejected the control vector.")
        self.active, self.directions = coefficients, directions

    def extract(self, positive: str, negative: str):
        self.cancel.clear()
        self.set_steering({}, [])
        self.capture_enabled = True
        captured = []
        for text in [positive, negative]:
            self.clear()
            tokens = self.tokenize(text)
            if not tokens or len(tokens) > 512:
                raise ValueError("Each contrastive prompt must contain between 1 and 512 tokens.")
            self.decode(tokens)
            captured.append({i: v.copy() for i, v in self.capture.items()})
        common = sorted(set(captured[0]) & set(captured[1]) & set(range(1, self.layers)))
        if len(common) < self.layers - 2:
            raise RuntimeError(f"This model exposes only {len(common)} usable residual layers. Native extraction is unsupported for its graph.")
        direction = np.zeros((self.layers - 1, self.dimensions), dtype=np.float32)
        norms = []
        for layer in common:
            difference = captured[0][layer] - captured[1][layer]
            norm = float(np.linalg.norm(difference))
            norms.append(norm)
            if norm > 1e-8:
                direction[layer - 1] = difference / norm
        if not np.any(direction):
            raise ValueError("These prompts produce no measurable contrast. Use a different pair.")
        self.capture_enabled = False
        return direction, {"layers": len(common), "dimensions": self.dimensions, "difference_norm": float(np.mean(norms)), "layer_norms": norms, "method": "mean positive residual − mean neutral residual; unit-normalized per layer"}

    def telemetry(self):
        norms = [float(np.linalg.norm(self.capture[i])) if i in self.capture else None for i in range(self.layers)]
        alignments = {}
        for key, _ in self.active:
            direction = self.directions[key]
            values = []
            for layer in range(1, self.layers):
                residual = self.capture.get(layer)
                if residual is not None:
                    denominator = float(np.linalg.norm(residual) * np.linalg.norm(direction[layer - 1]))
                    if denominator > 1e-8:
                        values.append(float(np.dot(residual, direction[layer - 1]) / denominator))
            alignments[key] = float(np.mean(values)) if values else None
        return {"layer_norms": norms, "alignments": alignments, "injection_norm": float(np.linalg.norm(self.mixed)), "layers": self.layers, "dimensions": self.dimensions}

    def format_chat(self, messages):
        template = self.lib.llama_model_chat_template(self.model, None)
        if not template:
            raise ValueError("This checkpoint has no chat template. Use an instruct/chat model.")
        encoded = [(m["role"].encode(), m["content"].encode()) for m in messages]
        chat = (ChatMessage * len(encoded))(*(ChatMessage(*m) for m in encoded))
        buffer = C.create_string_buffer(100000)
        count = self.lib.llama_chat_apply_template(template, chat, len(chat), True, buffer, len(buffer))
        if count < 0 or count >= len(buffer):
            raise ValueError("This model's chat template is not supported by the native adapter. Use Ollama chat for this model.")
        return buffer.raw[:count].decode("utf-8")

    def generate(self, messages):
        self.clear()
        self.capture_enabled = True
        tokens = self.tokenize(self.format_chat(messages))
        if not tokens or len(tokens) > 2048:
            raise ValueError("The conversation exceeds the 2048-token native context. Clear chat or shorten the prompt.")
        started = time.monotonic()
        for start in range(0, len(tokens), 512):
            self.decode(tokens[start:start + 512])
        prompt_seconds = time.monotonic() - started
        sampler = self.lib.llama_sampler_init_greedy()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        generated = 0
        generation_started = time.monotonic()
        context_tokens = tokens.copy()
        try:
            while True:
                if self.cancel.is_set():
                    break
                token = self.lib.llama_sampler_sample(sampler, self.ctx, -1)
                if self.lib.llama_vocab_is_eog(self.vocab, token):
                    break
                piece = C.create_string_buffer(512)
                size = self.lib.llama_token_to_piece(self.vocab, token, piece, len(piece), 0, False)
                if size < 0:
                    piece = C.create_string_buffer(-size)
                    size = self.lib.llama_token_to_piece(self.vocab, token, piece, len(piece), 0, False)
                generated += 1
                delta = decoder.decode(piece.raw[:size])
                yield {"type": "token", "text": delta, "tokens": generated}
                if len(context_tokens) == 2048:
                    context_tokens = context_tokens[-1024:]
                    self.clear()
                    for start in range(0, len(context_tokens), 512):
                        self.decode(context_tokens[start:start + 512])
                context_tokens.append(token)
                self.decode([token])
                if generated == 1 or generated % 8 == 0:
                    yield {"type": "metrics", **self.telemetry(), "tokens": generated, "tokens_per_second": generated / max(.001, time.monotonic() - generation_started)}
            tail = decoder.decode(b"", final=True)
            if tail:
                yield {"type": "token", "text": tail, "tokens": generated}
            yield {"type": "done", **self.telemetry(), "tokens": generated, "prompt_tokens": len(tokens), "prompt_seconds": prompt_seconds, "seconds": time.monotonic() - started, "tokens_per_second": generated / max(.001, time.monotonic() - generation_started), "cancelled": self.cancel.is_set()}
        finally:
            self.lib.llama_sampler_free(sampler)
            self.capture_enabled = False

    def close(self):
        if self.ctx:
            self.lib.llama_free(self.ctx)
            self.ctx = None
        if self.model:
            self.lib.llama_model_free(self.model)
            self.model = None
