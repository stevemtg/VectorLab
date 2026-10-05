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

# Reference methodology (Pain-axis): the injection layer is one of the fixed depth
# fractions, the extraction AUC-peak layer, or the final layer — whichever layer's
# raw-vector-norm / residual-norm ratio lands closest to RATIO_TARGET.
RATIO_TARGET = 0.6
LAYER_FRACTIONS = (0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)
DENOISE_VARIANCE = 0.5
# The three independent ratio probes used in Pain-axis's steering ladder.
# https://github.com/valen-research/Pain-axis/blob/4d75cd90e206ea962f7a9101e65c85efea56723b/scripts/4.2_steering/01_steering_ladder.py
RATIO_PROBES = (
    "I put the receipts in the drawer. I feel:",
    "The bus stops at the corner of the street. I feel:",
    "I fill out the form with my address. I feel:",
)


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
        self.directions: dict[str, dict] = {}
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
            self.capture[int(match[1])] = data.reshape(-1, self.dimensions).copy()
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

    def decode(self, tokens, all_outputs=False):
        array = (I * len(tokens))(*tokens)
        batch = self.lib.llama_batch_get_one(array, len(tokens))
        if all_outputs:
            # Otherwise llama.cpp prunes the final layer to the last token, so
            # mean pooling would silently become final-token pooling there.
            logits = (C.c_int8 * len(tokens))(*([1] * len(tokens)))
            batch.logits = logits
        result = self.lib.llama_decode(self.ctx, batch)
        self.lib.llama_synchronize(self.ctx)
        if self.capture_error:
            raise RuntimeError(self.capture_error)
        if result:
            raise RuntimeError("Generation cancelled." if self.cancel.is_set() else f"Native evaluation failed with code {result}.")

    @staticmethod
    def _pool(activation: np.ndarray, mode: str) -> np.ndarray:
        return activation[-1] if mode == "final_token" else activation.mean(axis=0)

    @staticmethod
    def _auc(positives: np.ndarray, negatives: np.ndarray) -> float:
        # Mann-Whitney U: probability a positive sample projects above a baseline
        # sample onto the direction, ties counting half. Identical to roc_auc_score.
        greater = positives[:, None] > negatives[None, :]
        ties = positives[:, None] == negatives[None, :]
        return float(np.mean(greater + 0.5 * ties))

    def set_steering(self, vectors: dict[str, dict], coefficients: list[tuple[str, float]]):
        mixed = np.zeros((self.layers - 1, self.dimensions), dtype=np.float32)
        for key, value in coefficients:
            record = vectors[key]
            direction = np.asarray(record["direction"], dtype=np.float32)
            if direction.shape != mixed.shape or not np.isfinite(direction).all() or not np.isfinite(value):
                raise ValueError("Vector dimensions or values do not match this model.")
            if "best_layer" not in record:
                # Saved before single-layer extraction: retain the original unit
                # directions and all-layer injection, without inventing AUCs.
                mixed += np.float32(value) * direction
                continue
            layer = record["best_layer"]
            source_layer = record.get("extraction_layer", layer)
            if any(type(index) is not int or not 1 <= index < self.layers for index in (layer, source_layer)):
                raise ValueError("Vector dimensions or injection layer do not match this model.")
            if record.get("mode") not in {"mean", "final_token"}:
                raise ValueError("Vector activation pooling mode is invalid.")
            mixed[layer - 1] += np.float32(value) * direction[source_layer - 1]
        self.mixed = np.ascontiguousarray(mixed, dtype=np.float32)
        result = self.lib.llama_set_adapter_cvec(self.ctx, self.mixed.ctypes.data_as(FP), self.mixed.size, self.dimensions, 1, self.layers - 1)
        if result:
            raise RuntimeError("Native runtime rejected the control vector.")
        self.active, self.directions = list(coefficients), dict(vectors)

    def extract(self, positives: list[str], negatives: list[str], mode: str = "mean"):
        if mode not in {"mean", "final_token"}:
            raise ValueError("Activation pooling mode must be mean or final_token.")
        if not positives or not negatives or len(positives) > 8 or len(negatives) > 8:
            raise ValueError("Provide between 1 and 8 contrastive prompts per class.")
        self.cancel.clear()
        self.set_steering({}, [])
        self.capture_enabled = True
        try:
            return self._extract(positives, negatives, mode)
        finally:
            self.capture_enabled = False

    def _extract(self, positives, negatives, mode):
        captured = {}
        for label, texts in (("positive", positives), ("negative", negatives), ("probe", RATIO_PROBES)):
            snapshots = []
            for text in texts:
                self.clear()
                tokens = self.tokenize(text)
                if not tokens or len(tokens) > 512:
                    raise ValueError("Each contrastive prompt must contain between 1 and 512 tokens.")
                pooling = "final_token" if label == "probe" else mode
                self.decode(tokens, all_outputs=pooling == "mean")
                # Keep one pooled vector per sample, not every token of every
                # layer across all samples (several GB for larger checkpoints).
                snapshots.append({i: self._pool(v, pooling).copy() for i, v in self.capture.items()})
            captured[label] = snapshots
        common = sorted(set.intersection(*(set(snapshot) for snapshots in captured.values() for snapshot in snapshots)) & set(range(1, self.layers)))
        if not common or len(common) != self.layers - 1:
            raise RuntimeError(f"This model exposes only {len(common)} usable residual layers. Native extraction is unsupported for its graph.")
        direction = np.zeros((self.layers - 1, self.dimensions), dtype=np.float32)
        norms, probe_norms, aucs = [], [], []
        for layer in common:
            positive = np.stack([snapshot[layer] for snapshot in captured["positive"]])
            negative = np.stack([snapshot[layer] for snapshot in captured["negative"]])
            probes = np.stack([snapshot[layer] for snapshot in captured["probe"]])
            if not all(np.isfinite(values).all() for values in (positive, negative, probes)):
                raise ValueError("The model produced non-finite activations; extraction was not saved.")
            difference = positive.mean(axis=0) - negative.mean(axis=0)
            if len(negative) > 1:
                centered = negative - negative.mean(axis=0)
                _, singular, row_space = np.linalg.svd(centered, full_matrices=False)
                variance = np.cumsum(singular ** 2)
                total = float(variance[-1]) if variance.size else 0.0
                if total > 0:
                    count = int(np.searchsorted(variance / total, DENOISE_VARIANCE) + 1)
                    for component in row_space[:count]:
                        difference = difference - np.dot(difference, component) * component
            norm = float(np.linalg.norm(difference))
            probe = float(np.mean(np.linalg.norm(probes, axis=1)))
            unit = difference / norm if norm > 1e-8 else np.zeros_like(difference)
            norms.append(norm)
            probe_norms.append(probe)
            aucs.append(self._auc(positive @ unit, negative @ unit))
            direction[layer - 1] = difference
        if not np.isfinite(direction).all() or not np.isfinite(norms).all() or not np.isfinite(probe_norms).all():
            raise ValueError("The model produced non-finite vector measurements; extraction was not saved.")
        usable = [layer for layer in common if norms[layer - 1] > 1e-8]
        if not usable:
            raise ValueError("These prompts produce no measurable contrast. Use a different pair.")
        auc_layer = max(usable, key=lambda layer: aucs[layer - 1])
        # The reference moves the SAME extracted vector between candidate layers.
        # Do not substitute a different layer's contrast when choosing injection.
        vector_norm = norms[auc_layer - 1]
        ratios = [vector_norm / probe if probe > 1e-8 else None for probe in probe_norms]
        candidates = sorted({min(max(1, int(self.layers * fraction)), self.layers - 1) for fraction in LAYER_FRACTIONS} | {auc_layer, self.layers - 1})
        candidates = [layer for layer in candidates if ratios[layer - 1] is not None]
        if not candidates:
            raise ValueError("No candidate layer has a measurable probe residual norm.")
        best_layer = min(candidates, key=lambda layer: abs(ratios[layer - 1] - RATIO_TARGET))
        stats = {"layers": len(common), "dimensions": self.dimensions, "mode": mode, "best_layer": int(best_layer), "extraction_layer": int(auc_layer), "auc_layer": int(auc_layer), "auc_evaluation": "training", "ratio_target": RATIO_TARGET, "ratio": ratios[best_layer - 1], "samples": {"positive": len(positives), "negative": len(negatives)}, "difference_norm": vector_norm, "layer_norms": norms, "layer_ratios": ratios, "layer_aucs": aucs, "method": f"raw difference of {mode} residual means (positive − baseline), denoised against the baseline spread; training-AUC extraction layer, independent neutral probes for injection-layer selection"}
        return direction, stats

    def telemetry(self):
        norms = [float(np.linalg.norm(self.capture[i].mean(axis=0))) if i in self.capture else None for i in range(self.layers)]
        alignments = {}
        for key, _ in self.active:
            record = self.directions[key]
            layers = [record["best_layer"]] if "best_layer" in record else range(1, self.layers)
            direction = np.asarray(record["direction"], dtype=np.float32)
            values = []
            for layer in layers:
                activation = self.capture.get(layer)
                if activation is None:
                    continue
                residual = self._pool(activation, record.get("mode", "mean"))
                row = direction[record.get("extraction_layer", layer) - 1]
                denominator = float(np.linalg.norm(residual) * np.linalg.norm(row))
                if denominator > 1e-8:
                    values.append(float(np.dot(residual, row) / denominator))
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
