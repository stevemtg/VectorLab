import ctypes as C
import json
from pathlib import Path
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from backend.native import NativeModel, P, I, FP


class NativeGenerationStreamingTest(unittest.TestCase):
    def test_generation_continues_past_context_window(self):
        model = NativeModel.__new__(NativeModel)
        sampled = 0
        clear_count = 0
        decoded = []

        def sample(*_):
            nonlocal sampled
            sampled += 1
            return 5 if sampled <= 2050 else 0

        def token_to_piece(_, __, piece, *___):
            piece.value = b"x"
            return 1

        def clear():
            nonlocal clear_count
            clear_count += 1
            model.capture.clear()
            model.capture_error = None

        model.lib = SimpleNamespace(
            llama_sampler_init_greedy=lambda: 1,
            llama_sampler_sample=sample,
            llama_vocab_is_eog=lambda _, token: token == 0,
            llama_token_to_piece=token_to_piece,
            llama_sampler_free=lambda _: None,
        )
        model.ctx = object()
        model.vocab = object()
        model.cancel = threading.Event()
        model.capture = {}
        model.capture_error = None
        model.capture_enabled = False
        model.format_chat = lambda _: "prompt"
        model.tokenize = lambda _: [1]
        model.decode = lambda tokens: decoded.extend(tokens)
        model.clear = clear
        model.telemetry = lambda: {"layer_norms": [], "alignments": {}, "injection_norm": 0}

        events = list(NativeModel.generate(model, [{"role": "user", "content": "prompt"}]))

        self.assertEqual(sum(event.get("type") == "token" for event in events), 2050)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["tokens"], 2050)
        self.assertEqual(clear_count, 2)


class NativeSteeringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        manifest = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library/qwen2.5-coder/1.5b"
        if not manifest.exists():
            raise unittest.SkipTest("Requires the user's installed qwen2.5-coder:1.5b model.")
        data = json.loads(manifest.read_text())
        digest = next(layer["digest"] for layer in data["layers"] if layer["mediaType"] == "application/vnd.ollama.image.model")
        cls.model = NativeModel(root / ".runtime/llama", Path.home() / ".ollama/models/blobs" / digest.replace(":", "-"))

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_real_extraction_injection_and_reversible_logits(self):
        model = self.model
        direction, stats = model.extract("You succeeded, excellent job!", "Task completed.")
        self.assertEqual(direction.shape, (model.layers - 1, model.dimensions))
        self.assertGreater(stats["difference_norm"], 0)
        np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1, atol=1e-5)
        logits_fn = model.lib.llama_get_logits_ith
        logits_fn.restype, logits_fn.argtypes = FP, [P, I]
        vocab_fn = model.lib.llama_vocab_n_tokens
        vocab_fn.restype, vocab_fn.argtypes = I, [P]
        count = vocab_fn(model.vocab)
        tokens = model.tokenize("Please explain why learning new things is useful.")
        def logits(value):
            model.set_steering({"pleasure": direction}, [("pleasure", value)])
            model.clear()
            model.capture_enabled = True
            model.decode(tokens)
            return np.ctypeslib.as_array(logits_fn(model.ctx, -1), shape=(count,)).copy()
        baseline, steered, restored = logits(0), logits(3), logits(0)
        shift = float(np.linalg.norm(steered - baseline))
        self.assertGreater(shift, .01, "Injection must change actual model logits.")
        np.testing.assert_allclose(baseline, restored, atol=1e-4)
        self.assertEqual(len(model.capture), model.layers)
        print(json.dumps({"real_model": "qwen2.5-coder:1.5b", "layers": model.layers, "dimensions": model.dimensions, "logit_shift_L2_at_lambda_3": shift, "neutral_restored": True}), flush=True)

    def test_real_generation(self):
        self.model.set_steering({}, [])
        events = list(self.model.generate([{"role": "user", "content": "What is 2 + 2? Answer briefly."}]))
        text = "".join(event.get("text", "") for event in events)
        self.assertIn("4", text)
        self.assertEqual(events[-1]["type"], "done")
        self.assertGreater(events[-1]["tokens"], 0)
        self.assertTrue(all(n is not None for n in events[-1]["layer_norms"]))


if __name__ == "__main__":
    unittest.main()
