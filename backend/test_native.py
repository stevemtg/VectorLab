import ctypes as C
import json
import os
from pathlib import Path
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from backend.native import NativeModel, P, I, FP, RATIO_TARGET, RATIO_PROBES, runtime_available


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
        model_root = Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama/models")))
        runtime = Path(os.environ.get("VECTOR_LAB_RUNTIME_DIR", str(root / ".runtime/llama")))
        manifest = model_root / "manifests/registry.ollama.ai/library/qwen2.5-coder/1.5b"
        if not manifest.exists():
            raise unittest.SkipTest("Requires the user's installed qwen2.5-coder:1.5b model.")
        if not runtime_available(runtime):
            raise unittest.SkipTest("Requires a matching b11146 CPU runtime for this process platform/architecture.")
        data = json.loads(manifest.read_text())
        digest = next(layer["digest"] for layer in data["layers"] if layer["mediaType"] == "application/vnd.ollama.image.model")
        cls.model = NativeModel(runtime, model_root / "blobs" / digest.replace(":", "-"))

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_real_extraction_injection_and_reversible_logits(self):
        model = self.model
        positives = ["You succeeded, excellent job!", "What wonderful luck! Everything went right."]
        negatives = ["Task completed.", "The form is on the desk."]
        direction, stats = model.extract(positives, negatives, "mean")
        self.assertEqual(direction.shape, (model.layers - 1, model.dimensions))
        self.assertEqual(stats["mode"], "mean")
        self.assertEqual(len(stats["layer_norms"]), model.layers - 1)
        self.assertTrue(1 <= stats["best_layer"] <= model.layers - 1)
        norms = np.linalg.norm(direction, axis=1)
        np.testing.assert_allclose(norms, np.asarray(stats["layer_norms"]), rtol=1e-4, atol=1e-4)
        self.assertGreater(float(norms.min()), 1.5, "Directions must stay raw (un-normalized) difference vectors.")
        logits_fn = model.lib.llama_get_logits_ith
        logits_fn.restype, logits_fn.argtypes = FP, [P, I]
        vocab_fn = model.lib.llama_vocab_n_tokens
        vocab_fn.restype, vocab_fn.argtypes = I, [P]
        count = vocab_fn(model.vocab)
        tokens = model.tokenize("Please explain why learning new things is useful.")
        def logits(value):
            model.set_steering({"pleasure": {**stats, "direction": direction}}, [("pleasure", value)])
            model.clear()
            model.capture_enabled = True
            model.decode(tokens)
            return np.ctypeslib.as_array(logits_fn(model.ctx, -1), shape=(count,)).copy()
        baseline, steered, restored = logits(0), logits(3), logits(0)
        shift = float(np.linalg.norm(steered - baseline))
        self.assertGreater(shift, .01, "Injection must change actual model logits.")
        np.testing.assert_allclose(baseline, restored, atol=1e-4)
        self.assertEqual(len(model.capture), model.layers)
        print(json.dumps({"real_model": "qwen2.5-coder:1.5b", "layers": model.layers, "dimensions": model.dimensions, "selected_layer": stats["best_layer"], "logit_shift_L_selected_at_lambda_3": shift, "neutral_restored": True}), flush=True)

    def test_real_generation(self):
        self.model.set_steering({}, [])
        events = list(self.model.generate([{"role": "user", "content": "What is 2 + 2? Answer briefly."}]))
        text = "".join(event.get("text", "") for event in events)
        self.assertIn("4", text)
        self.assertEqual(events[-1]["type"], "done")
        self.assertGreater(events[-1]["tokens"], 0)
        self.assertTrue(all(n is not None for n in events[-1]["layer_norms"]))

    def test_mean_pooling_can_capture_every_token_in_the_final_layer(self):
        model = self.model
        model.clear()
        model.capture_enabled = True
        tokens = model.tokenize("Please explain why learning new things is useful.")
        model.decode(tokens, all_outputs=True)
        self.assertEqual(model.capture[model.layers - 1].shape, (len(tokens), model.dimensions))
        model.capture_enabled = False


class NativeExtractionTest(unittest.TestCase):
    """Runtime-free checks for the reference methodology: raw difference vectors,
    per-layer AUC curves, ratio-based injection-layer selection, baseline-spread
    denoising, and single-row (selected-layer-only) injection."""

    def make_model(self, responses: dict[str, list[list[float]]]):
        model = NativeModel.__new__(NativeModel)
        model.layers = 5
        model.dimensions = 2
        model.ctx = object()
        model.vocab = object()
        model.cancel = threading.Event()
        model.capture = {}
        model.capture_error = None
        model.capture_enabled = False
        model.directions = {}
        model.active = []
        model.mixed = np.zeros((4, 2), dtype=np.float32)
        model.lib = SimpleNamespace(llama_set_adapter_cvec=lambda *_: 0)

        def clear():
            model.capture.clear()
            model.capture_error = None

        def decode(_tokens, all_outputs=False):
            rows = np.array(responses.get(model.pending, [[1.0, 0.0]]) if model.pending in RATIO_PROBES else responses[model.pending], dtype=np.float32)
            for layer in range(model.layers):
                model.capture[layer] = rows.copy()

        model.clear = clear
        model.decode = decode
        model.tokenize = lambda text: (setattr(model, "pending", text), [1, 2])[1]
        return model

    def test_extract_returns_raw_difference_and_layer_metadata(self):
        model = self.make_model({"+A": [[3.0, 6.0], [3.0, 2.0]], "-A": [[1.0, 0.0], [1.0, 0.0]]})
        direction, stats = model.extract(["+A"], ["-A"], "mean")
        self.assertEqual(direction.shape, (4, 2))
        expected_norm = float(np.linalg.norm([2.0, 4.0]))
        self.assertGreater(expected_norm, 1.5, "Directions must stay raw (un-normalized) difference vectors.")
        for row in direction:
            np.testing.assert_allclose(row, [2.0, 4.0], rtol=1e-5)
        self.assertAlmostEqual(stats["layer_norms"][0], expected_norm, places=4)
        self.assertEqual(stats["mode"], "mean")
        self.assertEqual(stats["auc_layer"], 1)
        self.assertEqual(stats["best_layer"], 1)
        self.assertEqual(stats["ratio_target"], RATIO_TARGET)
        self.assertAlmostEqual(stats["ratio"], expected_norm / 1.0, places=4)
        self.assertTrue(all(abs(value - 1.0) < 1e-9 for value in stats["layer_aucs"]))
        self.assertEqual(stats["samples"], {"positive": 1, "negative": 1})
        direction, stats = model.extract(["+A"], ["-A"], "final_token")
        np.testing.assert_allclose(direction[0], [2.0, 2.0], rtol=1e-5)
        self.assertEqual(stats["mode"], "final_token")

    def test_baseline_spread_is_projected_out_and_contrast_survives_raw(self):
        model = self.make_model({"+A": [[5.0, 5.0], [5.0, 5.0]], "+B": [[5.0, -5.0], [5.0, -5.0]], "-A": [[1.0, 7.0], [1.0, 7.0]], "-B": [[1.0, -7.0], [1.0, -7.0]]})
        direction, stats = model.extract(["+A", "+B"], ["-A", "-B"], "mean")
        for row in direction:
            np.testing.assert_allclose(row, [4.0, 0.0], atol=1e-5)
        self.assertAlmostEqual(stats["layer_norms"][0], 4.0, places=4)
        self.assertAlmostEqual(stats["layer_ratios"][0], 4.0, places=5)

    def test_contrast_inside_baseline_spread_is_denoised_away(self):
        model = self.make_model({"+A": [[1.0, 5.0], [1.0, 5.0]], "+B": [[1.0, -5.0], [1.0, -5.0]], "-A": [[1.0, 7.0], [1.0, 7.0]], "-B": [[1.0, -7.0], [1.0, -7.0]]})
        with self.assertRaises(ValueError):
            model.extract(["+A", "+B"], ["-A", "-B"], "mean")

    def test_injection_hits_only_the_selected_layer_row(self):
        model = self.make_model({"-A": [[1.0, 0.0], [1.0, 0.0]]})
        direction = np.zeros((4, 2), dtype=np.float32)
        direction[0] = [1.0, 0.0]
        direction[1] = [0.0, 10.0]
        direction[2] = [7.0, 7.0]
        direction[3] = [1.0, 1.0]
        model.set_steering({"k": {"direction": direction, "best_layer": 3, "mode": "mean"}}, [("k", 2.0)])
        np.testing.assert_allclose(model.mixed[2], [14.0, 14.0])
        self.assertFalse(model.mixed[[0, 1, 3]].any())
        self.assertEqual(model.active, [("k", 2.0)])
        self.assertEqual(model.directions["k"]["best_layer"], 3)

    def test_auc_counts_pairs_with_half_credit_for_ties(self):
        self.assertEqual(NativeModel._auc(np.array([3.0, 1.0]), np.array([0.0, 2.0])), 0.75)
        self.assertEqual(NativeModel._auc(np.array([1.0]), np.array([1.0])), 0.5)
        self.assertEqual(NativeModel._auc(np.array([0.0]), np.array([1.0])), 0.0)

    def test_ratio_selection_and_injection_use_the_same_extracted_vector(self):
        model = self.make_model({})

        def decode(_tokens, all_outputs=False):
            for layer in range(model.layers):
                if model.pending in RATIO_PROBES:
                    row = [([1, 1, 5, 10, 20][layer]), 0]
                elif model.pending == "+A":
                    row = [4, 0] if layer == 1 else [1, 6]
                else:
                    row = [1, 0]
                model.capture[layer] = np.array([row], dtype=np.float32)

        model.decode = decode
        direction, stats = model.extract(["+A"], ["-A"])
        self.assertEqual(stats["extraction_layer"], 1)
        self.assertEqual(stats["best_layer"], 2)
        self.assertAlmostEqual(stats["ratio"], 0.6)
        self.assertEqual(stats["difference_norm"], 3.0)
        self.assertEqual(stats["auc_evaluation"], "training")
        model.set_steering({"k": {**stats, "direction": direction}}, [("k", 2.0)])
        np.testing.assert_allclose(model.mixed[1], [6, 0])
        self.assertFalse(model.mixed[[0, 2, 3]].any())
        model.capture[2] = np.array([[0, 1]], dtype=np.float32)
        self.assertEqual(model.telemetry()["alignments"]["k"], 0.0)

    def test_legacy_vectors_keep_all_layer_injection_and_mean_alignment(self):
        model = self.make_model({})
        direction = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
        model.set_steering({"old": {"direction": direction}}, [("old", 2.0)])
        np.testing.assert_allclose(model.mixed, 2 * direction)
        model.capture = {layer: np.array([[1, 0]], dtype=np.float32) for layer in range(1, 5)}
        self.assertEqual(model.telemetry()["alignments"]["old"], 0.5)
        model.set_steering({}, [])
        self.assertFalse(model.mixed.any())

    def test_missing_residual_layer_is_rejected_and_capture_is_disabled(self):
        model = self.make_model({"+A": [[4, 0]], "-A": [[1, 0]]})
        decode = model.decode

        def missing(tokens, **kwargs):
            decode(tokens, **kwargs)
            del model.capture[2]

        model.decode = missing
        with self.assertRaisesRegex(RuntimeError, "unsupported for its graph"):
            model.extract(["+A"], ["-A"])
        self.assertFalse(model.capture_enabled)

    def test_failed_extraction_disables_capture(self):
        model = self.make_model({"+A": [[1, 0]], "-A": [[1, 0]]})
        with self.assertRaisesRegex(ValueError, "no measurable contrast"):
            model.extract(["+A"], ["-A"])
        self.assertFalse(model.capture_enabled)

    def test_nonfinite_activations_are_rejected(self):
        model = self.make_model({"+A": [[np.nan, 0]], "-A": [[1, 0]]})
        with self.assertRaisesRegex(ValueError, "non-finite activations"):
            model.extract(["+A"], ["-A"])

    def test_invalid_saved_vector_metadata_and_values_are_rejected(self):
        model = self.make_model({})
        valid = {"direction": np.ones((4, 2)), "best_layer": 2, "mode": "mean"}
        for change in ({"best_layer": None}, {"best_layer": 2.5}, {"extraction_layer": 5}, {"mode": "unknown"}, {"direction": np.full((4, 2), np.inf)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.set_steering({"k": {**valid, **change}}, [("k", 1.0)])


if __name__ == "__main__":
    unittest.main()
