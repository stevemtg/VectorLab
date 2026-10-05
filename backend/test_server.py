import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
import unittest.mock
from unittest.mock import MagicMock, patch
import uuid

import numpy as np
from starlette.requests import ClientDisconnect

from backend import server


class NativeConnectTest(unittest.TestCase):
    def test_chat_has_no_generated_token_limit(self):
        body = server.Chat(
            model="local-model",
            messages=[{"role": "user", "content": "Continue until finished."}],
        )

        self.assertFalse(hasattr(body, "max_tokens"))

    def test_native_connect_does_not_reject_large_model_files(self):
        model_path = MagicMock()
        model_path.stat.return_value.st_size = 32 * 1024**3
        digest = "sha256:" + "a" * 64
        native_model = SimpleNamespace(layers=80, dimensions=8192)
        initial_connection = {
            "engine": None,
            "model": None,
            "digest": None,
            "layers": None,
            "dimensions": None,
        }

        with (
            patch.object(server, "lock", threading.Lock()),
            patch.object(server, "native", None),
            patch.object(server, "connection", initial_connection.copy()),
            patch.object(server, "resolve_model", return_value=({}, model_path, digest)),
            patch.object(server, "NativeModel", return_value=native_model) as load_model,
            patch.object(server, "vectors_for", return_value=[]),
        ):
            result = server.connect(server.Connect(model="large-model:latest", engine="native"))

        self.assertTrue(result["connected"])
        self.assertEqual(result["model"], "large-model:latest")
        self.assertEqual(result["layers"], 80)
        load_model.assert_called_once_with(server.ROOT / ".runtime/llama", model_path)


class VectorStorageTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(server, "DATA", Path(directory.name))
        patcher.start()
        self.addCleanup(patcher.stop)


class ExtractEndpointTest(VectorStorageTest):
    def test_extract_splits_prompt_lines_and_passes_pooling_mode(self):
        digest = "sha256:" + "b" * 64
        direction = np.full((3, 2), 2.0, dtype=np.float32)
        stats = {"layers": 3, "dimensions": 2, "mode": "final_token", "best_layer": 2, "auc_layer": 1, "ratio_target": 0.6, "ratio": 0.5, "samples": {"positive": 2, "negative": 1}, "difference_norm": 2.0, "layer_norms": [1.0, 2.0, 3.0], "layer_ratios": [0.5, 0.4, 0.3], "layer_aucs": [0.9, 0.8, 0.7], "method": "raw difference"}
        fake = MagicMock()
        fake.extract.return_value = (direction, stats)
        connection = {"engine": "native", "model": "local-model", "digest": digest, "layers": 4, "dimensions": 2}
        with (
            patch.object(server, "lock", threading.Lock()),
            patch.object(server, "native", fake),
            patch.object(server, "connection", connection),
        ):
            record = server.extract(server.Extract(name="Joy", positive="+A\n+B", negative="-A", mode="final_token"))
        fake.extract.assert_called_once_with(["+A", "+B"], ["-A"], "final_token")
        self.assertEqual(record["best_layer"], 2)
        self.assertEqual(record["mode"], "final_token")
        self.assertEqual(record["layer_aucs"], [0.9, 0.8, 0.7])
        identifier = record["id"]
        try:
            np.testing.assert_allclose(np.load(server.DATA / f"{identifier}.npy", allow_pickle=False), direction)
        finally:
            (server.DATA / f"{identifier}.json").unlink(missing_ok=True)
            (server.DATA / f"{identifier}.npy").unlink(missing_ok=True)


class ChatInjectionTest(VectorStorageTest):
    def test_disconnected_stream_releases_model_even_before_first_chunk(self):
        for disconnect_at in ("http.response.start", "http.response.body"):
            native = SimpleNamespace(cancel=threading.Event(), set_steering=lambda *_: None, generate=lambda _: iter([{"type": "done"}]))
            with self.subTest(disconnect_at=disconnect_at), patch.object(server, "lock", threading.Lock()), patch.object(server, "native", native), patch.object(server, "connection", {"engine": "native", "model": "local", "digest": "test"}):
                response = server.chat(server.Chat(model="local", messages=[{"role": "user", "content": "hi"}]))

                async def send(message):
                    if message["type"] == disconnect_at:
                        raise OSError("Browser disconnected")

                async def receive():
                    return {"type": "http.disconnect"}

                with self.assertRaises(ClientDisconnect):
                    asyncio.run(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send))
                self.assertFalse(server.lock.locked())
                # Cleanup may run again after another request acquired the lock.
                server.lock.acquire()
                response.close_stream()
                self.assertTrue(server.lock.locked())
                server.lock.release()

    def test_legacy_metadata_remains_loadable_without_invented_measurements(self):
        identifier = str(uuid.uuid4())
        record = {"id": identifier, "name": "Legacy", "model_digest": "legacy-model", "created_at": 1.0, "layers": 3, "dimensions": 2, "difference_norm": 2.0, "layer_norms": [1.0, 2.0, 3.0]}
        (server.DATA / f"{identifier}.json").write_text(json.dumps(record), encoding="utf-8")
        np.save(server.DATA / f"{identifier}.npy", np.ones((3, 2)), allow_pickle=False)
        self.assertEqual(server.vectors_for("legacy-model"), [record])
        captured = {}
        native = SimpleNamespace(cancel=threading.Event(), set_steering=lambda vectors, coefficients: captured.update(vectors=vectors), generate=lambda _: iter([{"type": "done"}]))

        async def collect(response):
            return [chunk async for chunk in response.body_iterator]

        with patch.object(server, "lock", threading.Lock()), patch.object(server, "native", native), patch.object(server, "connection", {"engine": "native", "model": "local", "digest": "legacy-model"}):
            response = server.chat(server.Chat(model="local", messages=[{"role": "user", "content": "hi"}], coefficients=[{"id": identifier, "value": 1.0}]))
            self.assertIn('"type": "done"', asyncio.run(collect(response))[-1])
            self.assertFalse(server.lock.locked())
        self.assertNotIn("best_layer", captured["vectors"][identifier])
        np.testing.assert_allclose(captured["vectors"][identifier]["direction"], np.ones((3, 2)))

    def test_invalid_vector_returns_actionable_error_and_releases_lock(self):
        native = SimpleNamespace(cancel=threading.Event(), set_steering=MagicMock(side_effect=ValueError("Invalid injection layer")))
        with patch.object(server, "lock", threading.Lock()), patch.object(server, "native", native), patch.object(server, "connection", {"engine": "native", "model": "local", "digest": "invalid"}):
            with self.assertRaises(server.HTTPException) as error:
                server.chat(server.Chat(model="local", messages=[{"role": "user", "content": "hi"}]))
            self.assertEqual(error.exception.status_code, 422)
            self.assertIn("Invalid injection layer", error.exception.detail)
            self.assertFalse(server.lock.locked())

    def test_chat_merges_record_metadata_with_raw_direction(self):
        digest = "sha256:" + "c" * 64
        identifier = str(uuid.uuid4())
        direction = np.arange(6, dtype=np.float32).reshape(3, 2)
        np.save(server.DATA / f"{identifier}.npy", direction, allow_pickle=False)
        record = {"id": identifier, "name": "Joy", "positive": "+A", "negative": "-A", "model_digest": digest, "model": "local-model", "created_at": 1.0, "layers": 3, "dimensions": 2, "mode": "final_token", "best_layer": 2, "auc_layer": 1, "ratio_target": 0.6, "ratio": 0.5, "samples": {"positive": 1, "negative": 1}, "difference_norm": 2.0, "layer_norms": [1.0, 2.0, 3.0], "layer_ratios": [0.5, 0.4, 0.3], "layer_aucs": [0.9], "method": "raw difference"}
        captured = {}
        native = SimpleNamespace(cancel=threading.Event(), set_steering=lambda vectors, coefficients: captured.update(vectors=vectors, coefficients=coefficients), generate=lambda messages: iter([{"type": "done", "tokens": 1, "layer_norms": [1.0, 2.0, 3.0, 4.0], "alignments": {}, "injection_norm": 6.0, "layers": 4, "dimensions": 2}]))
        body = server.Chat(model="local-model", messages=[server.Message(role="user", content="hi")], coefficients=[server.Coefficient(id=identifier, value=2.5)])

        async def collect():
            return [chunk async for chunk in response.body_iterator]

        try:
            with (
                patch.object(server, "lock", threading.Lock()),
                patch.object(server, "native", native),
                patch.object(server, "connection", {"engine": "native", "model": "local-model", "digest": digest, "layers": 4, "dimensions": 2}),
                patch.object(server, "vectors_for", return_value=[record]),
            ):
                response = server.chat(body)
                chunks = asyncio.run(collect())
        finally:
            (server.DATA / f"{identifier}.npy").unlink(missing_ok=True)
        merged = captured["vectors"][identifier]
        np.testing.assert_allclose(np.asarray(merged["direction"], dtype=np.float32), direction)
        self.assertEqual(merged["best_layer"], 2)
        self.assertEqual(merged["mode"], "final_token")
        self.assertEqual(captured["coefficients"], [(identifier, 2.5)])
        self.assertIn('"type": "done"', chunks[-1])


if __name__ == "__main__":
    unittest.main()
