import asyncio
import json
from pathlib import Path
import tempfile
import socket
import threading
import unittest
from types import SimpleNamespace
import unittest.mock
from unittest.mock import MagicMock, patch
import uuid

import numpy as np
from starlette.requests import ClientDisconnect
from starlette.requests import Request
from fastapi.responses import JSONResponse

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
        load_model.assert_called_once_with(server.RUNTIME, model_path)


class VectorImportGateTest(unittest.TestCase):
    def test_import_endpoint_is_disabled_without_the_desktop_flag(self):
        with patch.object(server, "lock", threading.Lock()):
            with patch.object(server, "TOKEN", ""), patch.object(server, "CONTROL_TOKEN", ""):
                with self.assertRaises(server.HTTPException) as error:
                    server.import_vectors(server.VectorImportRequest(source="C:/somewhere"))
        self.assertEqual(error.exception.status_code, 403)

    def test_import_rejects_renderer_token_without_main_process_capability(self):
        with patch.object(server, "TOKEN", "secret"), patch.object(server, "CONTROL_TOKEN", "main-secret"):
            self.assertEqual(bridge_request("POST", "/api/vectors/import", {"authorization": "Bearer secret", "x-vector-lab": "1"}).status_code, 403)


def bridge_request(method="GET", path="/api/health", headers=None):
    values = {"host": "127.0.0.1:8788", **(headers or {})}
    request = Request({"type": "http", "method": method, "path": path, "headers": [(key.encode(), value.encode()) for key, value in values.items()], "query_string": b"", "scheme": "http", "server": ("127.0.0.1", 8788)})
    async def call_next(_request):
        return JSONResponse({"ok": True})
    return asyncio.run(server.local_only(request, call_next))


class DesktopSecurityTest(unittest.TestCase):
    def test_authentication_host_origin_and_mutation_checks_are_independent(self):
        with patch.object(server, "TOKEN", "secret"), patch.object(server, "ORIGINS", ["vectorlab://bundle"]):
            self.assertEqual(bridge_request().status_code, 401)
            authorized = {"authorization": "Bearer secret", "origin": "vectorlab://bundle"}
            self.assertEqual(bridge_request(headers=authorized).status_code, 200)
            for host in ("evil.example", "localhost.evil", "127.0.0.1@evil", "localhost:abc"):
                self.assertEqual(bridge_request(headers={**authorized, "host": host}).status_code, 403)
            for origin in ("null", "https://evil.example", "http://127.0.0.1:5173"):
                self.assertEqual(bridge_request(headers={**authorized, "origin": origin}).status_code, 403)
            self.assertEqual(bridge_request("POST", "/api/stop", authorized).status_code, 403)
            self.assertEqual(bridge_request("POST", "/api/stop", {**authorized, "x-vector-lab": "1"}).status_code, 200)
            self.assertEqual(bridge_request("OPTIONS", headers={"origin": "vectorlab://bundle"}).status_code, 200)
            self.assertEqual(bridge_request(headers={"authorization": "Bearer wrong"}).status_code, 401)

    def test_web_workflow_keeps_existing_header_protection(self):
        with patch.object(server, "TOKEN", ""):
            self.assertEqual(bridge_request().status_code, 200)
            self.assertEqual(bridge_request("POST", "/api/stop").status_code, 403)

    def test_ollama_chat_does_not_require_access_to_local_manifest(self):
        with patch.object(server, "lock", threading.Lock()), patch.object(server, "native", None), patch.object(server, "connection", {}), patch.object(server, "model_catalog", return_value=[{"name": "service-model", "digest": "manifest-digest"}]), patch.object(server, "resolve_model_files", side_effect=server.HTTPException(404, "Files are not accessible")), patch.object(server, "ollama") as upstream:
            import io
            upstream.return_value.__enter__.return_value = io.StringIO(json.dumps({"capabilities": ["completion"]}))
            self.assertEqual(server.connect(server.Connect(model="service-model", engine="ollama"))["engine"], "ollama")


class VectorStorageTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(server, "DATA", Path(directory.name))
        patcher.start()
        self.addCleanup(patcher.stop)


class VectorMigrationTest(VectorStorageTest):
    def test_import_validates_pairs_preserves_digest_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            identifier = str(uuid.uuid4())
            record = {"id": identifier, "name": "Legacy", "model": "local", "model_digest": "sha256:" + "a" * 64,
                      "positive": "Yes", "negative": "No", "method": "unit", "created_at": 1,
                      "layers": 2, "dimensions": 3, "difference_norm": 2, "layer_norms": [1, 2]}
            metadata = source / f"{identifier}.json"
            metadata.write_text(json.dumps(record), encoding="utf-8")
            array_file = source / f"{identifier}.npy"
            for array in (np.ones((2, 2)), np.full((2, 3), np.nan), np.array([[object()]], dtype=object)):
                np.save(array_file, array)
                self.assertEqual(server.migrate_vectors(source), 0)
            np.save(array_file, np.ones((2, 3)))
            self.assertEqual(server.migrate_vectors(source), 1)
            self.assertEqual(server.migrate_vectors(source), 0)
            self.assertEqual(server.vectors_for(record["model_digest"]), [record])
            self.assertNotIn("best_layer", server.vectors_for(record["model_digest"])[0])
            metadata.write_text("[]", encoding="utf-8")
            self.assertEqual(server.migrate_vectors(source), 0)

    def test_forged_saved_identifier_cannot_reference_paths_outside_data(self):
        (server.DATA / "bad.json").write_text(json.dumps({"id": "../outside", "model_digest": "test", "created_at": 1}), encoding="utf-8")
        self.assertEqual(server.vectors_for("test"), [])


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
    def test_stop_interrupts_ollama_while_waiting_for_first_response(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        self.addCleanup(listener.close)
        accepted = threading.Event()
        finish = threading.Event()

        def stall():
            client, _ = listener.accept()
            with client:
                client.recv(65536)
                accepted.set()
                finish.wait(5)

        threading.Thread(target=stall, daemon=True).start()
        events = []
        with patch.object(server, "lock", threading.Lock()), patch.object(server, "native", None), patch.object(server, "connection", {"engine": "ollama", "model": "local", "digest": "test"}), patch.object(server, "OLLAMA", f"http://127.0.0.1:{listener.getsockname()[1]}"):
            response = server.chat(server.Chat(model="local", messages=[{"role": "user", "content": "hi"}]))

            async def collect():
                events.extend([chunk async for chunk in response.body_iterator])

            worker = threading.Thread(target=lambda: asyncio.run(collect()), daemon=True)
            worker.start()
            try:
                self.assertTrue(accepted.wait(2))
                self.assertIsNotNone(server.ollama_task)
                server.stop()
                worker.join(2)
                self.assertFalse(worker.is_alive(), "Stop must interrupt the blocked Ollama read")
                self.assertFalse(server.lock.locked())
                self.assertTrue(json.loads(events[-1])["cancelled"])
            finally:
                finish.set()
                worker.join(3)

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
