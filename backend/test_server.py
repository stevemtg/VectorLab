import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


if __name__ == "__main__":
    unittest.main()
