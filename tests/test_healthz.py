import os
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

# Ensure repository root is importable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure required environment variables are present before importing the app
os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("API_KEY", "test")
os.environ.setdefault("SKIP_MODEL_LOAD", "1")

from api import app as app_module  # noqa: E402


client = TestClient(app_module.app)


def _assert_base_health_shape(data: dict[str, Any]) -> None:
    assert set(data.keys()) >= {"status", "model_loaded", "tokenizer_loaded", "device", "error"}


def test_healthz_returns_ok_status_code() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    _assert_base_health_shape(data)
    assert data["status"] in {"ok", "degraded"}


def test_healthz_handles_internal_errors(monkeypatch) -> None:
    def broken_health() -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.classifier, "health", broken_health)

    response = client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    _assert_base_health_shape(data)
    assert data["status"] == "error"
    assert data["error"] == "boom"
