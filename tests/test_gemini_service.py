import os
from unittest.mock import MagicMock

import pytest
from google.genai.errors import APIError

os.environ.setdefault("GEMINI_API_KEY", "test-key-for-unit-tests")

from services.gemini_service import GeminiService  # noqa: E402


class FakeResponse:
    def __init__(self, text):
        self.text = text


def _service(monkeypatch, max_attempts=3):
    monkeypatch.setattr("services.gemini_service.time.sleep", lambda *_: None)
    svc = GeminiService(max_attempts=max_attempts)
    svc.client.models.generate_content = MagicMock()
    return svc


VALID_JSON = '[{"id": 1, "defect_type": "short", "explanation": "e", "root_cause": "r", "severity": "Critical", "recommended_fix": "f", "prevention": "p"}]'


class TestGeminiServiceHappyPath:
    def test_returns_parsed_list_on_first_success(self, monkeypatch):
        svc = _service(monkeypatch)
        svc.client.models.generate_content.return_value = FakeResponse(VALID_JSON)

        result = svc.generate(["some prompt text"])

        assert isinstance(result, list)
        assert result[0]["id"] == 1
        assert svc.client.models.generate_content.call_count == 1

    def test_strips_markdown_fences(self, monkeypatch):
        svc = _service(monkeypatch)
        fenced = "```json\n" + VALID_JSON + "\n```"
        svc.client.models.generate_content.return_value = FakeResponse(fenced)

        result = svc.generate(["prompt"])
        assert result[0]["id"] == 1


class TestGeminiServiceJsonRetry:
    def test_retries_on_malformed_json_then_succeeds(self, monkeypatch):
        svc = _service(monkeypatch)
        svc.client.models.generate_content.side_effect = [
            FakeResponse("not json at all"),
            FakeResponse(VALID_JSON),
        ]

        result = svc.generate(["prompt"])
        assert result[0]["id"] == 1
        assert svc.client.models.generate_content.call_count == 2

    def test_retries_on_non_list_json(self, monkeypatch):
        svc = _service(monkeypatch)
        svc.client.models.generate_content.side_effect = [
            FakeResponse('{"not": "a list"}'),
            FakeResponse(VALID_JSON),
        ]
        result = svc.generate(["prompt"])
        assert result[0]["id"] == 1

    def test_raises_after_exhausting_attempts_on_bad_json(self, monkeypatch):
        svc = _service(monkeypatch, max_attempts=2)
        svc.client.models.generate_content.return_value = FakeResponse("still not json")

        with pytest.raises(RuntimeError):
            svc.generate(["prompt"])
        assert svc.client.models.generate_content.call_count == 2

    def test_retries_on_empty_response_text(self, monkeypatch):
        svc = _service(monkeypatch)
        svc.client.models.generate_content.side_effect = [
            FakeResponse(""),
            FakeResponse(VALID_JSON),
        ]
        result = svc.generate(["prompt"])
        assert result[0]["id"] == 1


class TestGeminiServiceApiErrorRetry:
    def test_retries_on_retryable_api_error_then_succeeds(self, monkeypatch):
        svc = _service(monkeypatch)
        busy = APIError(code=503, response_json={"error": {"message": "busy"}})
        svc.client.models.generate_content.side_effect = [busy, FakeResponse(VALID_JSON)]

        result = svc.generate(["prompt"])
        assert result[0]["id"] == 1

    def test_raises_runtime_error_after_exhausting_attempts_on_api_error(self, monkeypatch):
        svc = _service(monkeypatch, max_attempts=2)
        busy = APIError(code=503, response_json={"error": {"message": "busy"}})
        svc.client.models.generate_content.side_effect = [busy, busy]

        with pytest.raises(RuntimeError):
            svc.generate(["prompt"])


class TestGeminiServiceContentConversion:
    def test_accepts_mixed_text_and_image_contents(self, monkeypatch):
        import numpy as np
        svc = _service(monkeypatch)
        svc.client.models.generate_content.return_value = FakeResponse(VALID_JSON)

        img = np.zeros((10, 10, 3), dtype="uint8")
        result = svc.generate(["intro text", "label 1", img])
        assert result[0]["id"] == 1

    def test_rejects_unsupported_content_type(self, monkeypatch):
        svc = _service(monkeypatch)
        with pytest.raises(TypeError):
            svc.generate([object()])


class TestGeminiServiceInit:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            GeminiService()

    def test_model_env_override(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-test-model")
        svc = GeminiService()
        assert svc.model == "gemini-custom-test-model"
