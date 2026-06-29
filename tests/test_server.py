"""Tests for the NeMo OpenAI-compatible ASR server."""

import asyncio
import os
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("INTERNAL_API_KEY", "test")
os.environ.setdefault("MODEL_NAME", "test-model")

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from src.nemo_openai_server import (
    SUPPORTED_LANGUAGES,
    GlobalBatchManager,
    QueuedRequest,
    app,
    request_id_var,
    validate_language,
)


class MockModel:
    """Mock NeMo ASR model for testing."""

    def transcribe(self, audio_samples, batch_size=1, source_lang="en", target_lang="en"):
        """Mock transcription."""
        results = []
        for _audio_sample in audio_samples:
            results.append(type("Output", (), {"text": "Transcribed text"})())
        return results


class TestQueuedRequest:
    """Tests for the QueuedRequest NamedTuple."""

    @pytest.mark.asyncio
    async def test_queued_request_creation(self):
        """Test QueuedRequest can be created with all fields."""
        future = asyncio.get_running_loop().create_future()
        req = QueuedRequest(
            model="nvidia/canary-1b-v2",
            audio_data=b"\x00" * 100,
            audio_duration=1.0,
            source_lang="en",
            target_lang="en",
            future=future,
        )
        assert req.model == "nvidia/canary-1b-v2"
        assert len(req.audio_data) == 100
        assert req.audio_duration == 1.0
        assert req.source_lang == "en"
        assert req.target_lang == "en"
        assert isinstance(req.future, asyncio.Future)


class TestGlobalBatchManager:
    """Tests for GlobalBatchManager."""

    @pytest.fixture
    def mock_models(self):
        """Return a mock model dictionary."""
        model = MockModel()
        return {"model_cpu": model}

    @pytest.mark.asyncio
    async def test_initialization(self, mock_models):
        """Test that batch manager initializes correctly."""
        bm = GlobalBatchManager(
            models=mock_models,
            max_batch_size=4,
            max_wait_ms=5,
            thread_pool_size=1,
        )
        try:
            assert bm.max_batch_size == 4
            assert bm.max_wait_ms == 5
            assert hasattr(bm, "request_queue")
            assert hasattr(bm, "batch_processor_task")
            assert len(bm.gpu_workers) > 0
        finally:
            await bm.shutdown()

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self, mock_models):
        """Test enqueueing a request and getting a result."""
        bm = GlobalBatchManager(
            models=mock_models,
            max_batch_size=4,
            max_wait_ms=5,
            thread_pool_size=1,
        )
        try:
            with patch("src.nemo_openai_server.sf.read") as mock_read:
                mock_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
                result = await bm.enqueue(
                    model="model_cpu",
                    audio_data=b"\x00" * 100,
                    audio_duration=1.0,
                    source_lang="en",
                    target_lang="en",
                )
                assert result is not None
                assert "text" in result
                assert "model" in result
                assert result["model"] == "model_cpu"
        finally:
            await bm.shutdown()

    @pytest.mark.asyncio
    async def test_mixed_language_requests_are_not_rebatched_together(self, mock_models):
        """Test concurrent language pairs stay isolated after global batching."""
        bm = GlobalBatchManager(
            models=mock_models,
            max_batch_size=4,
            max_wait_ms=5,
            thread_pool_size=1,
        )
        try:
            with patch("src.nemo_openai_server.sf.read") as mock_read:
                mock_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
                results = await asyncio.gather(
                    bm.enqueue(
                        model="model_cpu",
                        audio_data=b"\x00" * 100,
                        audio_duration=1.0,
                        source_lang="en",
                        target_lang="en",
                    ),
                    bm.enqueue(
                        model="model_cpu",
                        audio_data=b"\x00" * 100,
                        audio_duration=1.0,
                        source_lang="es",
                        target_lang="es",
                    ),
                )

                assert [result["source_lang"] for result in results] == ["en", "es"]
                assert [result["target_lang"] for result in results] == ["en", "es"]
        finally:
            await bm.shutdown()


class TestValidateLang:
    """Tests for language validation."""

    def test_valid_language(self):
        """Test valid language codes."""
        for lang in SUPPORTED_LANGUAGES:
            result = validate_language(lang, "source_lang")
            assert result == lang.lower()

    def test_auto_is_invalid_target_language(self):
        """Test that auto language detection is only accepted for source_lang."""
        with pytest.raises(HTTPException) as exc_info:
            validate_language("auto", "target_lang")
        assert exc_info.value.status_code == 400

    def test_invalid_language_raises(self):
        """Test invalid language raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_language("xy", "source_lang")
        assert exc_info.value.status_code == 400

    def test_case_insensitive(self):
        """Test that language codes are case-insensitive."""
        assert validate_language("EN", "source_lang") == "en"


class TestEndpoints:
    """Integration tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Async client for testing with mock models."""
        with patch("src.nemo_openai_server.load_model") as mock_load:
            mock_model = MockModel()
            mock_load.return_value = mock_model
            # Mock the model names and batch manager as if lifespan ran
            app.state.model_names = ["test-model"]
            app.state.models = {"test-model": MockModel()}
            app.state.batch_manager = MagicMock()
            app.state.batch_manager.enqueue = AsyncMock(
                return_value={
                    "text": "Test transcription",
                    "model": "test-model",
                    "tokens": 0,
                    "duration": 0.5,
                }
            )
            transport = ASGITransport(app=app)
            return AsyncClient(transport=transport, base_url="http://test")

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health endpoint returns correct structure."""
        response = await client.get("/health", headers={"Authorization": "Bearer test"})
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "request_id" in data
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_auth_required(self, client):
        """Test health endpoint requires auth."""
        response = await client.get("/health")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_healthz_endpoint(self, client):
        """Test unauthenticated readiness endpoint."""
        app.state.ready = True
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns Prometheus metrics."""
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "nemo_requests_total" in response.text

    @pytest.mark.asyncio
    async def test_models_endpoint(self, client):
        """Test models endpoint returns list of models."""
        response = await client.get("/v1/models", headers={"Authorization": "Bearer test"})
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)

    @pytest.mark.asyncio
    async def test_models_auth_required(self, client):
        """Test models endpoint requires auth."""
        response = await client.get("/v1/models")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_transcribe_invalid_audio(self, client):
        """Test that invalid audio returns 400."""
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.txt", b"Not an audio file", "text/plain")},
            data={"model": "test"},
            headers={"Authorization": "Bearer test"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_transcribe_invalid_language(self, client):
        """Test that invalid language code returns 400."""
        sample_rate = 16000
        duration = 0.5
        audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)

        with patch("src.nemo_openai_server.sf.read") as mock_read:
            mock_read.return_value = (audio_data, sample_rate)
            # Write WAV to BytesIO (explicitly specify format)
            bio = BytesIO()
            sf.write(bio, audio_data, sample_rate, format="WAV")
            bio.seek(0)

            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", bio, "audio/wav")},
                data={"source_lang": "zzzz", "target_lang": "zzzz"},
                headers={"Authorization": "Bearer test"},
            )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_transcribe_success(self, client):
        """Test successful transcription with valid audio."""
        sample_rate = 16000
        duration = 0.5
        audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)

        with patch("src.nemo_openai_server.sf.read") as mock_read:
            mock_read.return_value = (audio_data, sample_rate)
            # Write WAV to BytesIO (explicitly specify format)
            bio = BytesIO()
            sf.write(bio, audio_data, sample_rate, format="WAV")
            bio.seek(0)

            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", bio, "audio/wav")},
                data={"source_lang": "en", "target_lang": "en"},
                headers={"Authorization": "Bearer test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "model" in data


class TestStructuredLogging:
    """Test structured logging with request ID."""

    def test_request_id_context(self):
        """Test that request ID context variable can be set and read."""
        request_id_var.set("test-request-123")
        assert request_id_var.get() == "test-request-123"
        request_id_var.set(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
