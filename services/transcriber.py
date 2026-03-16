"""
Whisper transcription service using faster-whisper (CTranslate2).

Runs transcription in a thread pool to avoid blocking the async event loop.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from pathlib import Path
from typing import NamedTuple

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class TranscriptSegment(NamedTuple):
    """A single segment of transcribed text."""

    start: float  # seconds offset within the chunk
    end: float  # seconds offset within the chunk
    text: str


class TranscriberService:
    """
    Manages a faster-whisper model and transcribes audio chunks.

    The model is loaded once at startup. Transcription runs in
    asyncio.to_thread() to keep the bot responsive.
    """

    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Parameters
        ----------
        model_size:
            Whisper model size (tiny, base, small, medium, large-v3).
        device:
            "cpu", "cuda", or "auto".
        """
        self.model_size = model_size
        self.device = device
        self._model: WhisperModel | None = None

    def load_model(self) -> None:
        """Load the Whisper model. Call once at startup."""
        logger.info(
            "Loading faster-whisper model '%s' on device '%s'...",
            self.model_size,
            self.device,
        )
        start = time.monotonic()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type="int8",  # Good balance for CPU
        )
        elapsed = time.monotonic() - start
        logger.info("Whisper model loaded in %.1fs", elapsed)

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            raise RuntimeError(
                "Whisper model not loaded. Call load_model() first."
            )
        return self._model

    def _transcribe_sync(self, wav_bytes: bytes) -> list[TranscriptSegment]:
        """
        Synchronous transcription of WAV audio bytes.

        Returns a list of TranscriptSegment with timestamps relative
        to the start of this chunk.
        """
        # Log input diagnostics
        duration_est = len(wav_bytes) / (16000 * 2)  # rough estimate for 16kHz 16-bit mono
        logger.info(
            "Transcribing: wav_size=%d bytes, est_duration=%.1fs",
            len(wav_bytes),
            duration_est,
        )

        segments_iter, info = self.model.transcribe(
            io.BytesIO(wav_bytes),
            beam_size=1,  # Faster, slightly less accurate
            language="en",
            vad_filter=True,  # Filter out silence
            vad_parameters=dict(
                min_silence_duration_ms=1000,  # More lenient: 1s silence before splitting
                speech_pad_ms=400,  # More padding around speech
                threshold=0.3,  # Lower threshold = more sensitive to speech (default 0.5)
            ),
        )

        results = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                results.append(
                    TranscriptSegment(
                        start=seg.start,
                        end=seg.end,
                        text=text,
                    )
                )

        logger.info(
            "Transcription result: %d segments, texts=%s",
            len(results),
            [r.text[:50] for r in results[:5]] if results else "(none)",
        )

        return results

    async def transcribe(self, wav_bytes: bytes) -> list[TranscriptSegment]:
        """
        Transcribe WAV audio bytes asynchronously.

        Runs the actual Whisper inference in a thread pool executor
        so it doesn't block the event loop.

        Parameters
        ----------
        wav_bytes:
            Complete WAV file as bytes.

        Returns
        -------
        List of TranscriptSegment with timestamps and text.
        """
        return await asyncio.to_thread(self._transcribe_sync, wav_bytes)
