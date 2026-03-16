"""
Custom PCM streaming sink for discord-ext-voice-recv.

Instead of waiting until stop_listening() to process audio, this sink
dispatches audio chunks to a callback as they accumulate, enabling
near-real-time transcription while recording is ongoing.

Uses the AudioSink base class from discord-ext-voice-recv (rdphillips7 fork).
"""

from __future__ import annotations

import array
import asyncio
import io
import logging
import struct
import time
import wave
from collections import defaultdict
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import discord
from discord.ext.voice_recv import AudioSink, VoiceData

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Type alias matching voice-recv's convention
User = Union[discord.Member, discord.User]

# Discord voice sends 48kHz, 16-bit, stereo PCM
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # bytes per sample (16-bit)
FRAME_SIZE = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH  # bytes per second

# Whisper target format
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1


class StreamingSink(AudioSink):
    """
    A sink that streams audio chunks for real-time transcription.

    Audio data is accumulated per-user. Once a user's buffer reaches
    `chunk_duration` seconds of audio, the chunk is dispatched to the
    `on_chunk` async callback for transcription.

    Parameters
    ----------
    chunk_duration:
        Seconds of audio to buffer before dispatching a chunk.
    on_chunk:
        Async callback: (user_id: int, wav_bytes: bytes, timestamp: float) -> None
    loop:
        The running asyncio event loop (needed to schedule coroutines from
        the recording thread).
    """

    def __init__(
        self,
        chunk_duration: int = 10,
        on_chunk: Callable[[int, bytes, float], Coroutine[Any, Any, None]] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__()  # No destination — this is an endpoint sink
        self.chunk_duration = chunk_duration
        self.on_chunk = on_chunk
        self.loop = loop

        # Per-user PCM buffers: user_id -> bytearray
        self._buffers: dict[int, bytearray] = defaultdict(bytearray)
        # Track when each user's current chunk started
        self._chunk_start_times: dict[int, float] = {}

        # How many bytes = one chunk of audio
        self._chunk_bytes = FRAME_SIZE * chunk_duration

        # Diagnostics
        self._write_count: int = 0
        self._write_logged: bool = False

    def wants_opus(self) -> bool:
        """We need decoded PCM for Whisper transcription."""
        return False

    def write(self, user: Optional[User], data: VoiceData) -> None:
        """
        Called by voice-recv's AudioReader whenever a decoded audio
        packet is received from a user.

        Parameters
        ----------
        user:
            The Discord user who sent this audio, or None if SSRC
            hasn't been mapped yet.
        data:
            VoiceData with .pcm (48kHz stereo 16-bit PCM bytes).
        """
        self._write_count += 1
        if not self._write_logged and self._write_count == 1:
            self._write_logged = True
            logger.info(
                "StreamingSink.write called: user=%s pcm_len=%d",
                getattr(user, "id", None) if user is not None else None,
                len(data.pcm) if data.pcm else 0,
            )

        # Extract user ID — skip if we can't identify the speaker
        if user is None:
            return
        user_id: int = user.id
        pcm_data: bytes = data.pcm

        if not pcm_data or len(pcm_data) == 0:
            return

        # Skip frames that are all zeros (failed decryption produces
        # empty PCM which gets zero-padded). These corrupt the audio
        # buffer and cause Whisper's VAD to reject the chunk as silence.
        if _is_zero_pcm(pcm_data):
            return

        if user_id not in self._chunk_start_times:
            self._chunk_start_times[user_id] = time.time()

        self._buffers[user_id].extend(pcm_data)

        buf_len = len(self._buffers[user_id])
        if self._write_count in (1, 50, 200):
            logger.info(
                "StreamingSink buffer: user=%s pcm_len=%d total_buf=%d "
                "chunk_target=%d",
                user_id,
                len(pcm_data),
                buf_len,
                self._chunk_bytes,
            )

        # Check if we have enough data for a chunk
        while len(self._buffers[user_id]) >= self._chunk_bytes:
            chunk_pcm = bytes(self._buffers[user_id][: self._chunk_bytes])
            self._buffers[user_id] = self._buffers[user_id][self._chunk_bytes :]

            chunk_ts = self._chunk_start_times.get(user_id, time.time())
            self._chunk_start_times[user_id] = time.time()

            if self.on_chunk and self.loop:
                logger.info(
                    "Dispatching chunk for user=%s (%.1f seconds of audio, %d bytes PCM)",
                    user_id,
                    self.chunk_duration,
                    len(chunk_pcm),
                )
                # Convert to 16kHz mono WAV optimized for Whisper
                wav_bytes = pcm_to_whisper_wav(chunk_pcm)
                asyncio.run_coroutine_threadsafe(
                    self.on_chunk(user_id, wav_bytes, chunk_ts),
                    self.loop,
                )

    def flush_all(self) -> dict[int, bytes]:
        """
        Flush any remaining buffered audio for all users.

        Returns a dict of user_id -> wav_bytes for any remaining data.
        Called when recording stops to capture the tail end of audio.

        Note: We snapshot the buffers first because write() may still
        be called from the voice-recv thread (adding keys to the
        defaultdict), causing a RuntimeError during iteration.
        """
        # Snapshot and clear atomically to avoid dict-changed-size errors
        buffers = dict(self._buffers)
        self._buffers.clear()
        self._chunk_start_times.clear()

        results = {}
        for user_id, buf in buffers.items():
            if len(buf) > 0:
                results[user_id] = pcm_to_whisper_wav(bytes(buf))
        return results

    def cleanup(self) -> None:
        """Called by voice-recv when listening stops. Clean up buffers."""
        self._buffers.clear()
        self._chunk_start_times.clear()


def _is_zero_pcm(pcm_data: bytes, threshold: int = 100) -> bool:
    """
    Check if a PCM frame is effectively all zeros / silence.

    Uses a fast check: examine first 64 samples. If all are below
    threshold, treat the whole frame as zero.
    """
    if not pcm_data:
        return True

    # Check up to first 64 samples (128 bytes for 16-bit)
    check_bytes = min(len(pcm_data), 128)
    num_samples = check_bytes // SAMPLE_WIDTH
    if num_samples == 0:
        return True

    for i in range(num_samples):
        sample = struct.unpack_from("<h", pcm_data, i * SAMPLE_WIDTH)[0]
        if abs(sample) > threshold:
            return False
    return True


def pcm_to_wav(pcm_data: bytes) -> bytes:
    """Convert raw PCM data (48kHz, 16-bit, stereo) to WAV format in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    buf.seek(0)
    return buf.read()


def pcm_to_whisper_wav(pcm_data: bytes) -> bytes:
    """
    Convert raw PCM (48kHz, 16-bit, stereo) to mono 16kHz WAV for Whisper.

    Steps:
    1. Stereo -> Mono (average L+R channels)
    2. 48kHz -> 16kHz (simple decimation by factor 3)
    """
    # Step 1: Stereo to Mono — average left and right channels
    samples = array.array("h")
    samples.frombytes(pcm_data)

    num_stereo_frames = len(samples) // 2
    mono_samples = array.array("h", [0] * num_stereo_frames)
    for i in range(num_stereo_frames):
        left = samples[i * 2]
        right = samples[i * 2 + 1]
        mono_samples[i] = (left + right) // 2

    # Step 2: 48kHz -> 16kHz (decimate by 3)
    decimated = array.array("h", mono_samples[::3])

    # Write as WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(WHISPER_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(WHISPER_SAMPLE_RATE)
        wf.writeframes(decimated.tobytes())
    buf.seek(0)
    return buf.read()


def detect_silence(
    pcm_data: bytes,
    threshold: int = 500,
    min_silence_ratio: float = 0.95,
) -> bool:
    """
    Check if a PCM chunk is mostly silence.

    Parameters
    ----------
    pcm_data:
        Raw 16-bit PCM audio data.
    threshold:
        Amplitude below which a sample is considered silent.
    min_silence_ratio:
        Proportion of samples that must be silent to consider the
        entire chunk as silence.

    Returns
    -------
    True if the chunk is considered silence.
    """
    if not pcm_data:
        return True

    num_samples = len(pcm_data) // SAMPLE_WIDTH
    if num_samples == 0:
        return True

    silent_count = 0
    for i in range(num_samples):
        sample = struct.unpack_from("<h", pcm_data, i * SAMPLE_WIDTH)[0]
        if abs(sample) < threshold:
            silent_count += 1

    return (silent_count / num_samples) >= min_silence_ratio
