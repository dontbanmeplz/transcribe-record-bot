"""
Session and transcript state management.

Tracks recording sessions per guild and stores transcripts in memory.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptEntry:
    """A single line in the transcript."""

    timestamp: float  # Unix timestamp
    speaker: str  # Display name of the speaker
    speaker_id: int  # Discord user ID
    text: str

    @property
    def time_str(self) -> str:
        """Format timestamp as HH:MM:SS."""
        dt = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        return dt.strftime("%H:%M:%S")

    def __str__(self) -> str:
        return f"[{self.time_str}] {self.speaker}: {self.text}"


@dataclass
class Session:
    """
    Represents an active or completed recording session for a guild.

    Stores the running transcript and metadata about the session.
    """

    guild_id: int
    channel_id: int
    channel_name: str
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    transcript: list[TranscriptEntry] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or time.time()
        return end - self.started_at

    @property
    def duration_str(self) -> str:
        """Human-readable duration string."""
        total = int(self.duration_seconds)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def add_entry(
        self,
        speaker: str,
        speaker_id: int,
        text: str,
        timestamp: float | None = None,
    ) -> None:
        """Add a transcript entry."""
        entry = TranscriptEntry(
            timestamp=timestamp or time.time(),
            speaker=speaker,
            speaker_id=speaker_id,
            text=text,
        )
        self.transcript.append(entry)
        logger.debug("Transcript entry: %s", entry)

    def get_full_transcript(self) -> str:
        """Return the full transcript as a formatted string."""
        if not self.transcript:
            return "(No transcript available)"
        return "\n".join(str(entry) for entry in self.transcript)

    def get_transcript_for_llm(self) -> str:
        """
        Return the transcript formatted for LLM consumption.
        Includes speaker labels and timestamps.
        """
        if not self.transcript:
            return "(Empty transcript - no speech was detected)"

        lines = []
        for entry in self.transcript:
            lines.append(f"[{entry.time_str}] {entry.speaker}: {entry.text}")
        return "\n".join(lines)

    def end(self) -> None:
        """Mark the session as ended."""
        self.ended_at = time.time()


class SessionManager:
    """
    Manages recording sessions across guilds.

    Keeps track of active sessions and retains the last completed
    session per guild for post-session summarization.
    """

    def __init__(self):
        # guild_id -> active Session
        self._active: dict[int, Session] = {}
        # guild_id -> last completed Session
        self._last_completed: dict[int, Session] = {}

    def start_session(
        self, guild_id: int, channel_id: int, channel_name: str
    ) -> Session:
        """Start a new recording session for a guild."""
        if guild_id in self._active:
            raise RuntimeError(
                f"Session already active for guild {guild_id}"
            )

        session = Session(
            guild_id=guild_id,
            channel_id=channel_id,
            channel_name=channel_name,
        )
        self._active[guild_id] = session
        logger.info(
            "Started session for guild %d in channel '%s'",
            guild_id,
            channel_name,
        )
        return session

    def get_active_session(self, guild_id: int) -> Session | None:
        """Get the active session for a guild, if any."""
        return self._active.get(guild_id)

    def get_session(self, guild_id: int) -> Session | None:
        """
        Get the current session for a guild (active or last completed).
        Prefers the active session if one exists.
        """
        return self._active.get(guild_id) or self._last_completed.get(
            guild_id
        )

    def end_session(self, guild_id: int) -> Session | None:
        """End the active session for a guild and archive it."""
        session = self._active.pop(guild_id, None)
        if session:
            session.end()
            self._last_completed[guild_id] = session
            logger.info(
                "Ended session for guild %d (duration: %s, entries: %d)",
                guild_id,
                session.duration_str,
                len(session.transcript),
            )
        return session

    def has_active_session(self, guild_id: int) -> bool:
        return guild_id in self._active
