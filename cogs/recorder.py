"""
Recorder cog: /record and /stop commands for stage/voice channel recording.

Handles connecting to voice channels, managing the streaming sink,
and dispatching audio chunks to the transcription service.

Uses discord.py with discord-ext-voice-recv for voice receiving.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands
from discord.ext import voice_recv

from config import Config
from services.audio_sink import StreamingSink
from services.session import SessionManager
from services.transcriber import TranscriberService

if TYPE_CHECKING:
    from bot import VoiceRecorderBot

logger = logging.getLogger(__name__)


class RecorderCog(commands.Cog):
    """Cog for recording stage/voice channels and live transcription."""

    def __init__(
        self,
        bot: VoiceRecorderBot,
        session_manager: SessionManager,
        transcriber: TranscriberService,
    ):
        self.bot = bot
        self.sessions = session_manager
        self.transcriber = transcriber

        # guild_id -> VoiceRecvClient
        self._voice_clients: dict[int, voice_recv.VoiceRecvClient] = {}
        # guild_id -> StreamingSink
        self._sinks: dict[int, StreamingSink] = {}
        # Cache: user_id -> display name (for transcript attribution)
        self._user_names: dict[int, str] = {}

    @app_commands.command(
        name="record",
        description="Join your current stage/voice channel and start recording + transcribing.",
    )
    async def record(self, interaction: discord.Interaction) -> None:
        """Start recording the user's current voice/stage channel."""
        assert interaction.guild is not None

        # interaction.user in a guild context is a Member
        member = interaction.user
        assert isinstance(member, discord.Member)

        # Validate the user is in a voice channel
        if not member.voice or not member.voice.channel:
            await interaction.response.send_message(
                "You need to be in a voice or stage channel to use this command.",
                ephemeral=True,
            )
            return

        channel = member.voice.channel
        guild_id = interaction.guild.id

        # Check if already recording
        if self.sessions.has_active_session(guild_id):
            await interaction.response.send_message(
                "Already recording in this server. Use `/stop` to end the current session.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        try:
            # Connect to the voice/stage channel using VoiceRecvClient
            # which has DAVE (E2E encryption) receive support.
            try:
                vc: voice_recv.VoiceRecvClient = await channel.connect(  # type: ignore[assignment]
                    cls=voice_recv.VoiceRecvClient,
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                await interaction.followup.send(
                    "**Failed:** Timed out connecting to the voice channel. "
                    "Please try again.",
                    ephemeral=True,
                )
                return

            self._voice_clients[guild_id] = vc

            # If it's a stage channel, try to become a speaker
            if isinstance(channel, discord.StageChannel):
                me = interaction.guild.me
                if me is not None:
                    try:
                        await me.edit(suppress=False)
                    except discord.HTTPException:
                        logger.warning(
                            "Could not unsuppress in stage channel %s. "
                            "The bot may need stage moderator permissions.",
                            channel.name,
                        )

            # Start a session
            session = self.sessions.start_session(
                guild_id=guild_id,
                channel_id=channel.id,
                channel_name=channel.name,
            )

            # Cache member names for attribution
            for m in channel.members:
                self._user_names[m.id] = m.display_name

            # Create the streaming sink with a chunk callback
            loop = asyncio.get_running_loop()
            sink = StreamingSink(
                chunk_duration=Config.WHISPER_CHUNK_SECONDS,
                on_chunk=self._make_chunk_handler(guild_id),
                loop=loop,
            )
            self._sinks[guild_id] = sink

            # Start listening (voice-recv API)
            vc.listen(
                sink,
                after=self._make_recording_finished_callback(guild_id),
            )

            await interaction.followup.send(
                f"Recording started in **{channel.name}**.\n"
                f"Use `/stop` to end the recording and post the transcript."
            )
            logger.info(
                "Recording started in guild %d, channel '%s'",
                guild_id,
                channel.name,
            )

        except Exception as e:
            logger.exception("Failed to start recording")
            # Clean up on failure
            self.sessions.end_session(guild_id)
            vc_cleanup = self._voice_clients.pop(guild_id, None)
            if vc_cleanup and vc_cleanup.is_connected():
                await vc_cleanup.disconnect(force=True)
            self._sinks.pop(guild_id, None)
            await interaction.followup.send(
                f"Failed to start recording: {e}",
                ephemeral=True,
            )

    @app_commands.command(
        name="stop",
        description="Stop recording and post the transcript.",
    )
    async def stop(self, interaction: discord.Interaction) -> None:
        """Stop the current recording session."""
        assert interaction.guild is not None
        guild_id = interaction.guild.id

        if not self.sessions.has_active_session(guild_id):
            await interaction.response.send_message(
                "Not currently recording in this server.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        try:
            vc = self._voice_clients.get(guild_id)
            sink = self._sinks.get(guild_id)

            # Stop listening
            if vc and vc.is_connected():
                vc.stop_listening()

            # Flush remaining audio from the sink
            if sink:
                remaining = sink.flush_all()
                for user_id, wav_bytes in remaining.items():
                    await self._process_chunk(guild_id, user_id, wav_bytes)

            # End the session
            session = self.sessions.end_session(guild_id)

            # Disconnect from voice
            if vc and vc.is_connected():
                await vc.disconnect()
            self._voice_clients.pop(guild_id, None)
            self._sinks.pop(guild_id, None)

            if session:
                # Post the transcript
                await self._post_transcript(interaction, session)
            else:
                await interaction.followup.send(
                    "Recording stopped (no session data found)."
                )

        except Exception as e:
            logger.exception("Error stopping recording")
            await interaction.followup.send(f"Error stopping recording: {e}")

    def _make_chunk_handler(self, guild_id: int):
        """
        Create an async callback for the streaming sink.

        This closure captures the guild_id so each chunk can be
        attributed to the correct session.
        """

        async def on_chunk(
            user_id: int, wav_bytes: bytes, timestamp: float
        ) -> None:
            await self._process_chunk(
                guild_id, user_id, wav_bytes, timestamp
            )

        return on_chunk

    async def _process_chunk(
        self,
        guild_id: int,
        user_id: int,
        wav_bytes: bytes,
        timestamp: float | None = None,
    ) -> None:
        """
        Transcribe an audio chunk and add it to the session transcript.

        Note: wav_bytes is already in Whisper format (16kHz mono) from the sink.
        """
        session = self.sessions.get_active_session(guild_id)
        if not session:
            return

        try:
            segments = await self.transcriber.transcribe(wav_bytes)
            speaker_name = self._user_names.get(user_id, f"User-{user_id}")

            for seg in segments:
                # Calculate the absolute timestamp for this segment
                seg_timestamp = (timestamp or 0) + seg.start
                session.add_entry(
                    speaker=speaker_name,
                    speaker_id=user_id,
                    text=seg.text,
                    timestamp=seg_timestamp if timestamp else None,
                )

        except Exception:
            logger.exception(
                "Failed to transcribe chunk for user %d in guild %d",
                user_id,
                guild_id,
            )

    def _make_recording_finished_callback(self, guild_id: int):
        """
        Create a callback for when listening finishes or errors out.
        """

        def callback(error: Exception | None) -> None:
            if error:
                error_name = type(error).__name__
                logger.error(
                    "Recording finished with error in guild %d: %s: %s",
                    guild_id,
                    error_name,
                    error,
                )
            else:
                logger.info(
                    "Recording finished successfully for guild %d", guild_id
                )

        return callback

    async def _post_transcript(
        self,
        interaction: discord.Interaction,
        session,
    ) -> None:
        """Post the completed transcript to the configured channel or current channel."""
        transcript_text = session.get_full_transcript()

        # Build an embed for the summary
        embed = discord.Embed(
            title=f"Transcript: {session.channel_name}",
            color=discord.Color.blurple(),
            description=(
                f"**Duration:** {session.duration_str}\n"
                f"**Entries:** {len(session.transcript)}"
            ),
        )

        # Try to post to the configured transcript channel
        target_channel = None
        if Config.TRANSCRIPT_CHANNEL_ID:
            target_channel = self.bot.get_channel(Config.TRANSCRIPT_CHANNEL_ID)

        if target_channel is None:
            target_channel = interaction.channel

        # Discord messages have a 2000 char limit; split if needed
        if hasattr(target_channel, "send"):
            await target_channel.send(embed=embed)  # type: ignore[union-attr]

            if transcript_text == "(No transcript available)":
                await target_channel.send(  # type: ignore[union-attr]
                    "No speech was detected during this session."
                )
            else:
                # Split transcript into chunks that fit in Discord messages
                chunks = _split_message(transcript_text, max_len=1900)
                for i, chunk in enumerate(chunks):
                    prefix = f"**Transcript ({i + 1}/{len(chunks)}):**\n" if len(chunks) > 1 else ""
                    await target_channel.send(f"{prefix}```\n{chunk}\n```")  # type: ignore[union-attr]

        await interaction.followup.send("Recording stopped and transcript posted.")

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Track user names when they join the recorded channel."""
        guild_id = member.guild.id
        session = self.sessions.get_active_session(guild_id)
        if not session:
            return

        # If someone joins the channel we're recording, cache their name
        if after.channel and after.channel.id == session.channel_id:
            self._user_names[member.id] = member.display_name


def _split_message(text: str, max_len: int = 1900) -> list[str]:
    """Split text into chunks that fit within Discord's message limit."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    lines = text.split("\n")
    current = ""

    for line in lines:
        if len(current) + len(line) + 1 > max_len:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current)

    return chunks


async def setup(bot: VoiceRecorderBot) -> None:
    """Called by discord.py when loading this cog as an extension."""
    session_manager = bot._session_manager
    transcriber = bot._transcriber
    await bot.add_cog(RecorderCog(bot, session_manager, transcriber))
