"""
Discord Stage Transcription & Summarization Bot

Entry point: loads configuration, initializes services, and starts the bot.

Usage:
    uv run python bot.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

import discord
from discord.ext import commands

from config import Config
from services.bedrock import BedrockService
from services.session import SessionManager
from services.transcriber import TranscriberService
from services.voice_recv_patch import apply as apply_voice_recv_patch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Silence noisy libraries
logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
# Suppress noisy voice-recv logs (RTCP SenderReport, WS payload extra keys)
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)

logger = logging.getLogger("voicerecorder")


class VoiceRecorderBot(commands.Bot):
    """Custom Bot subclass that initializes services and loads cogs."""

    def __init__(
        self,
        session_manager: SessionManager,
        transcriber: TranscriberService,
        bedrock: BedrockService,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Attach services so cogs can access them
        self._session_manager = session_manager
        self._transcriber = transcriber
        self._bedrock = bedrock

    async def setup_hook(self) -> None:
        """Called once when the bot starts, before on_ready."""
        await self.load_extension("cogs.recorder")
        await self.load_extension("cogs.ai")
        logger.info("Cogs loaded: recorder, ai")

        # Sync slash commands globally (or to specific guilds)
        if Config.GUILD_IDS:
            for guild_id in Config.GUILD_IDS:
                guild = discord.Object(id=guild_id)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                logger.info("Synced commands to guild %d", guild_id)
        else:
            await self.tree.sync()
            logger.info("Synced commands globally")


def main() -> None:
    # ------------------------------------------------------------------
    # Validate configuration
    # ------------------------------------------------------------------
    errors = Config.validate()
    if errors:
        for err in errors:
            logger.error("Config error: %s", err)
        logger.error("Copy .env.example to .env and fill in the required values.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Initialize services
    # ------------------------------------------------------------------
    logger.info("Initializing services...")

    # Apply safety patch for DAVE decryption failures in voice-recv.
    # Without this, unencrypted packets during DAVE handshake transitions
    # crash the entire audio receive pipeline.
    apply_voice_recv_patch()

    # Session manager (in-memory transcript storage)
    session_manager = SessionManager()

    # Whisper transcription service
    transcriber = TranscriberService(
        model_size=Config.WHISPER_MODEL,
        device="auto",
    )
    transcriber.load_model()

    # AWS Bedrock service
    bedrock = BedrockService(
        region=Config.AWS_REGION,
        model_id=Config.BEDROCK_MODEL_ID,
        api_key=Config.BEDROCK_API_KEY,
    )

    # ------------------------------------------------------------------
    # Create the bot
    # ------------------------------------------------------------------
    intents = discord.Intents.default()
    intents.voice_states = True
    intents.guilds = True
    intents.message_content = True

    bot = VoiceRecorderBot(
        session_manager=session_manager,
        transcriber=transcriber,
        bedrock=bedrock,
        command_prefix="!",  # Required by commands.Bot, but we use slash commands
        intents=intents,
    )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    @bot.event
    async def on_ready():
        logger.info("Bot is ready: %s (ID: %s)", bot.user, bot.user.id if bot.user else "?")
        logger.info("Connected to %d guild(s)", len(bot.guilds))
        logger.info(
            "Whisper model: %s | Bedrock model: %s",
            Config.WHISPER_MODEL,
            Config.BEDROCK_MODEL_ID,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    logger.info("Starting bot...")
    bot.run(Config.DISCORD_TOKEN)


if __name__ == "__main__":
    main()
