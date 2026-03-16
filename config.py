import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Bot configuration loaded from environment variables."""

    # Discord
    DISCORD_TOKEN: str = os.environ.get("DISCORD_TOKEN", "")
    GUILD_IDS: list[int] = [
        int(gid.strip())
        for gid in os.environ.get("GUILD_IDS", "").split(",")
        if gid.strip()
    ]

    # Channel where completed transcripts are posted
    TRANSCRIPT_CHANNEL_ID: int = int(
        os.environ.get("TRANSCRIPT_CHANNEL_ID", "0")
    )

    # AWS Bedrock
    AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
    BEDROCK_API_KEY: str = os.environ.get("BEDROCK_API_KEY", "")
    BEDROCK_MODEL_ID: str = os.environ.get(
        "BEDROCK_MODEL_ID",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    )

    # Whisper
    WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")
    # How many seconds of audio to buffer before sending to Whisper
    WHISPER_CHUNK_SECONDS: int = int(
        os.environ.get("WHISPER_CHUNK_SECONDS", "10")
    )

    # Audio settings (Discord sends 48kHz 16-bit stereo PCM)
    SAMPLE_RATE: int = 48000
    CHANNELS: int = 2
    SAMPLE_WIDTH: int = 2  # 16-bit = 2 bytes

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of missing required config values."""
        errors = []
        if not cls.DISCORD_TOKEN:
            errors.append("DISCORD_TOKEN is required")
        if not cls.TRANSCRIPT_CHANNEL_ID:
            errors.append("TRANSCRIPT_CHANNEL_ID is required")
        if not cls.BEDROCK_API_KEY:
            errors.append("BEDROCK_API_KEY is required")
        return errors
