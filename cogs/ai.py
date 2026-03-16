"""
AI cog: /summarize and /ask commands powered by AWS Bedrock (Claude).

Uses discord.py slash commands via app_commands.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from services.bedrock import BedrockService
from services.session import SessionManager

if TYPE_CHECKING:
    from bot import VoiceRecorderBot

logger = logging.getLogger(__name__)

# Discord embed description limit
EMBED_DESC_LIMIT = 4096
# Discord message limit
MESSAGE_LIMIT = 2000


class AICog(commands.Cog):
    """Cog for AI-powered summarization and Q&A on transcripts."""

    def __init__(
        self,
        bot: VoiceRecorderBot,
        session_manager: SessionManager,
        bedrock: BedrockService,
    ):
        self.bot = bot
        self.sessions = session_manager
        self.bedrock = bedrock

    @app_commands.command(
        name="summarize",
        description="Summarize the current or last recording session.",
    )
    async def summarize(self, interaction: discord.Interaction) -> None:
        """Generate a summary of the current or last session transcript."""
        assert interaction.guild is not None
        session = self.sessions.get_session(interaction.guild.id)

        if not session:
            await interaction.response.send_message(
                "No recording session found. Use `/record` to start one first.",
                ephemeral=True,
            )
            return

        transcript = session.get_transcript_for_llm()
        if "(Empty transcript" in transcript:
            await interaction.response.send_message(
                "The transcript is empty -- no speech was detected.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        try:
            context = (
                f"Channel: {session.channel_name}, "
                f"Duration: {session.duration_str}, "
                f"Status: {'ongoing' if session.is_active else 'completed'}"
            )
            summary = await self.bedrock.summarize(
                transcript=transcript,
                context=context,
            )

            # Post as an embed, splitting if needed
            status = "Live Session" if session.is_active else "Completed Session"
            embed = discord.Embed(
                title=f"Summary: {session.channel_name}",
                color=discord.Color.green(),
            )
            embed.set_footer(text=f"{status} | Duration: {session.duration_str}")

            # Embed description has a 4096 char limit
            if len(summary) <= EMBED_DESC_LIMIT:
                embed.description = summary
                await interaction.followup.send(embed=embed)
            else:
                # Post embed as header, then summary as follow-up messages
                embed.description = summary[:EMBED_DESC_LIMIT]
                await interaction.followup.send(embed=embed)

                remaining = summary[EMBED_DESC_LIMIT:]
                for chunk in _split_text(remaining, MESSAGE_LIMIT):
                    await interaction.followup.send(chunk)

        except Exception as e:
            logger.exception("Failed to generate summary")
            await interaction.followup.send(
                f"Failed to generate summary: {e}",
                ephemeral=True,
            )

    @app_commands.command(
        name="ask",
        description="Ask a question about the current or last recording session.",
    )
    @app_commands.describe(question="The question to ask about the transcript")
    async def ask(
        self,
        interaction: discord.Interaction,
        question: str,
    ) -> None:
        """Answer a question based on the session transcript."""
        assert interaction.guild is not None
        session = self.sessions.get_session(interaction.guild.id)

        if not session:
            await interaction.response.send_message(
                "No recording session found. Use `/record` to start one first.",
                ephemeral=True,
            )
            return

        transcript = session.get_transcript_for_llm()
        if "(Empty transcript" in transcript:
            await interaction.response.send_message(
                "The transcript is empty -- no speech was detected to search.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        try:
            answer = await self.bedrock.ask(
                transcript=transcript,
                question=question,
            )

            embed = discord.Embed(
                title="Answer",
                color=discord.Color.blue(),
            )
            embed.add_field(
                name="Question",
                value=question[:1024],  # Field value limit
                inline=False,
            )

            if len(answer) <= EMBED_DESC_LIMIT:
                embed.description = answer
                await interaction.followup.send(embed=embed)
            else:
                embed.description = answer[:EMBED_DESC_LIMIT]
                await interaction.followup.send(embed=embed)

                remaining = answer[EMBED_DESC_LIMIT:]
                for chunk in _split_text(remaining, MESSAGE_LIMIT):
                    await interaction.followup.send(chunk)

        except Exception as e:
            logger.exception("Failed to answer question")
            await interaction.followup.send(
                f"Failed to generate answer: {e}",
                ephemeral=True,
            )


def _split_text(text: str, max_len: int) -> list[str]:
    """Split text into chunks respecting the character limit."""
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


async def setup(bot: VoiceRecorderBot) -> None:
    """Called by discord.py when loading this cog as an extension."""
    session_manager = bot._session_manager
    bedrock = bot._bedrock
    await bot.add_cog(AICog(bot, session_manager, bedrock))
