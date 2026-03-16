"""
AWS Bedrock client for Claude-based summarization and Q&A.

Uses the Converse API which is the recommended approach for Bedrock.
"""

from __future__ import annotations

import asyncio
import logging
import os

import boto3

logger = logging.getLogger(__name__)

# Maximum characters of transcript to send to Claude.
MAX_TRANSCRIPT_CHARS = 150_000

SUMMARIZE_SYSTEM_PROMPT = """\
You are a meeting notes assistant. You summarize Discord stage channel \
sessions clearly and concisely. Your summaries should:

- Start with a brief one-paragraph overview
- List the main topics discussed as bullet points
- Under each topic, include key points, decisions, and action items
- Note any questions that were raised but left unanswered
- End with a list of participants who spoke

Use clear, professional language. Be concise but don't omit important details.\
"""

ASK_SYSTEM_PROMPT = """\
You are a helpful assistant with access to a transcript from a Discord stage \
channel session. Answer the user's question based ONLY on information found \
in the transcript. If the answer is not in the transcript, say so clearly. \
When referencing what someone said, attribute it to them by name.\
"""


class BedrockService:
    """
    Client for invoking Claude models on AWS Bedrock via the Converse API.

    Authenticates using a Bedrock API key (bearer token).
    See: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html
    """

    def __init__(self, region: str, model_id: str, api_key: str):
        self.region = region
        self.model_id = model_id
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            # Set the bearer token env var that boto3 uses for Bedrock API key auth
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.api_key
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
            )
        return self._client

    def _converse(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """
        Invoke Claude on Bedrock using the Converse API and return the response text.
        """
        logger.debug(
            "Invoking Bedrock model %s (message length: %d chars)",
            self.model_id,
            len(user_message),
        )

        response = self.client.converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_message}],
                }
            ],
            system=[{"text": system_prompt}],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        )

        text = response["output"]["message"]["content"][0]["text"]
        stop_reason = response.get("stopReason", "unknown")
        logger.debug(
            "Bedrock response received (%d chars, stop_reason=%s)",
            len(text),
            stop_reason,
        )
        return text

    async def summarize(
        self, transcript: str, context: str = ""
    ) -> str:
        """
        Summarize a transcript using Claude.

        Parameters
        ----------
        transcript:
            The formatted transcript text.
        context:
            Optional additional context (e.g., channel name, topic).

        Returns
        -------
        The summary text.
        """
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = transcript[:MAX_TRANSCRIPT_CHARS] + (
                "\n\n[... transcript truncated due to length ...]"
            )

        user_msg = "Please summarize the following stage session transcript."
        if context:
            user_msg += f"\n\nContext: {context}"
        user_msg += f"\n\n--- TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---"

        return await asyncio.to_thread(
            self._converse,
            SUMMARIZE_SYSTEM_PROMPT,
            user_msg,
        )

    async def ask(self, transcript: str, question: str) -> str:
        """
        Answer a question about the transcript using Claude.

        Parameters
        ----------
        transcript:
            The formatted transcript text.
        question:
            The user's question.

        Returns
        -------
        The answer text.
        """
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = transcript[:MAX_TRANSCRIPT_CHARS] + (
                "\n\n[... transcript truncated due to length ...]"
            )

        user_msg = (
            f"Based on the following transcript, please answer this question:\n\n"
            f"**Question:** {question}\n\n"
            f"--- TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---"
        )

        return await asyncio.to_thread(
            self._converse,
            ASK_SYSTEM_PROMPT,
            user_msg,
        )
