# Voice Recorder Bot

Discord stage transcription and summarization bot using discord.py, Whisper, and AWS Bedrock.

Joins Discord voice/stage channels, transcribes speech in real-time with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and generates summaries via AWS Bedrock (Claude).

## Prerequisites

- Python 3.11+ (3.13 recommended)
- A [Discord bot token](https://discord.com/developers/applications) with voice and message content intents enabled
- An [AWS Bedrock](https://console.aws.amazon.com/bedrock) API key with access to Claude
- System packages: `ffmpeg`, `libopus`, `libsodium` (for Discord voice support)

## Configuration

Copy the example env file and fill in the required values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | Yes | | Discord bot token |
| `TRANSCRIPT_CHANNEL_ID` | Yes | | Channel ID where transcripts are posted |
| `BEDROCK_API_KEY` | Yes | | AWS Bedrock API key |
| `GUILD_IDS` | No | | Comma-separated guild IDs for faster slash command sync |
| `AWS_REGION` | No | `us-east-1` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | No | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | Bedrock model ID |
| `WHISPER_MODEL` | No | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_CHUNK_SECONDS` | No | `10` | Seconds of audio to buffer before transcribing |

## Running Locally

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) handles Python version management and dependency installation automatically.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the bot (installs dependencies automatically on first run)
uv run python bot.py
```

### With pip

Make sure you have the required system packages installed first:

```bash
# macOS
brew install ffmpeg opus libsodium

# Ubuntu / Debian
sudo apt-get install ffmpeg libopus0 libsodium23
```

Then install Python dependencies and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python bot.py
```

On first startup, the Whisper model will be downloaded and cached to `~/.cache/huggingface` (~150 MB for `base`, ~3 GB for `large-v3`).

## Docker Deployment

### Build the image

```bash
docker build -t voicerecorder .
```

### Run the container

```bash
docker run -d \
  --name voicerecorder \
  --env-file .env \
  -v whisper-cache:/root/.cache/huggingface \
  --restart unless-stopped \
  voicerecorder
```

The `whisper-cache` volume persists the Whisper model across container restarts so it only downloads once.

### Managing the container

```bash
# View logs
docker logs -f voicerecorder

# Stop
docker stop voicerecorder

# Restart
docker restart voicerecorder

# Remove and rebuild
docker rm -f voicerecorder
docker build -t voicerecorder .
docker run -d --name voicerecorder --env-file .env -v whisper-cache:/root/.cache/huggingface --restart unless-stopped voicerecorder
```

### Environment variables

You can pass environment variables directly instead of using `--env-file`:

```bash
docker run -d \
  --name voicerecorder \
  -e DISCORD_TOKEN=your-token \
  -e TRANSCRIPT_CHANNEL_ID=123456789 \
  -e BEDROCK_API_KEY=your-key \
  -v whisper-cache:/root/.cache/huggingface \
  --restart unless-stopped \
  voicerecorder
```

### Whisper model selection

Larger models are more accurate but use more memory and CPU:

| Model | Size on Disk | RAM Usage | Relative Speed |
|---|---|---|---|
| `tiny` | ~75 MB | ~1 GB | Fastest |
| `base` | ~150 MB | ~1 GB | Fast |
| `small` | ~500 MB | ~2 GB | Moderate |
| `medium` | ~1.5 GB | ~5 GB | Slow |
| `large-v3` | ~3 GB | ~10 GB | Slowest |

Set via the `WHISPER_MODEL` environment variable. The default (`base`) is a good balance of accuracy and speed for real-time transcription on CPU.

## Project Structure

```
bot.py                          # Entry point
config.py                       # Environment variable loading and validation
cogs/
  recorder.py                   # Voice recording slash commands
  ai.py                         # AI summarization slash commands
services/
  audio_sink.py                 # In-memory audio capture and PCM-to-WAV conversion
  transcriber.py                # Whisper transcription service
  session.py                    # In-memory transcript session management
  bedrock.py                    # AWS Bedrock client for Claude
  voice_recv_patch.py           # Monkey-patch for DAVE decryption edge cases
```
