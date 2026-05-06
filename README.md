# LiveKit Voice Agent

![alt text](<SUPER NATURE DESIGN.gif>)

Realtime voice assistant built with LiveKit Agents (Python), provider failover, tool calling, and MCP integration.

## Features

- Voice pipeline: `VAD -> STT -> LLM -> TTS`
- Semantic turn detection: `MultilingualModel`
- Fallback adapters:
  - LLM: `openai/gpt-4.1-mini` -> `google/gemini-2.5-flash`
  - STT: `deepgram/nova-3` -> `assemblyai/universal-streaming`
  - TTS: `cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc` -> `inworld/inworld-tts-1`
- Preemptive generation for faster responses
- Metrics collection and usage summary logging
- Function tool: `lookup_weather(location)`
- MCP server integration: `https://shayne.app/sse`

## Run

```powershell
uv sync
.\run-console.ps1
```

First run only (if model files are missing):

```powershell
uv run agent.py download-files
```

## Environment

Required `.env` keys:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `DEEPGRAM_API_KEY`
- `OPENAI_API_KEY`
- `CARTESIA_API_KEY`

Optional for configured fallbacks:

- `ASSEMBLYAI_API_KEY`
- `INWORLD_API_KEY`
- `GROQ_API_KEY`

## Files

- `agent.py`: agent behavior, tools, MCP, metrics, and session config
- `run-console.ps1`: UTF-8 Windows launcher for console mode
- `pyproject.toml`: dependencies
- `livekit.toml`: LiveKit project metadata
