# AGENTS.md

This repository provides a Python 3.11 CLI for YouTube/local ASR transcription with manual backend switching and unified transcript outputs.

## Essentials
- Runtime/package manager: `python3.11` + `pip` in virtualenv
- Main command: `python transcribe_youtube.py <input>`
- Long inputs are auto-chunked by default (`--chunk-mode auto`)
- Tests: `python -m pytest -q`
- Non-standard dependency split:
  - Core: `requirements.txt`
  - Optional VibeVoice stack: `requirements-vibevoice.txt`

## Detailed Guidance
- Workflow and architecture: [docs/agents/transcription-workflow.md](docs/agents/transcription-workflow.md)
- Backend constraints/capabilities: [docs/agents/backend-capabilities.md](docs/agents/backend-capabilities.md)
- Command cookbook by scenario: [docs/agents/command-recipes.md](docs/agents/command-recipes.md)
- Test/release process: [docs/agents/testing-and-release.md](docs/agents/testing-and-release.md)
- Troubleshooting: [docs/agents/troubleshooting.md](docs/agents/troubleshooting.md)
