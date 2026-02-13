# Testing and Release / Pruebas y Release

## EN
Local checks:
1. `python -m py_compile transcribe_youtube.py`
2. `python -m pytest -q`

Release checklist:
1. Validate one run per backend with explicit `--backend` (mlx-whisper, openai).
2. Validate `txt,srt,json` outputs.
3. Validate OpenAI capability errors for non-timestamp models + `srt`.
4. Validate auto-chunk merge path for long audio.
5. Confirm no automatic backend fallback behavior.
6. Validate mlx-whisper on Apple Silicon with `--backend mlx-whisper`.

## ES
Checks locales:
1. `python -m py_compile transcribe_youtube.py`
2. `python -m pytest -q`

Checklist de release:
1. Validar una ejecucion por backend con `--backend` explicito (mlx-whisper, openai).
2. Validar salidas `txt,srt,json`.
3. Validar errores de capacidad en OpenAI cuando se pide `srt` sin timestamps.
4. Validar el flujo de auto-chunk + fusion para audio largo.
5. Confirmar que no existe fallback automatico de backend.
6. Validar mlx-whisper en Apple Silicon con `--backend mlx-whisper`.
