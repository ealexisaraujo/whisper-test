# Troubleshooting / Resolución de Problemas

## EN
- `ffmpeg not found`: install ffmpeg and ensure `ffmpeg`/`ffprobe` are in `PATH`.
- VibeVoice import errors: install `requirements-vibevoice.txt`.
- VibeVoice OOM on macOS: switch to `--backend openai` or `--device cpu`.
- OpenAI auth errors: set `OPENAI_API_KEY`.
- OpenAI + `srt` error: use `--model whisper-1` or `--model gpt-4o-transcribe-diarize`.
- Disable chunking if needed: `--chunk-mode off`.
- Force chunking for long jobs: `--chunk-mode force`.
- Wrong YouTube video downloaded: avoid escaped URL chars (`watch\\?v\\=...`). Use `watch?v=...`.

## ES
- `ffmpeg not found`: instala ffmpeg y verifica `ffmpeg`/`ffprobe` en `PATH`.
- Error importando VibeVoice: instala `requirements-vibevoice.txt`.
- OOM con VibeVoice en macOS: cambia a `--backend openai` o `--device cpu`.
- Error de autenticación OpenAI: configura `OPENAI_API_KEY`.
- Error OpenAI + `srt`: usa `--model whisper-1` o `--model gpt-4o-transcribe-diarize`.
- Desactiva chunking si es necesario: `--chunk-mode off`.
- Fuerza chunking para trabajos largos: `--chunk-mode force`.
- Se descargó un video incorrecto de YouTube: evita URL con escapes (`watch\\?v\\=...`). Usa `watch?v=...`.
