# Troubleshooting / Resolucion de Problemas

## EN
- `ffmpeg not found`: install ffmpeg and ensure `ffmpeg`/`ffprobe` are in `PATH`.
- OpenAI auth errors: set `OPENAI_API_KEY`.
- OpenAI + `srt` error: use `--model whisper-1` or `--model gpt-4o-transcribe-diarize`.
- Disable chunking if needed: `--chunk-mode off`.
- Force chunking for long jobs: `--chunk-mode force`.
- Wrong YouTube video downloaded: avoid escaped URL chars (`watch\\?v\\=...`). Use `watch?v=...`.
- Need stronger progress visibility: use `--log-level DEBUG --progress-log-seconds 10 --log-file ./transcribe_debug.log`.
- MLX Whisper import errors: install with `pip install mlx-whisper`. Requires Apple Silicon (M1+).
- MLX Whisper model download slow: models are fetched from Hugging Face on first use; subsequent runs use the cache.

## ES
- `ffmpeg not found`: instala ffmpeg y verifica `ffmpeg`/`ffprobe` en `PATH`.
- Error de autenticacion OpenAI: configura `OPENAI_API_KEY`.
- Error OpenAI + `srt`: usa `--model whisper-1` o `--model gpt-4o-transcribe-diarize`.
- Desactiva chunking si es necesario: `--chunk-mode off`.
- Fuerza chunking para trabajos largos: `--chunk-mode force`.
- Se descargo un video incorrecto de YouTube: evita URL con escapes (`watch\\?v\\=...`). Usa `watch?v=...`.
- Si necesitas mas visibilidad: usa `--log-level DEBUG --progress-log-seconds 10 --log-file ./transcribe_debug.log`.
- Error importando MLX Whisper: instala con `pip install mlx-whisper`. Requiere Apple Silicon (M1+).
- Descarga lenta de modelo MLX Whisper: los modelos se descargan de Hugging Face la primera vez; las siguientes ejecuciones usan cache.
