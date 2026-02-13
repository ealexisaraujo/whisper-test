# Backend Capabilities / Capacidades de Backends

## EN
Execution policy:
- Only one backend runs per execution.
- Backend is chosen only by `--backend` (default: `mlx-whisper`).
- No automatic backend fallback.
- Chunking (`--chunk-mode`) never switches backend; it only splits audio.

### MLX Whisper
- Default model: `mlx-community/whisper-turbo`
- MLX-optimized local backend for Apple Silicon (M1/M2/M3/M4)
- ~17x realtime on M1 Pro
- Accepts optional language and prompt via merged context info
- No torch/CUDA dependency — uses Apple MLX framework natively
- Reuses `normalize_whisper_segments()` for output normalization

### OpenAI
- Supported models: `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, `whisper-1`
- `srt` is valid only when timestamps are available
  - Use `whisper-1` or `gpt-4o-transcribe-diarize`
- Requires `OPENAI_API_KEY`
- Auto-chunk protects against oversized uploads in `--chunk-mode auto`

Quick choice:
- Apple Silicon Mac, best local speed: `mlx-whisper`.
- Need cloud API speed: `openai`.
- Need OpenAI + `srt`: `whisper-1` or `gpt-4o-transcribe-diarize`.

## ES
Politica de ejecucion:
- Solo se ejecuta un backend por comando.
- El backend se define unicamente con `--backend` (por defecto: `mlx-whisper`).
- No existe fallback automatico entre backends.
- Chunking (`--chunk-mode`) nunca cambia backend; solo divide audio.

### MLX Whisper
- Modelo por defecto: `mlx-community/whisper-turbo`
- Backend local optimizado con MLX para Apple Silicon (M1/M2/M3/M4)
- ~17x realtime en M1 Pro
- Acepta idioma y prompt opcional usando contexto combinado
- Sin dependencia de torch/CUDA — usa Apple MLX de forma nativa
- Reutiliza `normalize_whisper_segments()` para normalizar salida

### OpenAI
- Modelos soportados: `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, `whisper-1`
- `srt` solo cuando hay timestamps
  - Usar `whisper-1` o `gpt-4o-transcribe-diarize`
- Requiere `OPENAI_API_KEY`
- Auto-chunk protege contra cargas demasiado grandes en `--chunk-mode auto`

Seleccion rapida:
- Mac con Apple Silicon, mejor velocidad local: `mlx-whisper`.
- Velocidad via API cloud: `openai`.
- OpenAI + `srt`: `whisper-1` o `gpt-4o-transcribe-diarize`.
