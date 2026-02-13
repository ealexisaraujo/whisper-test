# Backend Capabilities / Capacidades de Backends

## EN
Execution policy:
- Only one backend runs per execution.
- Backend is chosen only by `--backend` (default: `vibevoice`).
- No automatic backend fallback.
- Chunking (`--chunk-mode`) never switches backend; it only splits audio.

### VibeVoice
- Default model: `microsoft/VibeVoice-ASR`
- Primary backend in this repo
- Supports `context_info` + hotwords
- Structured segments with speaker/timestamps
- macOS is best-effort

### OpenAI
- Supported models: `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, `whisper-1`
- `srt` is valid only when timestamps are available
  - Use `whisper-1` or `gpt-4o-transcribe-diarize`
- Requires `OPENAI_API_KEY`
- Auto-chunk protects against oversized uploads in `--chunk-mode auto`

### Whisper
- Default model: `medium`
- Local open-source backend
- Accepts optional prompt via merged context info

Quick choice:
- Need strongest local context-aware path: `vibevoice`.
- Need local stability for long jobs: `whisper`.
- Need cloud API speed: `openai`.
- Need OpenAI + `srt`: `whisper-1` or `gpt-4o-transcribe-diarize`.

## ES
Política de ejecución:
- Solo se ejecuta un backend por comando.
- El backend se define únicamente con `--backend` (por defecto: `vibevoice`).
- No existe fallback automático entre backends.
- Chunking (`--chunk-mode`) nunca cambia backend; solo divide audio.

### VibeVoice
- Modelo por defecto: `microsoft/VibeVoice-ASR`
- Backend primario del repositorio
- Soporta `context_info` + hotwords
- Segmentos estructurados con hablante y timestamps
- En macOS funciona en modo best-effort

### OpenAI
- Modelos soportados: `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, `whisper-1`
- `srt` solo cuando hay timestamps
  - Usar `whisper-1` o `gpt-4o-transcribe-diarize`
- Requiere `OPENAI_API_KEY`
- Auto-chunk protege contra cargas demasiado grandes en `--chunk-mode auto`

### Whisper
- Modelo por defecto: `medium`
- Backend local open-source
- Acepta prompt opcional usando contexto combinado

Selección rápida:
- Mejor ruta local con contexto: `vibevoice`.
- Mayor estabilidad local para trabajos largos: `whisper`.
- Velocidad vía API cloud: `openai`.
- OpenAI + `srt`: `whisper-1` o `gpt-4o-transcribe-diarize`.
