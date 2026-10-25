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

### EG2 (EmbeddingGemma 2, Core ML)
- Default model: `eg2demo-v5` (715 MB, downloaded once to `~/.cache/whisper_translate/eg2/v5`, SHA-256 verified; override with `EG2_MODEL_DIR`)
- Runs on the Apple Neural Engine; requires `coremltools` (macOS)
- ~100-120x realtime on M1 Pro, 5.3% WER on LibriSpeech test-clean (Whisper turbo: 16.5x, 1.9%); see `eg2-benchmark.md`
- English only: a non-English `--language` raises `CapabilityError`
- Ignores context/hotwords (fixed vocabulary); handles long audio itself, so auto-chunking is skipped
- Produces `txt`/`srt`/`json`; JSON `raw.words` carries per-word times and confidence

### Refine stage (opt-in, any backend)
- `--refine` (or `TRANSCRIBE_REFINE=1`) sends the final segment texts to Claude through the locally installed, logged-in Claude Code CLI (`claude -p` headless, no tools/plugins/MCP) to fix ASR errors: misheard words, homophones, broken word boundaries, names/brands (using the video title, channel and description as context), casing/punctuation.
- Timestamps, speakers and segment count are never changed; only `text`. It is not a backend fallback: the selected backend's output is refined, nothing is re-transcribed.
- Model: `--refine-model` / `TRANSCRIBE_REFINE_MODEL`, default alias `sonnet`. Binary: `CLAUDE_BIN` or `claude` on PATH; missing binary raises `CapabilityError` before transcription starts.
- Batches of 60 segments, 4 in parallel, each with the previous 5 segments as read-only context. EG2 low-confidence words are passed as `uncertain` hints.
- Guard: a segment whose refined text diverges too much from the original (word-level similarity < 0.5, char-level < 0.4 for segments under 4 words) or comes back empty keeps the original text.
- A batch that fails twice raises `TranscriptionError` (no silent skip). Needs segmented output: OpenAI text-only models (`gpt-4o-transcribe`, `gpt-4o-mini-transcribe`) are rejected with `CapabilityError`.
- JSON `metadata.refine` records model, batches, changed/guarded counts, `cost_usd`, `duration_s` and `original_texts`.

Quick choice:
- Apple Silicon Mac, fastest English draft: `eg2`.
- Fast English with fewer errors: `eg2 --refine`.
- Apple Silicon Mac, best local accuracy: `mlx-whisper`.
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

### Etapa refine (opcional, cualquier backend)
- `--refine` (o `TRANSCRIBE_REFINE=1`) envia los textos finales de los segmentos a Claude mediante el CLI de Claude Code instalado localmente y con sesion iniciada (`claude -p` headless, sin tools/plugins/MCP) para corregir errores de ASR: palabras mal oidas, homofonos, limites de palabra rotos, nombres/marcas (usando titulo, canal y descripcion del video como contexto), mayusculas/puntuacion.
- Nunca cambia timestamps, hablantes ni numero de segmentos; solo `text`. No es un fallback de backend: se refina la salida del backend elegido, no se vuelve a transcribir.
- Modelo: `--refine-model` / `TRANSCRIBE_REFINE_MODEL`, alias por defecto `sonnet`. Binario: `CLAUDE_BIN` o `claude` en PATH; si falta, `CapabilityError` antes de transcribir.
- Lotes de 60 segmentos, 4 en paralelo, cada uno con los 5 segmentos previos como contexto de solo lectura. Las palabras de baja confianza de EG2 se envian como pistas `uncertain`.
- Guarda: si el texto refinado se aleja demasiado del original (similitud por palabras < 0.5; por caracteres < 0.4 en segmentos de menos de 4 palabras) o llega vacio, se conserva el original.
- Un lote que falla dos veces lanza `TranscriptionError` (nada se omite en silencio). Requiere salida segmentada: los modelos OpenAI solo-texto (`gpt-4o-transcribe`, `gpt-4o-mini-transcribe`) se rechazan con `CapabilityError`.
- El JSON guarda en `metadata.refine`: modelo, lotes, segmentos cambiados/protegidos, `cost_usd`, `duration_s` y `original_texts`.

Seleccion rapida:
- Mac con Apple Silicon, mejor velocidad local: `mlx-whisper`.
- Ingles rapido con menos errores: `eg2 --refine`.
- Velocidad via API cloud: `openai`.
- OpenAI + `srt`: `whisper-1` o `gpt-4o-transcribe-diarize`.
