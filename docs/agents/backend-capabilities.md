# Backend Capabilities / Capacidades de Backends

## EN
Execution policy:
- Only one backend runs per execution.
- Backend is chosen only by `--backend` (default: `mlx-whisper`).
- No automatic fallback; the auto pipeline uses deterministic routing + fusion and raises when a chosen engine fails.
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

### Auto (local multi-stage pipeline)
- `--backend auto`: VAD chunks -> probe (language, phone band, SNR) -> profile from `transcribe/profiles.json` -> Qwen3-ASR 1.7B and/or Whisper large-v3 -> guards -> alignment -> Gemma 4 26B-A4B picker -> Whisper audio verify -> gated Claude refine -> speakers -> segments.
- English and Spanish fully supported; other languages best effort (no refine, no phonetic gates, flagged in the JSON).
- Routing is deterministic: the profile names the engines and stages; `benchmarks/suite/make_profiles.py` turns off any stage or second engine that doesn't beat the best single engine on the dev split. A profile may be single-engine. The architecture supports fusion, pick, verify and refine, but under the calibrated profiles (2026-10-25 suite) **English = Qwen3-ASR alone, Spanish = Whisper large-v3 alone, and no correction stages run**; `other` (any other language, code-switching) is Whisper large-v3 alone, set by hand and unvalidated (pick/verify measured harmful on en/es).
- Core stages (load, VAD, probe, chosen engines, assemble) raise `TranscriptionError`; enhancement stages (diarize, pick, verify, refine) skip and record `skipped: <reason>`.
- `[unclear: ...]` markers appear only from Whisper low-probability words (Spanish and other languages), with no `| other` alternative (a single-engine profile has no second opinion); English (Qwen3 has no word probabilities) never marks. With a fused profile a marker would read `[unclear: best | other]`. JSON `metadata.auto` carries per-word tier/confidence/source, stage statuses, the profile entry used, a trust score and `timestamps`.
- Known limitation: Qwen3-ASR (English) has no word times, so SRT cue and word times are approximate (words spread over each VAD chunk by character length); `metadata.auto.timestamps` is `"approximate"` then, `"engine"` for Whisper. Pending: wire Qwen3-ForcedAligner.
- The trust score is `"uncalibrated"` (no estimated WER) for the shipped profiles: its measured Spearman with per-file WER (0.24 English, 0.63 Spanish) is below the 0.7 bar.
- Diarization needs `HF_TOKEN` (pyannote community-1 terms accepted); without it every word is `Speaker 1`.
- Local only (Claude refine via the logged-in CLI); peak memory footprint 16.1 GB measured on a 16-min talk with the dual pipeline (picker 13.4 GB; 30-min audio not measured end to end); Whisper large-v3 alone peaks at 6.3 GB on 30-min audio; single-engine profiles never load the picker; measured speed and accuracy per slice: `auto-pipeline-results.md`.

### Refine stage (opt-in, any backend)
- `--refine` (or `TRANSCRIBE_REFINE=1`) sends the final segment texts to Claude through the locally installed, logged-in Claude Code CLI (`claude -p` headless, no tools/plugins/MCP) to fix ASR errors: misheard words, homophones, broken word boundaries, names/brands (using the video title, channel and description as context), casing/punctuation.
- Timestamps, speakers and segment count are never changed; only `text`. It is not a backend fallback: the selected backend's output is refined, nothing is re-transcribed.
- Model: `--refine-model` / `TRANSCRIBE_REFINE_MODEL`, default alias `sonnet`. Binary: `CLAUDE_BIN` or `claude` on PATH; missing binary raises `CapabilityError` before transcription starts.
- Batches of 60 segments, 4 in parallel, each with the previous 5 segments as read-only context. EG2 low-confidence words are passed as `uncertain` hints.
- Guard: a segment whose refined text diverges too much from the original (word-level similarity < 0.5, char-level < 0.4 for segments under 4 words) or comes back empty keeps the original text.
- A batch that fails twice raises `TranscriptionError` (no silent skip). Needs segmented output: OpenAI text-only models (`gpt-4o-transcribe`, `gpt-4o-mini-transcribe`) are rejected with `CapabilityError`.
- JSON `metadata.refine` records model, batches, changed/guarded counts, `cost_usd`, `duration_s` and `original_texts`.

Quick choice:
- Meetings or several overlapping speakers: `mlx-whisper` (the suite's meeting slice: turbo 21.9 % WER vs `auto` 30.7 %).
- Best local accuracy on single-speaker English or Spanish talks, accents, business audio and noise: `auto` (it loses to `mlx-whisper` turbo on read Spanish speech, `es_mls`, and on English phone audio, `en_phone`). `[unclear]` markers appear only for Spanish and other languages; the English profile (Qwen3-ASR alone) never marks.
- Apple Silicon Mac, fastest English draft: `eg2`.
- Fast English with fewer errors: `eg2 --refine`.
- Apple Silicon Mac, general-purpose local default (and the safest choice for meetings, read speech and phone audio): `mlx-whisper`.
- Need cloud API speed: `openai`.
- Need OpenAI + `srt`: `whisper-1` or `gpt-4o-transcribe-diarize`.

## ES
Politica de ejecucion:
- Solo se ejecuta un backend por comando.
- El backend se define unicamente con `--backend` (por defecto: `mlx-whisper`).
- No existe fallback automatico; el pipeline auto usa ruteo deterministico + fusion y falla si un motor elegido falla.
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

### Auto (pipeline local de varias etapas)
- `--backend auto`: chunks VAD -> sondeo (idioma, banda telefonica, SNR) -> perfil de `transcribe/profiles.json` -> Qwen3-ASR 1.7B y/o Whisper large-v3 -> guardas -> alineacion -> selector Gemma 4 26B-A4B -> verificacion por audio con Whisper -> refine de Claude con compuertas -> hablantes -> segmentos.
- Ingles y espanol con soporte completo; otros idiomas en modo best effort (sin refine ni compuertas foneticas, marcado en el JSON).
- Ruteo deterministico: el perfil define motores y etapas; `benchmarks/suite/make_profiles.py` apaga toda etapa o segundo motor que no supere al mejor motor individual en el split dev. Un perfil puede usar un solo motor. La arquitectura soporta fusion, pick, verify y refine, pero con los perfiles calibrados (suite 2026-10-25) **ingles = Qwen3-ASR solo, espanol = Whisper large-v3 solo y no corre ninguna etapa de correccion**; `other` (otros idiomas, code-switching) es Whisper large-v3 solo, definido a mano y sin validar (pick/verify resultaron danninos en en/es).
- Etapas core (carga, VAD, sondeo, motores elegidos, ensamblado) lanzan `TranscriptionError`; las de mejora (diarize, pick, verify, refine) se omiten y registran `skipped: <motivo>`.
- Las marcas `[unclear: ...]` solo salen de palabras de baja probabilidad de Whisper (espanol y otros idiomas), sin alternativa `| otra` (un perfil de un solo motor no tiene segunda opinion); el ingles (Qwen3 no da probabilidades por palabra) nunca marca. Con un perfil fusionado seria `[unclear: mejor | otra]`. El JSON `metadata.auto` trae tier/confianza/origen por palabra, estado de cada etapa, la entrada de perfil usada, un trust score y `timestamps`.
- Limitacion conocida: Qwen3-ASR (ingles) no da tiempos por palabra, asi que los tiempos de SRT y de palabra son aproximados (palabras repartidas por caracteres en cada chunk VAD); `metadata.auto.timestamps` es `"approximate"` en ese caso y `"engine"` con Whisper. Pendiente: conectar Qwen3-ForcedAligner.
- El trust score es `"uncalibrated"` (sin WER estimado) en los perfiles actuales: su Spearman medido con el WER por archivo (0.24 ingles, 0.63 espanol) queda bajo el umbral de 0.7.
- La diarizacion requiere `HF_TOKEN` (terminos de pyannote community-1 aceptados); sin el, todo es `Speaker 1`.
- Solo local (refine de Claude via el CLI con sesion iniciada); memoria pico medida de 16.1 GB en una charla de 16 min con el pipeline dual (selector 13.4 GB; audio de 30 min sin medir de punta a punta); Whisper large-v3 solo llega a 6.3 GB con 30 min de audio; los perfiles de un solo motor nunca cargan el selector; velocidad y precision medidas por slice: `auto-pipeline-results.md`.

### Etapa refine (opcional, cualquier backend)
- `--refine` (o `TRANSCRIBE_REFINE=1`) envia los textos finales de los segmentos a Claude mediante el CLI de Claude Code instalado localmente y con sesion iniciada (`claude -p` headless, sin tools/plugins/MCP) para corregir errores de ASR: palabras mal oidas, homofonos, limites de palabra rotos, nombres/marcas (usando titulo, canal y descripcion del video como contexto), mayusculas/puntuacion.
- Nunca cambia timestamps, hablantes ni numero de segmentos; solo `text`. No es un fallback de backend: se refina la salida del backend elegido, no se vuelve a transcribir.
- Modelo: `--refine-model` / `TRANSCRIBE_REFINE_MODEL`, alias por defecto `sonnet`. Binario: `CLAUDE_BIN` o `claude` en PATH; si falta, `CapabilityError` antes de transcribir.
- Lotes de 60 segmentos, 4 en paralelo, cada uno con los 5 segmentos previos como contexto de solo lectura. Las palabras de baja confianza de EG2 se envian como pistas `uncertain`.
- Guarda: si el texto refinado se aleja demasiado del original (similitud por palabras < 0.5; por caracteres < 0.4 en segmentos de menos de 4 palabras) o llega vacio, se conserva el original.
- Un lote que falla dos veces lanza `TranscriptionError` (nada se omite en silencio). Requiere salida segmentada: los modelos OpenAI solo-texto (`gpt-4o-transcribe`, `gpt-4o-mini-transcribe`) se rechazan con `CapabilityError`.
- El JSON guarda en `metadata.refine`: modelo, lotes, segmentos cambiados/protegidos, `cost_usd`, `duration_s` y `original_texts`.

Seleccion rapida:
- Reuniones o varios hablantes que se solapan: `mlx-whisper` (slice de reuniones de la suite: turbo 21.9 % WER vs `auto` 30.7 %).
- Mejor precision local en ingles o espanol con un solo hablante (charlas, acentos, audio de negocios, ruido): `auto` (pierde contra `mlx-whisper` turbo en espanol leido, `es_mls`, y en telefono en ingles, `en_phone`). Las marcas `[unclear]` solo aparecen en espanol y otros idiomas; el perfil ingles (Qwen3-ASR solo) nunca marca.
- Mac con Apple Silicon, mejor velocidad local: `mlx-whisper`.
- Ingles rapido con menos errores: `eg2 --refine`.
- Velocidad via API cloud: `openai`.
- OpenAI + `srt`: `whisper-1` o `gpt-4o-transcribe-diarize`.
