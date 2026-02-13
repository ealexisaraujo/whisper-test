# whisper_translate

Backend-driven CLI to transcribe YouTube/local media with manual backend selection (`mlx-whisper`, `openai`) and unified `txt`/`srt`/`json` outputs.

## English

### Requirements
- Python `3.11`
- `ffmpeg` and `ffprobe` in `PATH`
- Apple Silicon Mac (M1/M2/M3/M4) for `mlx-whisper`

### Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment
Copy `.env.example` and set values if needed:
- `OPENAI_API_KEY` for OpenAI backend
- `OPENAI_BASE_URL` for compatible endpoints

### Usage
Backend execution rule (critical):
- The script runs exactly one backend: the value from `--backend`.
- If `--backend` is omitted, it runs `mlx-whisper`.
- It never auto-switches backend on failure.
- `--chunk-mode auto` only splits long audio; it does not change backend.

Default backend is `mlx-whisper`:
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=..."
```
Important: do not escape `?` or `=` inside quoted URLs. Use `watch?v=...`, not `watch\?v\=...`.

Long YouTube/meeting video (single command, auto-chunk enabled by default):
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=7jERZcUcBZA" \
  --backend mlx-whisper \
  --language es
```
The script detects long audio and processes chunks internally, then writes one merged `txt/srt/json` output set.

Local file with context/hotwords:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend mlx-whisper \
  --context-info "Quarterly planning call" \
  --hotwords "Alexis,Platzi,Revenue"
```

OpenAI backend with diarization-capable model:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend openai \
  --model gpt-4o-transcribe-diarize
```

MLX Whisper with custom model:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend mlx-whisper \
  --model mlx-community/whisper-large-v3 \
  --language en
```

### Which backend should I use?
| Situation | Backend | Why | Command |
|---|---|---|---|
| Apple Silicon Mac (M1/M2/M3/M4) | `mlx-whisper` | Fast local transcription (~17x realtime) | `python transcribe_youtube.py "<input>" --backend mlx-whisper --language es` |
| Fast cloud transcription | `openai` | Managed API backend | `python transcribe_youtube.py "<input>" --backend openai --model gpt-4o-transcribe` |
| OpenAI + must output `srt` | `openai` + timestamp model | `srt` needs timestamp-capable model | `python transcribe_youtube.py "<input>" --backend openai --model whisper-1 --output-formats txt,srt,json` |
| Strict offline local workflow | `mlx-whisper` | No cloud dependency | `python transcribe_youtube.py "<input>" --backend mlx-whisper` |
| Backend failed and you want manual switch | explicit `--backend` rerun | No fallback by design | `python transcribe_youtube.py "<input>" --backend openai` |

More copy/paste recipes: `docs/agents/command-recipes.md`.

### Long-audio controls
- `--chunk-mode auto|off|force` (default: `auto`)
- `--chunk-seconds 1800` (default: 30 minutes per chunk)
- `--chunk-threshold-minutes 45` (auto-chunk trigger threshold)
- `--openai-max-file-mb 24` (auto-chunk trigger for OpenAI upload limits)
- `--save-chunks-dir <path>` (optional: persist generated chunk WAV files)

### Logging controls
- `--log-level DEBUG|INFO|WARNING|ERROR`
- `--progress-log-seconds 20` (heartbeat interval for long operations)
- `--log-file ./transcribe.log` (persist logs to file)

Debug-first command (recommended for troubleshooting):
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=fcZMmP5dsl4" \
  --backend mlx-whisper \
  --log-level DEBUG \
  --progress-log-seconds 10 \
  --log-file ./transcribe_debug.log
```

### Backend matrix
- `mlx-whisper`:
  - Default model: `mlx-community/whisper-turbo`
  - MLX-optimized local backend for Apple Silicon (~17x realtime on M1 Pro)
  - Accepts optional language and prompt via context info
- `openai`:
  - Default model: `gpt-4o-transcribe`
  - Supports `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, `whisper-1`
  - `srt` requires timestamp-capable models (`whisper-1` or `gpt-4o-transcribe-diarize`)

### Output schema (`json`)
```json
{
  "backend": "mlx-whisper",
  "model": "mlx-community/whisper-turbo",
  "source": "https://www.youtube.com/watch?v=...",
  "language": "en",
  "duration_sec": 123.45,
  "context_info": "Hotwords: Alexis, Platzi",
  "segments": [
    {"start": 0.0, "end": 1.2, "speaker": null, "text": "Hello"}
  ],
  "text": "Hello",
  "created_at": "2026-02-13T00:00:00+00:00",
  "metadata": {}
}
```

### Manual-switch policy
No automatic fallback is performed. On failure, rerun explicitly with another backend.

### Tests
```bash
python -m pytest -q
```

## Español

### Requisitos
- Python `3.11`
- `ffmpeg` y `ffprobe` en `PATH`
- Mac con Apple Silicon (M1/M2/M3/M4) para `mlx-whisper`

### Instalacion
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Variables de entorno
Copia `.env.example` y configura:
- `OPENAI_API_KEY` para backend OpenAI
- `OPENAI_BASE_URL` para endpoints compatibles

### Uso
Regla de ejecucion de backend (critica):
- El script ejecuta exactamente un backend: el valor de `--backend`.
- Si omites `--backend`, ejecuta `mlx-whisper`.
- Nunca cambia automaticamente de backend ante errores.
- `--chunk-mode auto` solo divide audio largo; no cambia backend.

Backend por defecto: `mlx-whisper`:
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=..."
```
Importante: no escapes `?` ni `=` dentro de la URL entre comillas. Usa `watch?v=...`, no `watch\?v\=...`.

Video largo de YouTube/reunion (un solo comando, auto-chunk activo por defecto):
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=7jERZcUcBZA" \
  --backend mlx-whisper \
  --language es
```
El script detecta audio largo, lo divide internamente y genera una sola salida final combinada (`txt/srt/json`).

Archivo local con contexto/hotwords:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend mlx-whisper \
  --context-info "Llamada trimestral" \
  --hotwords "Alexis,Platzi,Ingresos"
```

Backend OpenAI con modelo que soporta diarizacion:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend openai \
  --model gpt-4o-transcribe-diarize
```

MLX Whisper con modelo personalizado:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend mlx-whisper \
  --model mlx-community/whisper-large-v3 \
  --language en
```

### Que backend usar?
| Situacion | Backend | Motivo | Comando |
|---|---|---|---|
| Mac con Apple Silicon (M1/M2/M3/M4) | `mlx-whisper` | Transcripcion local rapida (~17x realtime) | `python transcribe_youtube.py "<input>" --backend mlx-whisper --language es` |
| Transcripcion cloud rapida | `openai` | Backend administrado por API | `python transcribe_youtube.py "<input>" --backend openai --model gpt-4o-transcribe` |
| OpenAI + salida obligatoria `srt` | `openai` + modelo con timestamps | `srt` requiere timestamps | `python transcribe_youtube.py "<input>" --backend openai --model whisper-1 --output-formats txt,srt,json` |
| Flujo local 100% offline | `mlx-whisper` | Sin dependencia cloud | `python transcribe_youtube.py "<input>" --backend mlx-whisper` |
| Fallo un backend y quieres cambio manual | reintentar con `--backend` explicito | No hay fallback por diseno | `python transcribe_youtube.py "<input>" --backend openai` |

Mas recetas copy/paste: `docs/agents/command-recipes.md`.

### Controles para audio largo
- `--chunk-mode auto|off|force` (por defecto: `auto`)
- `--chunk-seconds 1800` (por defecto: 30 minutos por chunk)
- `--chunk-threshold-minutes 45` (umbral para activar auto-chunk)
- `--openai-max-file-mb 24` (umbral de tamano para auto-chunk con OpenAI)
- `--save-chunks-dir <path>` (opcional: guardar WAVs de chunks)

### Controles de logging
- `--log-level DEBUG|INFO|WARNING|ERROR`
- `--progress-log-seconds 20` (intervalo de heartbeat para operaciones largas)
- `--log-file ./transcribe.log` (guardar logs en archivo)

Comando recomendado para diagnostico:
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=fcZMmP5dsl4" \
  --backend mlx-whisper \
  --log-level DEBUG \
  --progress-log-seconds 10 \
  --log-file ./transcribe_debug.log
```

### Matriz de backends
- `mlx-whisper`:
  - Modelo por defecto: `mlx-community/whisper-turbo`
  - Backend local optimizado con MLX para Apple Silicon (~17x realtime en M1 Pro)
  - Acepta idioma y prompt opcional usando contexto combinado
- `openai`:
  - Modelo por defecto: `gpt-4o-transcribe`
  - Soporta `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, `whisper-1`
  - `srt` requiere modelos con timestamps (`whisper-1` o `gpt-4o-transcribe-diarize`)

### Politica de cambio manual
No hay fallback automatico. Si falla un backend, vuelve a ejecutar indicando otro backend.

### Tests
```bash
python -m pytest -q
```
