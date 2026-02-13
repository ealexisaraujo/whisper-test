# whisper_translate

Backend-driven CLI to transcribe YouTube/local media with manual backend selection (`vibevoice`, `openai`, `whisper`) and unified `txt`/`srt`/`json` outputs.

## English

### Requirements
- Python `3.11`
- `ffmpeg` and `ffprobe` in `PATH`

### Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional VibeVoice stack:
```bash
pip install -r requirements-vibevoice.txt
```

### Environment
Copy `.env.example` and set values if needed:
- `OPENAI_API_KEY` for OpenAI backend
- `OPENAI_BASE_URL` for compatible endpoints

### Usage
Backend execution rule (critical):
- The script runs exactly one backend: the value from `--backend`.
- If `--backend` is omitted, it runs `vibevoice`.
- It never auto-switches backend on failure.
- `--chunk-mode auto` only splits long audio; it does not change backend.

Default backend is `vibevoice`:
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=..."
```
Important: do not escape `?` or `=` inside quoted URLs. Use `watch?v=...`, not `watch\?v\=...`.

Long YouTube/meeting video (single command, auto-chunk enabled by default):
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=7jERZcUcBZA" \
  --backend whisper \
  --language es
```
The script detects long audio and processes chunks internally, then writes one merged `txt/srt/json` output set.

Local file with context/hotwords:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend vibevoice \
  --context-info "Quarterly planning call" \
  --hotwords "Alexis,Platzi,Revenue"
```

OpenAI backend with diarization-capable model:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend openai \
  --model gpt-4o-transcribe-diarize
```

Whisper backend:
```bash
python transcribe_youtube.py ./meeting.mp4 --backend whisper --model medium
```

### Which backend should I use?
| Situation | Backend | Why | Command |
|---|---|---|---|
| Best local quality with context/hotwords | `vibevoice` | Primary model with context injection | `python transcribe_youtube.py "<input>" --backend vibevoice --context-info "..." --hotwords "..."` |
| Long local run on Mac with less memory risk | `whisper` | More stable local runtime than heavy VibeVoice | `python transcribe_youtube.py "<input>" --backend whisper --language es` |
| Fast cloud transcription | `openai` | Managed API backend | `python transcribe_youtube.py "<input>" --backend openai --model gpt-4o-transcribe` |
| OpenAI + must output `srt` | `openai` + timestamp model | `srt` needs timestamp-capable model | `python transcribe_youtube.py "<input>" --backend openai --model whisper-1 --output-formats txt,srt,json` |
| Strict offline local workflow | `whisper` | No cloud dependency | `python transcribe_youtube.py "<input>" --backend whisper` |
| Backend failed and you want manual switch | explicit `--backend` rerun | No fallback by design | `python transcribe_youtube.py "<input>" --backend whisper` |

More copy/paste recipes: `docs/agents/command-recipes.md`.

### Long-audio controls
- `--chunk-mode auto|off|force` (default: `auto`)
- `--chunk-seconds 1800` (default: 30 minutes per chunk)
- `--chunk-threshold-minutes 45` (auto-chunk trigger threshold)
- `--openai-max-file-mb 24` (auto-chunk trigger for OpenAI upload limits)
- `--save-chunks-dir <path>` (optional: persist generated chunk WAV files)

### Backend matrix
- `vibevoice`:
  - Default model: `microsoft/VibeVoice-ASR`
  - Supports context/hotwords and speaker-aware segments
  - macOS mode is best-effort and may be slow/OOM
- `openai`:
  - Default model: `gpt-4o-transcribe`
  - Supports `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, `whisper-1`
  - `srt` requires timestamp-capable models (`whisper-1` or `gpt-4o-transcribe-diarize`)
- `whisper`:
  - Default model: `medium`
  - Local open-source fallback backend

### Output schema (`json`)
```json
{
  "backend": "vibevoice",
  "model": "microsoft/VibeVoice-ASR",
  "source": "https://www.youtube.com/watch?v=...",
  "language": "en",
  "duration_sec": 123.45,
  "context_info": "Hotwords: Alexis, Platzi",
  "segments": [
    {"start": 0.0, "end": 1.2, "speaker": "A", "text": "Hello"}
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

### Instalación
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Stack opcional de VibeVoice:
```bash
pip install -r requirements-vibevoice.txt
```

### Variables de entorno
Copia `.env.example` y configura:
- `OPENAI_API_KEY` para backend OpenAI
- `OPENAI_BASE_URL` para endpoints compatibles

### Uso
Regla de ejecución de backend (crítica):
- El script ejecuta exactamente un backend: el valor de `--backend`.
- Si omites `--backend`, ejecuta `vibevoice`.
- Nunca cambia automáticamente de backend ante errores.
- `--chunk-mode auto` solo divide audio largo; no cambia backend.

Backend por defecto: `vibevoice`:
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=..."
```
Importante: no escapes `?` ni `=` dentro de la URL entre comillas. Usa `watch?v=...`, no `watch\?v\=...`.

Video largo de YouTube/reunión (un solo comando, auto-chunk activo por defecto):
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=7jERZcUcBZA" \
  --backend whisper \
  --language es
```
El script detecta audio largo, lo divide internamente y genera una sola salida final combinada (`txt/srt/json`).

Archivo local con contexto/hotwords:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend vibevoice \
  --context-info "Llamada trimestral" \
  --hotwords "Alexis,Platzi,Ingresos"
```

Backend OpenAI con modelo que soporta diarización:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend openai \
  --model gpt-4o-transcribe-diarize
```

Backend Whisper:
```bash
python transcribe_youtube.py ./meeting.mp4 --backend whisper --model medium
```

### ¿Qué backend usar?
| Situación | Backend | Motivo | Comando |
|---|---|---|---|
| Mejor calidad local con contexto/hotwords | `vibevoice` | Modelo principal con inyección de contexto | `python transcribe_youtube.py "<input>" --backend vibevoice --context-info "..." --hotwords "..."` |
| Ejecución local larga en Mac con menor riesgo de memoria | `whisper` | Runtime local más estable que VibeVoice pesado | `python transcribe_youtube.py "<input>" --backend whisper --language es` |
| Transcripción cloud rápida | `openai` | Backend administrado por API | `python transcribe_youtube.py "<input>" --backend openai --model gpt-4o-transcribe` |
| OpenAI + salida obligatoria `srt` | `openai` + modelo con timestamps | `srt` requiere timestamps | `python transcribe_youtube.py "<input>" --backend openai --model whisper-1 --output-formats txt,srt,json` |
| Flujo local 100% offline | `whisper` | Sin dependencia cloud | `python transcribe_youtube.py "<input>" --backend whisper` |
| Falló un backend y quieres cambio manual | reintentar con `--backend` explícito | No hay fallback por diseño | `python transcribe_youtube.py "<input>" --backend whisper` |

Más recetas copy/paste: `docs/agents/command-recipes.md`.

### Controles para audio largo
- `--chunk-mode auto|off|force` (por defecto: `auto`)
- `--chunk-seconds 1800` (por defecto: 30 minutos por chunk)
- `--chunk-threshold-minutes 45` (umbral para activar auto-chunk)
- `--openai-max-file-mb 24` (umbral de tamaño para auto-chunk con OpenAI)
- `--save-chunks-dir <path>` (opcional: guardar WAVs de chunks)

### Matriz de backends
- `vibevoice`:
  - Modelo por defecto: `microsoft/VibeVoice-ASR`
  - Soporta contexto/hotwords y segmentos por hablante
  - En macOS funciona en modo best-effort (puede ser lento o fallar por memoria)
- `openai`:
  - Modelo por defecto: `gpt-4o-transcribe`
  - Soporta `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, `whisper-1`
  - `srt` requiere modelos con timestamps (`whisper-1` o `gpt-4o-transcribe-diarize`)
- `whisper`:
  - Modelo por defecto: `medium`
  - Backend local open-source

### Política de cambio manual
No hay fallback automático. Si falla un backend, vuelve a ejecutar indicando otro backend.

### Tests
```bash
python -m pytest -q
```
