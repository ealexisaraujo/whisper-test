# Command Recipes / Recetas de Comandos

## EN
Use this decision order:
1. Choose backend first (`--backend`).
2. Add language if known (`--language es`).
3. Keep chunking in `auto` unless you need to force or disable it.
4. Request outputs (`--output-formats txt,srt,json`).

Single command for your long Spanish YouTube case:
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=7jERZcUcBZA" \
  --backend mlx-whisper \
  --language es \
  --output-formats txt,srt,json \
  --output-dir work/out \
  --output-basename 7jERZcUcBZA_es
```

Local file with context:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend mlx-whisper \
  --language es \
  --context-info "Reunion comercial trimestral" \
  --hotwords "Alexis,Platzi,Ingresos"
```

OpenAI fast path:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend openai \
  --model gpt-4o-transcribe \
  --language es
```

OpenAI with SRT (timestamp-capable model required):
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend openai \
  --model whisper-1 \
  --output-formats txt,srt,json
```

MLX Whisper with custom model:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend mlx-whisper \
  --model mlx-community/whisper-large-v3 \
  --language en
```

Fast English draft + Claude refine (requires the logged-in Claude Code CLI):
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=<id>" \
  --backend eg2 \
  --refine
# cheaper/faster refine model, or enable by default via TRANSCRIBE_REFINE=1 in .env
python transcribe_youtube.py ./talk.mp4 --backend eg2 --refine --refine-model haiku
```

Auto pipeline (per-profile local engines, chosen by the eval suite; best for single-speaker en/es audio; `[unclear: word]` markers (no alternative) appear for Spanish and other languages only; for meetings use `--backend mlx-whisper`):
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=<id>" --backend auto --output-dir work/out
# known speaker count (1 skips diarization), extra terms for the refine gates, no Claude
python transcribe_youtube.py ./talk.mp4 --backend auto --num-speakers 1 --terms "Karpathy,nanochat" --no-refine
```
Note: `--terms` and `--refine` only matter for profiles with refine on; the calibrated en/es profiles currently have refine off (the CLI logs `refine is off for profile <key>; --refine ignored`), so they change nothing today. `TRANSCRIBE_REFINE=0` also turns auto refine off.

Eval suite (auto pipeline; datasets in ~/.cache/whisper_translate/suite):
```bash
.venv/bin/python -m benchmarks.suite.run_suite --manifest smoke --out /tmp/smoke.json \
  --systems mlx-whisper-turbo,eg2,auto:single-a,auto:single-b,auto:fused,auto:+pick,auto:+verify,auto:+refine
.venv/bin/python -m benchmarks.suite.gate --base base.json --cand cand.json --split test
.venv/bin/python -m benchmarks.suite.make_profiles --results benchmarks/suite/results/auto-full.json --write transcribe/profiles.json
```

Disable chunking:
```bash
python transcribe_youtube.py ./audio.wav --backend mlx-whisper --chunk-mode off
```

Force chunking:
```bash
python transcribe_youtube.py ./audio.wav --backend mlx-whisper --chunk-mode force --chunk-seconds 1200
```

If a backend fails (manual switch only):
```bash
python transcribe_youtube.py "<input>" --backend mlx-whisper
python transcribe_youtube.py "<input>" --backend openai --model whisper-1
```

High-visibility logging (for "it looks stuck" runs):
```bash
python transcribe_youtube.py "<input>" \
  --backend mlx-whisper \
  --timeout-seconds 900 \
  --log-level DEBUG \
  --progress-log-seconds 10 \
  --log-file ./transcribe_debug.log
```

## ES
Usa este orden de decision:
1. Elige backend primero (`--backend`).
2. Agrega idioma si lo conoces (`--language es`).
3. Manten chunking en `auto` salvo que necesites forzar o desactivar.
4. Define salidas (`--output-formats txt,srt,json`).

Comando unico para tu caso de YouTube largo en espanol:
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=7jERZcUcBZA" \
  --backend mlx-whisper \
  --language es \
  --output-formats txt,srt,json \
  --output-dir work/out \
  --output-basename 7jERZcUcBZA_es
```

Archivo local con contexto:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend mlx-whisper \
  --language es \
  --context-info "Reunion comercial trimestral" \
  --hotwords "Alexis,Platzi,Ingresos"
```

Ruta rapida OpenAI:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend openai \
  --model gpt-4o-transcribe \
  --language es
```

OpenAI con SRT (requiere modelo con timestamps):
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend openai \
  --model whisper-1 \
  --output-formats txt,srt,json
```

MLX Whisper con modelo personalizado:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend mlx-whisper \
  --model mlx-community/whisper-large-v3 \
  --language en
```

Borrador rapido en ingles + refine con Claude (requiere el CLI de Claude Code con sesion iniciada):
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=<id>" \
  --backend eg2 \
  --refine
# modelo de refine mas barato/rapido, o activarlo por defecto con TRANSCRIBE_REFINE=1 en .env
python transcribe_youtube.py ./talk.mp4 --backend eg2 --refine --refine-model haiku
```

Pipeline auto (motores locales por perfil, elegidos por la suite de evaluacion; mejor para audio en/es de un solo hablante; las marcas `[unclear: palabra]` (sin alternativa) solo aparecen en espanol y otros idiomas; para reuniones usa `--backend mlx-whisper`):
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=<id>" --backend auto --output-dir work/out
# numero de hablantes conocido (1 omite la diarizacion), terminos extra para las compuertas de refine, sin Claude
python transcribe_youtube.py ./charla.mp4 --backend auto --num-speakers 1 --terms "Karpathy,nanochat" --no-refine
```
Nota: `--terms` y `--refine` solo importan en perfiles con refine activo; los perfiles en/es calibrados hoy lo tienen apagado (el CLI registra `refine is off for profile <clave>; --refine ignored`), asi que hoy no cambian nada. `TRANSCRIBE_REFINE=0` tambien apaga el refine de auto.

Suite de evaluacion (pipeline auto; datasets en ~/.cache/whisper_translate/suite):
```bash
.venv/bin/python -m benchmarks.suite.run_suite --manifest smoke --out /tmp/smoke.json \
  --systems mlx-whisper-turbo,eg2,auto:single-a,auto:single-b,auto:fused,auto:+pick,auto:+verify,auto:+refine
.venv/bin/python -m benchmarks.suite.gate --base base.json --cand cand.json --split test
.venv/bin/python -m benchmarks.suite.make_profiles --results benchmarks/suite/results/auto-full.json --write transcribe/profiles.json
```

Desactivar chunking:
```bash
python transcribe_youtube.py ./audio.wav --backend mlx-whisper --chunk-mode off
```

Forzar chunking:
```bash
python transcribe_youtube.py ./audio.wav --backend mlx-whisper --chunk-mode force --chunk-seconds 1200
```

Si falla un backend (solo cambio manual):
```bash
python transcribe_youtube.py "<input>" --backend mlx-whisper
python transcribe_youtube.py "<input>" --backend openai --model whisper-1
```

Logging de alta visibilidad (cuando parece que se queda congelado):
```bash
python transcribe_youtube.py "<input>" \
  --backend mlx-whisper \
  --timeout-seconds 900 \
  --log-level DEBUG \
  --progress-log-seconds 10 \
  --log-file ./transcribe_debug.log
```
