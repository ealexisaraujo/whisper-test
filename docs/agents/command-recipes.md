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
