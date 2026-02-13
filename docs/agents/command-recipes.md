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
  --backend whisper \
  --language es \
  --output-formats txt,srt,json \
  --output-dir work/out \
  --output-basename 7jERZcUcBZA_es
```

Best local quality with context:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend vibevoice \
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

Disable chunking:
```bash
python transcribe_youtube.py ./audio.wav --backend whisper --chunk-mode off
```

Force chunking:
```bash
python transcribe_youtube.py ./audio.wav --backend whisper --chunk-mode force --chunk-seconds 1200
```

If a backend fails (manual switch only):
```bash
python transcribe_youtube.py "<input>" --backend whisper
python transcribe_youtube.py "<input>" --backend openai --model whisper-1
python transcribe_youtube.py "<input>" --backend vibevoice --device cpu
```

## ES
Usa este orden de decisión:
1. Elige backend primero (`--backend`).
2. Agrega idioma si lo conoces (`--language es`).
3. Mantén chunking en `auto` salvo que necesites forzar o desactivar.
4. Define salidas (`--output-formats txt,srt,json`).

Comando único para tu caso de YouTube largo en español:
```bash
python transcribe_youtube.py "https://www.youtube.com/watch?v=7jERZcUcBZA" \
  --backend whisper \
  --language es \
  --output-formats txt,srt,json \
  --output-dir work/out \
  --output-basename 7jERZcUcBZA_es
```

Mejor calidad local con contexto:
```bash
python transcribe_youtube.py ./meeting.mp4 \
  --backend vibevoice \
  --language es \
  --context-info "Reunion comercial trimestral" \
  --hotwords "Alexis,Platzi,Ingresos"
```

Ruta rápida OpenAI:
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

Desactivar chunking:
```bash
python transcribe_youtube.py ./audio.wav --backend whisper --chunk-mode off
```

Forzar chunking:
```bash
python transcribe_youtube.py ./audio.wav --backend whisper --chunk-mode force --chunk-seconds 1200
```

Si falla un backend (solo cambio manual):
```bash
python transcribe_youtube.py "<input>" --backend whisper
python transcribe_youtube.py "<input>" --backend openai --model whisper-1
python transcribe_youtube.py "<input>" --backend vibevoice --device cpu
```
