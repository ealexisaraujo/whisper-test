# Transcription Workflow / Flujo de Transcripción

## EN
Pipeline order:
1. `InputResolver` decides URL vs local path.
2. `MediaDownloader` downloads YouTube audio via `yt-dlp` (if needed).
3. `AudioNormalizer` converts media to normalized WAV (`24k` for VibeVoice, `16k` for OpenAI/Whisper).
4. Long-audio guard auto-chunks by duration/size when `--chunk-mode auto`.
5. Selected backend transcribes (`vibevoice`/`openai`/`whisper`) once or per chunk.
6. Chunked runs are merged into one final timeline.
7. `OutputWriter` writes `txt`, `srt`, `json`.

Manual backend switch only: failures return non-zero and print rerun commands.

## ES
Orden del pipeline:
1. `InputResolver` decide URL vs archivo local.
2. `MediaDownloader` descarga audio de YouTube con `yt-dlp` (si aplica).
3. `AudioNormalizer` convierte a WAV normalizado (`24k` para VibeVoice, `16k` para OpenAI/Whisper).
4. La protección de audio largo aplica auto-chunk por duración/tamaño cuando `--chunk-mode auto`.
5. El backend seleccionado transcribe (`vibevoice`/`openai`/`whisper`) una vez o por chunk.
6. Las ejecuciones por chunks se fusionan en una sola línea de tiempo final.
7. `OutputWriter` genera `txt`, `srt`, `json`.

Cambio de backend solo manual: ante errores se retorna código no-cero y se muestran comandos sugeridos.
