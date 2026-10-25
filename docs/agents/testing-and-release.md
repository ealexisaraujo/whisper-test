# Testing and Release / Pruebas y Release

## EN
Local checks:
1. `python -m py_compile transcribe_youtube.py`
2. `python -m pytest -q`

Release checklist:
1. Validate one run per backend with explicit `--backend` (mlx-whisper, openai).
2. Validate `txt,srt,json` outputs.
3. Validate OpenAI capability errors for non-timestamp models + `srt`.
4. Validate auto-chunk merge path for long audio.
5. Confirm no automatic backend fallback behavior (auto: a failing chosen engine raises; enhancement stages show `skipped: <reason>`).
6. Validate mlx-whisper on Apple Silicon with `--backend mlx-whisper`.
7. Auto pipeline: any prompt, model, threshold or profile change runs the smoke suite, then the gate on the test split (`benchmarks.suite.gate --split test`); it ships only if no slice's WER gets worse beyond its bootstrap CI, no stage's harm rate rises, and no-speech insertions don't rise. Regenerate `transcribe/profiles.json` with `make_profiles` from a full dev run. Real-model tests: `RUN_SLOW=1 .venv/bin/python -m pytest -q -m slow`.
8. Eval-suite data is local-only: `benchmarks/suite/manifests/*.jsonl` and `benchmarks/suite/results/*.json` embed reference transcripts (TED-LIUM is CC BY-NC-ND) and local paths, so they are git-ignored and must not be committed. Regenerate the manifests with `.venv/bin/python -m benchmarks.suite.datasets prepare` and the results with `benchmarks.suite.run_suite`; commit only the aggregates, `.venv/bin/python -m benchmarks.suite.run_suite --summarize benchmarks/suite/results/<name>.json` (writes `<name>.summary.json`). `make_profiles` and `gate` need the local per-item files.

## ES
Checks locales:
1. `python -m py_compile transcribe_youtube.py`
2. `python -m pytest -q`

Checklist de release:
1. Validar una ejecucion por backend con `--backend` explicito (mlx-whisper, openai).
2. Validar salidas `txt,srt,json`.
3. Validar errores de capacidad en OpenAI cuando se pide `srt` sin timestamps.
4. Validar el flujo de auto-chunk + fusion para audio largo.
5. Confirmar que no existe fallback automatico de backend (auto: un motor elegido que falla lanza error; las etapas de mejora muestran `skipped: <motivo>`).
6. Validar mlx-whisper en Apple Silicon con `--backend mlx-whisper`.
7. Pipeline auto: todo cambio de prompt, modelo, umbral o perfil corre la suite smoke y luego el gate en el split test (`benchmarks.suite.gate --split test`); solo se publica si ningun slice empeora su WER mas alla de su IC bootstrap, ninguna etapa sube su tasa de dano y las inserciones en audio sin voz no suben. Regenerar `transcribe/profiles.json` con `make_profiles` desde una corrida full en dev. Tests con modelos reales: `RUN_SLOW=1 .venv/bin/python -m pytest -q -m slow`.
8. Los datos de la suite de evaluacion son solo locales: `benchmarks/suite/manifests/*.jsonl` y `benchmarks/suite/results/*.json` incluyen las transcripciones de referencia (TED-LIUM es CC BY-NC-ND) y rutas locales, por eso estan en `.gitignore` y no se commitean. Regenera los manifests con `.venv/bin/python -m benchmarks.suite.datasets prepare` y los resultados con `benchmarks.suite.run_suite`; commitea solo los agregados, `.venv/bin/python -m benchmarks.suite.run_suite --summarize benchmarks/suite/results/<nombre>.json` (escribe `<nombre>.summary.json`). `make_profiles` y `gate` necesitan los archivos por item locales.
