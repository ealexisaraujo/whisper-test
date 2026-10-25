"""Auto pipeline (`--backend auto`): probe -> VAD chunks -> two local ASR engines -> align/pick -> refine.

Dependency rule: modules here import only `transcribe.models`, `constants`, `refine` (`resolve_claude_bin`) and `eg2`
(lazily, inside the EG2 engine). Nothing outside `transcribe/auto/` imports this package at module load: `cli.py` and
`transcribe/__init__.py` import it lazily (inside the auto branch / on first `transcribe.run_auto` access), so
`--backend mlx-whisper|openai|eg2` never loads `transcribe.auto`, `jellyfish` or `pronouncing`.
"""
