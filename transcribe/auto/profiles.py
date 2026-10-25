"""Engine/stage profiles for `--backend auto`, loaded from `transcribe/profiles.json`.

A profile key is `<lang>[-phone][-noisy]` (see `probe.build_profile`). `select` falls back from the
exact key to the language-only key and then to `default`, flagging the entry `"unvalidated": True`
whenever the exact key is not present (the entry then was not calibrated for this audio type).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..models import TranscriptionError
from .engines import known_engine_ids

PROFILES_PATH = Path(__file__).resolve().parent.parent / "profiles.json"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_entry(name: str, entry: Any) -> None:
    """Raise `TranscriptionError` unless `entry` is a well-formed profile entry."""
    if not isinstance(entry, dict):
        raise TranscriptionError(f"profiles: entry '{name}' must be an object")
    engines = entry.get("engines")
    known = known_engine_ids()
    if not isinstance(engines, list) or not engines or any(e not in known for e in engines):
        raise TranscriptionError(
            f"profiles: entry '{name}' needs a non-empty 'engines' list of known ids ({', '.join(known)})"
        )
    for flag in ("pick", "verify", "refine"):
        if not isinstance(entry.get(flag), bool):
            raise TranscriptionError(f"profiles: entry '{name}' needs a boolean '{flag}'")
    for number in ("margin", "solid_min_prob"):
        if not _is_number(entry.get(number)):
            raise TranscriptionError(f"profiles: entry '{name}' needs a numeric '{number}'")
    if entry.get("trust_spearman") is not None and not _is_number(entry["trust_spearman"]):
        raise TranscriptionError(f"profiles: entry '{name}' has a non-numeric 'trust_spearman'")


def validate_profiles(profiles: Any) -> None:
    """Raise `TranscriptionError` on a malformed table (missing `default`, bad entries or trust)."""
    if not isinstance(profiles, dict) or not isinstance(profiles.get("profiles"), dict):
        raise TranscriptionError("profiles: expected an object with a 'profiles' table")
    if "default" not in profiles:
        raise TranscriptionError("profiles: missing 'default' entry")
    validate_entry("default", profiles["default"])
    for name, entry in profiles["profiles"].items():
        validate_entry(name, entry)
    trust = profiles.get("trust")
    if trust is not None and not (isinstance(trust, dict) and isinstance(trust.get("coef"), dict)):
        raise TranscriptionError("profiles: 'trust' must be null or {\"coef\": {...}}")


def load_profiles(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read and validate the profiles file (default: `transcribe/profiles.json`)."""
    try:
        profiles = json.loads(Path(path or PROFILES_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TranscriptionError(f"profiles: cannot read {path or PROFILES_PATH}: {exc}") from exc
    validate_profiles(profiles)
    return profiles


def select(profiles: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a copy of the entry for `key`, falling back to the language key, then `default`."""
    table = profiles.get("profiles") if isinstance(profiles, dict) else None
    if not isinstance(table, dict) or "default" not in profiles:
        raise TranscriptionError("profiles: expected a 'profiles' table and a 'default' entry")
    if key in table:
        validate_entry(key, table[key])
        return copy.deepcopy(table[key])
    language_key = key.split("-", 1)[0]
    entry = table.get(language_key)
    name = language_key if entry is not None else "default"
    chosen_entry = entry if entry is not None else profiles["default"]
    validate_entry(name, chosen_entry)
    chosen = copy.deepcopy(chosen_entry)
    chosen["unvalidated"] = True
    return chosen
