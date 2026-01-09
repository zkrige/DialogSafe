from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, TypeVar, get_args

from dotenv import load_dotenv


Mode = Literal["mute", "bleep"]

WhisperBackend = Literal["local_whisper", "openai_api"]


@dataclass
class AppConfig:
    """Top-level configuration for the profanity cleaner CLI."""

    input_path: Path
    output_path: Path
    mode: Mode = "mute"
    profanity_list_path: Optional[Path] = None
    audio_language: str = "en"
    chunk_length_seconds: int = 300
    min_confidence: float = 0.6
    max_gap_combine_ms: int = 500
    bleep_sound_path: Optional[Path] = None
    output_dir: Path = Path("out")
    emit_clean_transcript: bool = True
    emit_subtitles: bool = False

    # Logging
    verbose: bool = False

    # Speech-to-text backend selection
    whisper_backend: WhisperBackend = "local_whisper"

    # Whisper model selection (varies by backend).
    # - local_whisper supports: tiny, base, small, medium, large, ...
    # - openai_api supports: whisper-1, gpt-4o-mini-transcribe, ...
    whisper_model: str = "base"

    # Debugging / inspection flags
    debug_dump_audio: bool = False  # when True, copy audio.wav and chunk WAVs into output_dir/audio_debug

    # Pipeline control
    # When True, bypass "already-clean" detection and force processing.
    force: bool = False

    # Derived / loaded values
    profanity_terms: List[str] = field(default_factory=list)


DEFAULT_PROFANITY_WORDS_EN: List[str] = [
    # Fallback built-in list. For real use, prefer the configurable text file
    # in config/profanity_en.txt or a custom file passed via --profanity-list.
    "fuck",
    "shit",
    "bitch",
    "asshole",
    "bastard",
    "damn",
    "crap",
]


_T = TypeVar("_T")


def _env_get(name: str) -> Optional[str]:
    """Return env var value stripped, treating empty/whitespace as missing."""
    raw = os.getenv(name)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _parse_bool(value: str, *, var_name: str) -> bool:
    v = value.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    raise ValueError(
        f"Invalid boolean for {var_name}={value!r}. Expected 'true' or 'false' (case-insensitive)."
    )


def _parse_whisper_backend(value: str, *, var_name: str = "WHISPER_BACKEND") -> WhisperBackend:
    """Parse a user/env-provided Whisper backend selector into a `WhisperBackend`.

    Accepts ONLY canonical values (case-insensitive):
    - local_whisper
    - openai_api
    """

    v = value.strip().lower()

    # Keep explicit branches so the return type stays a strict `WhisperBackend` literal.
    if v == "local_whisper":
        return "local_whisper"
    if v == "openai_api":
        return "openai_api"

    allowed = ", ".join(str(x) for x in get_args(WhisperBackend))
    raise ValueError(
        f"Invalid {var_name} value {value!r}. Expected one of: {allowed}."
    )


def _env_int(name: str) -> Optional[int]:
    raw = _env_get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:  # noqa: PERF203
        raise ValueError(f"Invalid integer for {name}={raw!r}") from exc


def _env_float(name: str) -> Optional[float]:
    raw = _env_get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:  # noqa: PERF203
        raise ValueError(f"Invalid float for {name}={raw!r}") from exc


def _env_bool(name: str) -> Optional[bool]:
    raw = _env_get(name)
    if raw is None:
        return None
    return _parse_bool(raw, var_name=name)


def _coalesce(cli_value: Optional[_T], env_value: Optional[_T], default: _T) -> _T:
    """Precedence: CLI value > env value > default."""
    if cli_value is not None:
        return cli_value
    if env_value is not None:
        return env_value
    return default


def _load_words_from_text_file(path: Path) -> List[str]:
    """
    Load a profanity list from a plain text file.

    Each non-empty, non-comment line is treated as a word/phrase.
    Lines starting with '#' are ignored as comments.
    """
    words: List[str] = []
    if not path.exists():
        return words

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.append(line.lower())
    return words


def load_default_profanity_list(language: str) -> List[str]:
    """
    Return the default profanity list for the given language.

    For English, this prefers the editable text file at config/profanity_en.txt.
    If that file is missing or empty, it falls back to DEFAULT_PROFANITY_WORDS_EN.
    """
    lang = language.lower()

    if lang.startswith("en"):
        # Resolve project root: src/config.py -> project root
        project_root = Path(__file__).resolve().parent.parent
        text_path = project_root / "config" / "profanity_en.txt"
        words = _load_words_from_text_file(text_path)
        if words:
            return words
        return DEFAULT_PROFANITY_WORDS_EN

    # Fallback to English defaults if we do not yet support this language.
    return DEFAULT_PROFANITY_WORDS_EN


def load_config_from_args(args) -> AppConfig:
    """Create an AppConfig from parsed argparse.Namespace."""

    # Load .env (if present) so config values can be driven by environment.
    # Prefer repo-root .env (canonical), but allow a legacy fallback to src/.env.
    project_root = Path(__file__).resolve().parent.parent
    root_env = project_root / ".env"
    legacy_env = Path(__file__).resolve().parent / ".env"

    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=False)
    elif legacy_env.exists():
        load_dotenv(dotenv_path=legacy_env, override=False)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    mode: Mode = _coalesce(
        getattr(args, "mode", None),
        _env_get("MODE"),
        "mute",
    )
    if mode not in ("mute", "bleep"):
        raise ValueError(f"Invalid mode '{mode}', expected 'mute' or 'bleep'.")

    audio_language = _coalesce(
        getattr(args, "audio_language", None),
        _env_get("AUDIO_LANGUAGE"),
        "en",
    )

    chunk_length_seconds = _coalesce(
        getattr(args, "chunk_length_seconds", None),
        _env_int("CHUNK_LENGTH_SECONDS"),
        300,
    )
    min_confidence = _coalesce(
        getattr(args, "min_confidence", None),
        _env_float("MIN_CONFIDENCE"),
        0.6,
    )
    max_gap_combine_ms = _coalesce(
        getattr(args, "max_gap_combine_ms", None),
        _env_int("MAX_GAP_COMBINE_MS"),
        500,
    )

    emit_clean_transcript = _coalesce(
        getattr(args, "emit_clean_transcript", None),
        _env_bool("EMIT_CLEAN_TRANSCRIPT"),
        False,
    )
    emit_subtitles = _coalesce(
        getattr(args, "emit_subtitles", None),
        _env_bool("EMIT_SUBTITLES"),
        False,
    )
    verbose = _coalesce(
        getattr(args, "verbose", None),
        _env_bool("VERBOSE"),
        False,
    )
    debug_dump_audio = _coalesce(
        getattr(args, "debug_dump_audio", None),
        _env_bool("DEBUG_DUMP_AUDIO"),
        False,
    )

    force = _coalesce(
        getattr(args, "force", None),
        _env_bool("FORCE"),
        False,
    )

    whisper_model = _coalesce(
        getattr(args, "whisper_model", None),
        _env_get("WHISPER_MODEL"),
        "base",
    )

    whisper_backend_raw = _coalesce(
        getattr(args, "whisper_backend", None),
        _env_get("WHISPER_BACKEND"),
        "local_whisper",
    )
    whisper_backend = _parse_whisper_backend(whisper_backend_raw)

    output_dir_raw = _coalesce(
        getattr(args, "output_dir", None),
        _env_get("OUTPUT_DIR"),
        "out",
    )
    output_dir = Path(output_dir_raw).expanduser().resolve()

    profanity_list_raw = _coalesce(
        getattr(args, "profanity_list", None),
        _env_get("PROFANITY_LIST"),
        None,
    )
    bleep_sound_path_raw = _coalesce(
        getattr(args, "bleep_sound_path", None),
        _env_get("BLEEP_SOUND_PATH"),
        None,
    )

    profanity_list_path: Optional[Path] = None
    profanity_terms: List[str] = []

    if profanity_list_raw:
        profanity_list_path = Path(profanity_list_raw).expanduser().resolve()
        if not profanity_list_path.exists():
            raise FileNotFoundError(
                f"Profanity list file not found: {profanity_list_path}"
            )

        # Support either JSON (list of strings) or plain text (one term per line).
        import json

        text = profanity_list_path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Not valid JSON; treat as a plain text word list.
            profanity_terms = _load_words_from_text_file(profanity_list_path)
            if not profanity_terms:
                raise ValueError(
                    "Profanity list text file is empty or has only comments/blank lines."
                )
        else:
            if not isinstance(data, list) or not all(
                isinstance(item, str) for item in data
            ):
                raise ValueError(
                    "Profanity list JSON must be a list of strings (words/phrases)."
                )
            profanity_terms = [w.strip().lower() for w in data if w.strip()]
    else:
        profanity_terms = load_default_profanity_list(audio_language)

    config = AppConfig(
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        profanity_list_path=profanity_list_path,
        audio_language=audio_language,
        chunk_length_seconds=int(chunk_length_seconds),
        min_confidence=float(min_confidence),
        max_gap_combine_ms=int(max_gap_combine_ms),
        bleep_sound_path=Path(bleep_sound_path_raw).expanduser().resolve()
        if bleep_sound_path_raw
        else None,
        output_dir=output_dir,
        emit_clean_transcript=bool(emit_clean_transcript),
        emit_subtitles=bool(emit_subtitles),
        verbose=bool(verbose),
        debug_dump_audio=bool(debug_dump_audio),
        force=bool(force),
        profanity_terms=profanity_terms,
        whisper_backend=whisper_backend,
        whisper_model=str(whisper_model).strip(),
    )

    return config
