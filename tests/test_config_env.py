from __future__ import annotations

from argparse import Namespace

import pytest

from src.config import load_config_from_args


def _base_args(*, input_path: str, output_path: str) -> Namespace:
    """Create an argparse.Namespace with all known CLI options.

    We set most fields to None to simulate "flag not provided" so env can apply.
    """

    return Namespace(
        input=input_path,
        output=output_path,
        mode=None,
        profanity_list=None,
        audio_language=None,
        chunk_length_seconds=None,
        bleep_sound_path=None,
        min_confidence=None,
        max_gap_combine_ms=None,
        emit_clean_transcript=None,
        emit_subtitles=None,
        output_dir=None,
        verbose=None,
        debug_dump_audio=None,
        whisper_model=None,
        force=None,
    )


def test_env_values_used_when_cli_omits_them(tmp_path, monkeypatch):
    input_file = tmp_path / "in.mp4"
    input_file.write_bytes(b"x")

    args = _base_args(input_path=str(input_file), output_path=str(tmp_path / "out.mp4"))

    monkeypatch.setenv("MODE", "bleep")
    monkeypatch.setenv("AUDIO_LANGUAGE", "es")
    monkeypatch.setenv("CHUNK_LENGTH_SECONDS", "123")
    monkeypatch.setenv("MIN_CONFIDENCE", "0.42")
    monkeypatch.setenv("MAX_GAP_COMBINE_MS", "999")
    monkeypatch.setenv("EMIT_CLEAN_TRANSCRIPT", "true")
    monkeypatch.setenv("EMIT_SUBTITLES", "true")
    monkeypatch.setenv("VERBOSE", "true")
    monkeypatch.setenv("DEBUG_DUMP_AUDIO", "true")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WHISPER_BACKEND", "openai_api")
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("FORCE", "true")

    config = load_config_from_args(args)

    assert config.mode == "bleep"
    assert config.audio_language == "es"
    assert config.chunk_length_seconds == 123
    assert config.min_confidence == pytest.approx(0.42)
    assert config.max_gap_combine_ms == 999
    assert config.emit_clean_transcript is True
    assert config.emit_subtitles is True
    assert config.verbose is True
    assert config.debug_dump_audio is True
    assert config.output_dir == (tmp_path / "logs").resolve()
    assert config.whisper_backend == "openai_api"
    assert config.whisper_model == "small"
    assert config.force is True


@pytest.mark.parametrize("raw", ["1", "0", "yes", "no", "on", "off", "t", "f"])
def test_non_canonical_boolean_env_raises(tmp_path, monkeypatch, raw):
    input_file = tmp_path / "in.mp4"
    input_file.write_bytes(b"x")
    args = _base_args(input_path=str(input_file), output_path=str(tmp_path / "out.mp4"))

    monkeypatch.setenv("EMIT_SUBTITLES", raw)

    with pytest.raises(ValueError, match=r"Expected 'true' or 'false'"):
        load_config_from_args(args)


def test_cli_overrides_env_when_both_provided(tmp_path, monkeypatch):
    input_file = tmp_path / "in.mp4"
    input_file.write_bytes(b"x")

    args = _base_args(input_path=str(input_file), output_path=str(tmp_path / "out.mp4"))

    # Env says bleep, CLI says mute.
    monkeypatch.setenv("MODE", "bleep")
    args.mode = "mute"

    # Env says 10, CLI says 20
    monkeypatch.setenv("CHUNK_LENGTH_SECONDS", "10")
    args.chunk_length_seconds = 20

    # Env says true, CLI flag omitted => None would use env; setting True should override.
    monkeypatch.setenv("EMIT_SUBTITLES", "false")
    args.emit_subtitles = True

    # Env says "small", CLI says "medium"
    monkeypatch.setenv("WHISPER_MODEL", "small")
    args.whisper_model = "medium"

    config = load_config_from_args(args)

    assert config.mode == "mute"
    assert config.chunk_length_seconds == 20
    assert config.emit_subtitles is True
    assert config.whisper_model == "medium"


def test_whisper_model_default_is_base(tmp_path):
    input_file = tmp_path / "in.mp4"
    input_file.write_bytes(b"x")

    args = _base_args(input_path=str(input_file), output_path=str(tmp_path / "out.mp4"))
    config = load_config_from_args(args)

    assert config.whisper_model == "base"


def test_invalid_boolean_env_raises(tmp_path, monkeypatch):
    input_file = tmp_path / "in.mp4"
    input_file.write_bytes(b"x")
    args = _base_args(input_path=str(input_file), output_path=str(tmp_path / "out.mp4"))

    monkeypatch.setenv("EMIT_SUBTITLES", "maybe")

    with pytest.raises(ValueError, match=r"Invalid boolean for EMIT_SUBTITLES"):
        load_config_from_args(args)


def test_invalid_whisper_backend_env_raises(tmp_path, monkeypatch):
    input_file = tmp_path / "in.mp4"
    input_file.write_bytes(b"x")
    args = _base_args(input_path=str(input_file), output_path=str(tmp_path / "out.mp4"))

    monkeypatch.setenv("WHISPER_BACKEND", "definitely-not-a-backend")

    with pytest.raises(ValueError, match=r"Invalid WHISPER_BACKEND value"):
        load_config_from_args(args)


@pytest.mark.parametrize("raw", ["openai", "openai-api", "local", "local-whisper", "LOCAL WHISPER"])
def test_non_canonical_whisper_backend_env_raises(tmp_path, monkeypatch, raw):
    input_file = tmp_path / "in.mp4"
    input_file.write_bytes(b"x")
    args = _base_args(input_path=str(input_file), output_path=str(tmp_path / "out.mp4"))

    monkeypatch.setenv("WHISPER_BACKEND", raw)

    with pytest.raises(ValueError, match=r"Expected one of: local_whisper, openai_api"):
        load_config_from_args(args)
