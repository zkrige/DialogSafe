from __future__ import annotations

from pathlib import Path

import pytest

from src.audio_tools import AudioChunk
from src.config import AppConfig
from src.domain import TranscriptWord, TranscriptSegment, TranscriptionResult, ProfanitySpan
from src.transcriber import build_clean_transcript
from src import transcriber as transcriber_mod


def _dummy_config(**overrides) -> AppConfig:
    base = AppConfig(
        input_path=Path("input.mp4"),
        output_path=Path("output.mp4"),
        mode="mute",
        profanity_list_path=None,
        audio_language="en",
        chunk_length_seconds=300,
        min_confidence=0.6,
        max_gap_combine_ms=500,
        bleep_sound_path=None,
        output_dir=Path("out"),
        emit_clean_transcript=True,
        emit_subtitles=False,
        profanity_terms=["fuck", "shit", "ass"],
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_parse_transcript_response_produces_absolute_timestamps():
    chunk = AudioChunk(index=0, path=Path("dummy.wav"), start_time=10.0, duration=5.0)
    raw = {
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 0.5,
                "end": 1.5,
                "text": "hello world",
                "confidence": 0.9,
                "words": [
                    {"word": "hello", "start": 0.5, "end": 0.9, "confidence": 0.95},
                    {"word": "world", "start": 1.0, "end": 1.5, "confidence": 0.9},
                ],
            }
        ],
    }

    language, segments = transcriber_mod._parse_transcript_response(raw, chunk, chunk_index=0)  # type: ignore[attr-defined]

    assert language == "en"
    assert len(segments) == 1
    seg = segments[0]
    assert pytest.approx(seg.start, rel=1e-6) == 10.5
    assert pytest.approx(seg.end, rel=1e-6) == 11.5
    assert len(seg.words) == 2
    assert pytest.approx(seg.words[0].start, rel=1e-6) == 10.5
    assert pytest.approx(seg.words[1].end, rel=1e-6) == 11.5


def test_build_clean_transcript_masks_profanity_spans():
    seg = TranscriptSegment(
        id=0,
        start=0.0,
        end=5.0,
        text="what the fuck is this",
        words=[
            TranscriptWord(word="what", start=0.0, end=0.4, confidence=1.0),
            TranscriptWord(word="the", start=0.4, end=0.7, confidence=1.0),
            TranscriptWord(word="fuck", start=0.7, end=1.0, confidence=0.95),
            TranscriptWord(word="is", start=1.0, end=1.3, confidence=1.0),
            TranscriptWord(word="this", start=1.3, end=1.7, confidence=1.0),
        ],
        avg_confidence=0.95,
        chunk_index=0,
    )
    result = TranscriptionResult(segments=[seg], language="en", raw_responses=[])

    span = ProfanitySpan(
        start=0.7,
        end=1.0,
        hits=[],
        max_confidence=0.95,
    )

    clean = build_clean_transcript(result, [span])
    assert "****" in clean
    assert "fuck" not in clean.lower()


def test_transcribe_chunk_local_backend_translates_default_model(monkeypatch: pytest.MonkeyPatch):
    """When using local Whisper, the OpenAI default model name is translated."""

    cfg = _dummy_config(whisper_backend="local_whisper")
    chunk = AudioChunk(index=0, path=Path("dummy.wav"), start_time=0.0, duration=1.0)

    captured: dict[str, str] = {}

    def fake_local_backend(audio_path: Path, *, language: str, model: str):
        captured["language"] = language
        captured["model"] = model
        return {"language": language, "segments": []}

    monkeypatch.setattr(transcriber_mod, "_local_whisper_transcribe_audio", fake_local_backend)

    res = transcriber_mod.transcribe_chunk(chunk, cfg)
    assert res["language"] == "en"
    assert captured["model"] == "base"  # translated from whisper-1


def test_transcribe_chunk_openai_backend_keeps_default_model(monkeypatch: pytest.MonkeyPatch):
    cfg = _dummy_config(whisper_backend="openai_api")
    chunk = AudioChunk(index=0, path=Path("dummy.wav"), start_time=0.0, duration=1.0)

    captured: dict[str, str] = {}

    def fake_openai_backend(audio_path: Path, *, language: str, model: str):
        captured["language"] = language
        captured["model"] = model
        return {"language": language, "segments": []}

    monkeypatch.setattr(transcriber_mod, "_openai_api_transcribe_audio", fake_openai_backend)

    res = transcriber_mod.transcribe_chunk(chunk, cfg)
    assert res["language"] == "en"
    assert captured["model"] == "whisper-1"

