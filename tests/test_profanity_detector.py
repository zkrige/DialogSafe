from __future__ import annotations

from pathlib import Path

import pytest

from src.config import AppConfig
from src.domain import (
    TranscriptWord,
    TranscriptSegment,
    TranscriptionResult,
    ProfanityHit,
    ProfanitySpan,
    ProfanityTerm,
)
from src.profanity_detector import build_censor_log, detect_profanity, load_profanity_terms, merge_profanity_spans


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


def test_load_profanity_terms_wraps_config_terms():
    cfg = _dummy_config()
    terms = load_profanity_terms(cfg)
    # All entries should be ProfanityTerm instances with non-empty, lowercase text.
    assert all(isinstance(t, ProfanityTerm) for t in terms)
    assert all(t.text for t in terms)
    assert all(t.text == t.text.lower() for t in terms)


def test_detect_profanity_respects_word_boundaries():
    cfg = _dummy_config(min_confidence=0.0)
    terms = [ProfanityTerm(text="ass")]

    seg = TranscriptSegment(
        id=0,
        start=0.0,
        end=2.0,
        text="classy ass move",
        words=[
            TranscriptWord(word="classy", start=0.0, end=0.5, confidence=1.0),
            TranscriptWord(word="ass", start=0.5, end=1.0, confidence=0.9),
            TranscriptWord(word="move", start=1.0, end=1.5, confidence=1.0),
        ],
        avg_confidence=0.9,
        chunk_index=0,
    )
    result = TranscriptionResult(segments=[seg], language="en", raw_responses=[])

    hits = detect_profanity(result, terms, cfg)
    assert len(hits) == 1
    assert hits[0].word == "ass"


def test_detect_profanity_keeps_low_confidence_profane_tokens_in_mute_mode():
    """
    Regression test:
    When using --mode mute, we rely on word timestamps to build mute spans.
    Profanity tokens must not be dropped purely because word confidence is low.
    """

    cfg = _dummy_config(min_confidence=0.6, mode="mute")
    terms = [ProfanityTerm(text="fucking")]

    seg = TranscriptSegment(
        id=0,
        start=4203.0,
        end=4204.0,
        text="You're fucking me...",
        words=[
            TranscriptWord(word="You're", start=4203.00, end=4203.10, confidence=0.9),
            TranscriptWord(word="fucking", start=4203.22, end=4203.38, confidence=0.1697),
            TranscriptWord(word="me", start=4203.38, end=4203.50, confidence=0.9),
        ],
        avg_confidence=0.7,
        chunk_index=0,
    )
    result = TranscriptionResult(segments=[seg], language="en", raw_responses=[])

    hits = detect_profanity(result, terms, cfg)
    assert len(hits) == 1
    assert hits[0].word == "fucking"
    assert hits[0].start == pytest.approx(4203.22)
    assert hits[0].end == pytest.approx(4203.38)


def test_merge_profanity_spans_merges_close_hits():
    hits = [
        # first cluster
        ProfanityHit(word="a", start=1.0, end=1.2, confidence=0.9, context="", segment_id=0, chunk_index=0),
        ProfanityHit(word="b", start=1.25, end=1.4, confidence=0.8, context="", segment_id=0, chunk_index=0),
        # far apart
        ProfanityHit(word="c", start=5.0, end=5.2, confidence=0.9, context="", segment_id=0, chunk_index=0),
    ]

    spans = merge_profanity_spans(hits, max_gap_ms=500)
    assert len(spans) == 2
    first, second = spans
    assert first.start == pytest.approx(1.0)
    assert first.end == pytest.approx(1.4)
    assert second.start == pytest.approx(5.0)


def test_build_censor_log_creates_expected_json(tmp_path: Path):
    span = ProfanitySpan(
        start=10.0,
        end=11.0,
        hits=[],
        max_confidence=0.9,
    )
    log_path = tmp_path / "censor_log.json"
    build_censor_log([span], log_path)

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert '"start": 10.0' in content
    assert '"end": 11.0' in content
