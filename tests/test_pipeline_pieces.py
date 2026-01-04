from __future__ import annotations

from pathlib import Path

from src.config import AppConfig
from src.domain import TranscriptWord, TranscriptSegment, TranscriptionResult
from src.profanity_detector import detect_profanity, load_profanity_terms, merge_profanity_spans
from src.transcriber import build_clean_transcript


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


def test_profane_segment_flow_to_spans_and_clean_text():
    cfg = _dummy_config(min_confidence=0.0)
    terms = load_profanity_terms(cfg)

    seg = TranscriptSegment(
        id=0,
        start=0.0,
        end=5.0,
        text="this is shit man",
        words=[
            TranscriptWord(word="this", start=0.0, end=0.4, confidence=1.0),
            TranscriptWord(word="is", start=0.4, end=0.8, confidence=1.0),
            TranscriptWord(word="shit", start=0.8, end=1.2, confidence=0.95),
            TranscriptWord(word="man", start=1.2, end=1.6, confidence=1.0),
        ],
        avg_confidence=0.95,
        chunk_index=0,
    )
    result = TranscriptionResult(segments=[seg], language="en", raw_responses=[])

    hits = detect_profanity(result, terms, cfg)
    spans = merge_profanity_spans(hits, max_gap_ms=500)

    assert spans, "Expected at least one profanity span"
    clean = build_clean_transcript(result, spans)
    assert "shit" not in clean.lower()
    assert "****" in clean

