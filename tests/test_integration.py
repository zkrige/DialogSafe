from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from src import main as main_mod
from src.audio_tools import AudioChunk
from src.config import AppConfig
from src.domain import (
    TranscriptWord,
    TranscriptSegment,
    TranscriptionResult,
    ProfanityHit,
    ProfanitySpan,
)


def _make_dummy_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000) -> None:
    """Create a small mono PCM WAV file for tests."""
    n_frames = int(duration_sec * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _fake_transcription_result() -> TranscriptionResult:
    """Build a small transcript containing a known profanity."""
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
    return TranscriptionResult(
        segments=[seg],
        language="en",
        raw_responses=[],
    )


def _fake_spans() -> list[ProfanitySpan]:
    hit = ProfanityHit(
        word="fuck",
        start=0.7,
        end=1.0,
        confidence=0.95,
        context="what the fuck is this",
        segment_id=0,
        chunk_index=0,
    )
    return [
        ProfanitySpan(
            start=0.7,
            end=1.0,
            hits=[hit],
            max_confidence=0.95,
        )
    ]


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    """
    Common patching logic for integration tests.

    Patches the functions that src.main.run_pipeline imports directly:
      - extract_mono_pcm_audio
      - chunk_audio
      - transcribe_audio_chunks
      - apply_audio_filters_and_mux

    All patches are applied on the src.main module, so run_pipeline() uses them.
    """
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "safe.mp4"
    out_dir = tmp_path / "out"

    input_video.write_bytes(b"dummy-video-content")

    # Patch extract_mono_pcm_audio: create a dummy wav and return its path.
    def fake_extract(input_video_path: Path, output_wav: Path, sample_rate: int = 16000) -> Path:
        _make_dummy_wav(output_wav, duration_sec=1.0, sample_rate=sample_rate)
        return output_wav

    monkeypatch.setattr(main_mod, "extract_mono_pcm_audio", fake_extract)

    # Patch chunk_audio: return a single AudioChunk pointing at the dummy wav.
    def fake_chunk_audio(input_wav: Path, chunk_length_seconds: int, temp_dir: Path) -> list[AudioChunk]:
        temp_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = temp_dir / "chunk_000.wav"
        if not chunk_path.exists():
            _make_dummy_wav(chunk_path, duration_sec=1.0, sample_rate=16000)
        return [
            AudioChunk(
                index=0,
                path=chunk_path,
                start_time=0.0,
                duration=1.0,
            )
        ]

    monkeypatch.setattr(main_mod, "chunk_audio", fake_chunk_audio)

    # Patch transcribe_audio_chunks: avoid real OpenAI calls.
    def fake_transcribe_audio_chunks(
        chunks: list[AudioChunk],
        config: AppConfig,
        max_workers: int | None = None,
    ) -> TranscriptionResult:
        return _fake_transcription_result()

    monkeypatch.setattr(main_mod, "transcribe_audio_chunks", fake_transcribe_audio_chunks)

    # Patch apply_audio_filters_and_mux: avoid real ffmpeg calls.
    def fake_apply_audio_filters_and_mux(
        input_video: Path,
        output_video: Path,
        spans: list[ProfanitySpan],
        config: AppConfig,
    ) -> None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        # Write a small dummy file representing the processed video.
        output_video.write_bytes(b"dummy-safe-video")

    monkeypatch.setattr(main_mod, "apply_audio_filters_and_mux", fake_apply_audio_filters_and_mux)

    return input_video, output_video, out_dir


@pytest.mark.integration
def test_full_pipeline_mute_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    End-to-end test for the main pipeline in mute mode.

    Uses the real CLI entrypoint but patches:
      - ffmpeg audio extraction
      - audio chunking
      - OpenAI STT
      - final ffmpeg mux

    This verifies:
      - CLI & config wiring
      - profanity detection & span merging
      - censor_log + transcript outputs
    """
    input_video, output_video, out_dir = _patch_pipeline(monkeypatch, tmp_path)

    argv = [
        "--input",
        str(input_video),
        "--output",
        str(output_video),
        "--mode",
        "mute",
        "--audio-language",
        "en",
        "--chunk-length-seconds",
        "60",
        "--min-confidence",
        "0.0",
        "--max-gap-combine-ms",
        "500",
        "--emit-clean-transcript",
        "--emit-subtitles",
        "--output-dir",
        str(out_dir),
    ]

    exit_code = main_mod.main(argv)

    # Assert: pipeline completed successfully.
    assert exit_code == 0
    assert output_video.exists()

    transcript_path = out_dir / "transcript.json"
    censor_log_path = out_dir / "censor_log.json"
    clean_transcript_path = out_dir / "transcript_clean.txt"
    subtitles_path = out_dir / "censored_subtitles.srt"

    assert transcript_path.exists()
    assert censor_log_path.exists()
    assert clean_transcript_path.exists()
    assert subtitles_path.exists()

    # Check censor_log has at least one entry for the known profanity word.
    log_entries = json.loads(censor_log_path.read_text(encoding="utf-8"))
    assert isinstance(log_entries, list)
    assert log_entries, "Expected at least one profanity entry in censor_log.json"
    words = {entry.get("word") for entry in log_entries}
    assert "fuck" in words


@pytest.mark.integration
def test_full_pipeline_bleep_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    End-to-end-ish test for the main pipeline in bleep mode.

    Reuses the same faked pipeline as mute mode but runs with --mode bleep
    and a different output path.
    """
    input_video, output_video, out_dir = _patch_pipeline(monkeypatch, tmp_path)
    output_video = out_dir / "safe-bleep.mp4"

    argv = [
        "--input",
        str(input_video),
        "--output",
        str(output_video),
        "--mode",
        "bleep",
        "--audio-language",
        "en",
        "--chunk-length-seconds",
        "60",
        "--min-confidence",
        "0.0",
        "--max-gap-combine-ms",
        "500",
        "--emit-clean-transcript",
        "--emit-subtitles",
        "--output-dir",
        str(out_dir),
    ]

    exit_code = main_mod.main(argv)
    assert exit_code == 0
    assert output_video.exists()

    transcript_path = out_dir / "transcript.json"
    censor_log_path = out_dir / "censor_log.json"

    assert transcript_path.exists()
    assert censor_log_path.exists()

    log_entries = json.loads(censor_log_path.read_text(encoding="utf-8"))
    assert any(entry.get("word") == "fuck" for entry in log_entries)