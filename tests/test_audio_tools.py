from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import pytest

from src.audio_tools import chunk_audio


def _create_dummy_wav(path: Path, duration_sec: float = 2.0, sample_rate: int = 16000) -> None:
    n_frames = int(duration_sec * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def test_chunk_audio_splits_into_expected_chunks():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        wav_path = tmpdir_path / "audio.wav"
        _create_dummy_wav(wav_path, duration_sec=2.0, sample_rate=16000)

        chunks_dir = tmpdir_path / "chunks"
        chunks = chunk_audio(wav_path, chunk_length_seconds=1, temp_dir=chunks_dir)

        assert len(chunks) == 2
        assert chunks[0].index == 0
        assert pytest.approx(chunks[0].start_time, rel=1e-3) == 0.0
        assert chunks[1].index == 1
        assert chunks[1].start_time > chunks[0].start_time
        assert chunks[0].duration > 0
        assert chunks[1].duration > 0

