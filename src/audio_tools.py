from __future__ import annotations

import contextlib
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class AudioChunk:
    index: int
    path: Path
    start_time: float  # seconds in full audio
    duration: float  # seconds


def extract_mono_pcm_audio(input_video: Path, output_wav: Path, sample_rate: int = 16000) -> Path:
    """
    Extract mono 16k PCM audio from input video using ffmpeg.

    Equivalent to:
        ffmpeg -i INPUT_FILE -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
    """

    output_wav = output_wav.expanduser().resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_wav),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed with code {result.returncode}: {result.stderr}"
        )

    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not produce expected audio file: {output_wav}")

    return output_wav


def _compute_duration_seconds(wav_path: Path) -> float:
    with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def chunk_audio(
    input_wav: Path,
    chunk_length_seconds: int,
    temp_dir: Path,
) -> List[AudioChunk]:
    """
    Split audio.wav into N chunks of roughly `chunk_length_seconds`.

    Returns a list of AudioChunk objects with absolute start offsets.
    """

    input_wav = input_wav.expanduser().resolve()
    temp_dir = temp_dir.expanduser().resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[AudioChunk] = []

    with contextlib.closing(wave.open(str(input_wav), "rb")) as src:
        n_channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        n_frames = src.getnframes()

        if n_channels != 1:
            raise RuntimeError(
                f"Expected mono audio (1 channel), got {n_channels} channels."
            )

        frames_per_chunk = int(chunk_length_seconds * framerate)
        if frames_per_chunk <= 0:
            raise ValueError("chunk_length_seconds must be positive.")

        total_chunks = (n_frames + frames_per_chunk - 1) // frames_per_chunk

        # Whisper requires a minimum audio duration (~0.1s). Very short tail
        # chunks will cause 400 "audio_too_short" errors, so we skip them.
        min_chunk_duration_seconds = 0.11

        for index in range(total_chunks):
            start_frame = index * frames_per_chunk
            src.setpos(start_frame)

            frames_to_read = min(frames_per_chunk, n_frames - start_frame)
            if frames_to_read <= 0:
                break

            duration = frames_to_read / float(framerate)
            if duration < min_chunk_duration_seconds:
                # Read and discard frames to advance the file pointer, but do not
                # emit a chunk that the STT API would reject as too short.
                src.readframes(frames_to_read)
                continue

            audio_data = src.readframes(frames_to_read)

            chunk_path = temp_dir / f"chunk_{index:03d}.wav"
            with contextlib.closing(wave.open(str(chunk_path), "wb")) as dst:
                dst.setnchannels(n_channels)
                dst.setsampwidth(sampwidth)
                dst.setframerate(framerate)
                dst.writeframes(audio_data)

            start_time = start_frame / float(framerate)

            chunks.append(
                AudioChunk(
                    index=index,
                    path=chunk_path,
                    start_time=start_time,
                    duration=duration,
                )
            )

    return chunks