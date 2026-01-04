from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import AppConfig
from .profanity_detector import ProfanitySpan

logger = logging.getLogger(__name__)


# In mute mode, add a small tail to ensure the trailing consonant/air of
# profane words is fully covered.
MUTE_END_PADDING_SEC = 0.150


@dataclass
class FilterSpec:
    """Represents an ffmpeg filter description."""

    af: Optional[str] = None
    filter_complex: Optional[str] = None
    audio_output_label: str = "aout"


def build_mute_filter(spans: Sequence[ProfanitySpan]) -> Optional[str]:
    """
    Build a simple -af volume filter string that mutes all profanity spans.

    Example:
        volume=enable='between(t,START,END)':volume=0,volume=...
    """
    if not spans:
        return None

    parts: List[str] = []
    for span in spans:
        start = max(0.0, float(span.start))
        end = max(start, float(span.end)) + MUTE_END_PADDING_SEC
        parts.append(
            f"volume=enable='between(t,{start:.3f},{end:.3f})':volume=0"
        )

    return ",".join(parts)


def build_bleep_filter(spans: Sequence[ProfanitySpan], sample_rate: int = 16000) -> Optional[str]:
    """
    Build a -filter_complex graph that overlays synthetic beep tones on profanity spans.

    This implementation uses a 1kHz sine tone generated via aevalsrc and
    delayed/mixed over the original audio:

        [0:a:0]anull[a0];
        aevalsrc=... [t0];[t0]adelay=... [b0];
        [a0][b0]amix=inputs=2[aout]

    For multiple spans, multiple tone streams are created and mixed together.
    """
    if not spans:
        return None

    chains: List[str] = []

    # Base audio
    chains.append("[0:a:0]anull[a0]")

    beep_labels: List[str] = []

    for idx, span in enumerate(spans):
        start = max(0.0, float(span.start))
        end = max(start, float(span.end))
        duration = max(0.1, end - start)
        delay_ms = int(round(start * 1000))

        tone_label = f"tone{idx}"
        beep_label = f"b{idx}"

        # Generate tone of given duration
        chains.append(
            f"aevalsrc=0.5*sin(2*PI*1000*t):s={sample_rate}:d={duration:.3f}[{tone_label}]"
        )
        # Delay tone to align with profanity span
        chains.append(
            f"[{tone_label}]adelay={delay_ms}|{delay_ms}[{beep_label}]"
        )

        beep_labels.append(f"[{beep_label}]")

    # Mix original with all beep tracks
    inputs = "[a0]" + "".join(beep_labels)
    n_inputs = 1 + len(beep_labels)
    chains.append(f"{inputs}amix=inputs={n_inputs}:normalize=0[aout]")

    return ";".join(chains)


def probe_primary_audio_codec_and_bitrate(input_video: Path) -> Tuple[Optional[str], Optional[int]]:
    """Return (codec_name, bit_rate) for the primary input audio stream (0:a:0).

    - codec_name is a lowercase ffprobe codec name (e.g. aac, ac3, eac3)
    - bit_rate is in bits/second when available.

    This helper is best-effort: it returns (None, None) on ffprobe failures.
    """

    input_video = input_video.expanduser().resolve()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,bit_rate",
        "-of",
        "json",
        str(input_video),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.warning("ffprobe failed (code=%s): %s", result.returncode, result.stderr)
        return None, None

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        logger.warning("ffprobe returned non-JSON output")
        return None, None

    streams = data.get("streams") or []
    if not streams:
        return None, None

    stream0 = streams[0] or {}
    codec_name_raw = stream0.get("codec_name")
    codec_name = str(codec_name_raw).strip().lower() if codec_name_raw else None

    bit_rate_raw = stream0.get("bit_rate")
    bit_rate: Optional[int]
    if bit_rate_raw is None:
        bit_rate = None
    else:
        try:
            bit_rate = int(str(bit_rate_raw).strip())
        except ValueError:
            bit_rate = None

    return codec_name, bit_rate


def _encoder_for_codec_name(codec_name: Optional[str]) -> str:
    """Return an ffmpeg audio encoder name for a given ffprobe codec name."""

    codec = (codec_name or "").strip().lower()
    mapping = {
        "aac": "aac",
        "ac3": "ac3",
        "eac3": "eac3",
    }

    encoder = mapping.get(codec)
    if encoder:
        return encoder

    if codec:
        logger.warning("Unsupported input audio codec %r; falling back to AAC", codec)
    return "aac"


def _build_ffmpeg_censor_and_mux_cmd(
    *,
    input_video: Path,
    output_video: Path,
    spans: Sequence[ProfanitySpan],
    config: AppConfig,
    primary_audio_codec: Optional[str],
    primary_audio_bit_rate: Optional[int],
) -> List[str]:
    """Build the ffmpeg command used by apply_audio_filters_and_mux()."""

    input_video = input_video.expanduser().resolve()
    output_video = output_video.expanduser().resolve()

    if not spans:
        return [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-c",
            "copy",
            str(output_video),
        ]

    if config.mode == "mute":
        af = build_mute_filter(spans)
        if not af:
            return [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video),
                "-c",
                "copy",
                str(output_video),
            ]
        filter_complex = f"[0:a:0]{af}[aout]"
    else:
        fc = build_bleep_filter(spans)
        if not fc:
            return [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video),
                "-c",
                "copy",
                str(output_video),
            ]
        filter_complex = fc

    encoder = _encoder_for_codec_name(primary_audio_codec)

    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-map",
        "0:a",
        "-map",
        "-0:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-c:a:0",
        encoder,
    ]

    if primary_audio_bit_rate and primary_audio_bit_rate > 0:
        cmd.extend(["-b:a:0", str(primary_audio_bit_rate)])

    cmd.append(str(output_video))
    return cmd


def apply_audio_filters_and_mux(
    input_video: Path,
    output_video: Path,
    spans: Sequence[ProfanitySpan],
    config: AppConfig,
) -> None:
    """
    Apply mute or bleep filters to the audio of input_video and write output_video.

    - When spans are present, filtering is applied only to 0:a:0 via -filter_complex.
    - Other audio streams are preserved via stream copy.
    """
    input_video = input_video.expanduser().resolve()
    output_video = output_video.expanduser().resolve()

    codec_name: Optional[str]
    bit_rate: Optional[int]
    if spans:
        codec_name, bit_rate = probe_primary_audio_codec_and_bitrate(input_video)
    else:
        codec_name, bit_rate = None, None
    cmd = _build_ffmpeg_censor_and_mux_cmd(
        input_video=input_video,
        output_video=output_video,
        spans=spans,
        config=config,
        primary_audio_codec=codec_name,
        primary_audio_bit_rate=bit_rate,
    )

    logger.info("Running ffmpeg for final mux: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg muxing failed with code {result.returncode}: {result.stderr}"
        )
