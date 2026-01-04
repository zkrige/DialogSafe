from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .config import AppConfig, Mode
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

        [0:a]anull[a0];
        aevalsrc=... [t0];[t0]adelay=... [b0];
        [a0][b0]amix=inputs=2[aout]

    For multiple spans, multiple tone streams are created and mixed together.
    """
    if not spans:
        return None

    chains: List[str] = []

    # Base audio
    chains.append("[0:a]anull[a0]")

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


def apply_audio_filters_and_mux(
    input_video: Path,
    output_video: Path,
    spans: Sequence[ProfanitySpan],
    config: AppConfig,
) -> None:
    """
    Apply mute or bleep filters to the audio of input_video and write output_video.

    - In mute mode, a simple -af volume filter is used.
    - In bleep mode, a synthetic tone is overlaid via -filter_complex.
    """
    input_video = input_video.expanduser().resolve()
    output_video = output_video.expanduser().resolve()

    if not spans:
        # No profanity: copy streams as-is for speed.
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-c",
            "copy",
            str(output_video),
        ]
    else:
        if config.mode == "mute":
            af = build_mute_filter(spans)
            if not af:
                # Fallback: copy if filter failed to build.
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_video),
                    "-c",
                    "copy",
                    str(output_video),
                ]
            else:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_video),
                    "-af",
                    af,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    str(output_video),
                ]
        else:
            # Bleep mode
            fc = build_bleep_filter(spans)
            if not fc:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_video),
                    "-c",
                    "copy",
                    str(output_video),
                ]
            else:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_video),
                    "-filter_complex",
                    fc,
                    "-map",
                    "0:v",
                    "-map",
                    "[aout]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    str(output_video),
                ]

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
