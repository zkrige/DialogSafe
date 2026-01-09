from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

from .audio_tools import chunk_audio, extract_mono_pcm_audio
from .config import AppConfig, load_config_from_args
from .profanity_detector import (
    build_censor_log,
    build_subtitles,
    load_profanity_terms,
    detect_profanity,
    merge_profanity_spans,
)
from .transcriber import (
    build_clean_transcript,
    save_transcript_json,
    transcribe_audio_chunks,
)
from .video_tools import apply_audio_filters_and_mux, input_has_clean_track_marker

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated movie profanity stripping system.",
    )

    parser.add_argument("--input", "-i", required=True, help="Input movie file (MP4/MKV/AVI).")
    parser.add_argument("--output", "-o", required=True, help="Output cleaned video filename.")

    parser.add_argument(
        "--mode",
        choices=["mute", "bleep"],
        default=None,
        help="Censorship mode: 'mute' or 'bleep' (default: mute).",
    )

    parser.add_argument(
        "--profanity-list",
        help="Path to JSON file with list of banned words/phrases.",
    )

    parser.add_argument(
        "--audio-language",
        default=None,
        help="Primary spoken language ISO code for transcription (default: en).",
    )

    parser.add_argument(
        "--chunk-length-seconds",
        type=int,
        default=None,
        help="Length of audio chunks in seconds for transcription (default: 300).",
    )

    parser.add_argument(
        "--bleep-sound-path",
        help="Optional WAV beep file (currently unused; synthetic tone is used instead).",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum confidence threshold for profanity hits (default: 0.6).",
    )

    parser.add_argument(
        "--max-gap-combine-ms",
        type=int,
        default=None,
        help="Maximum gap in ms to merge adjacent profanity hits (default: 500).",
    )

    parser.add_argument(
        "--emit-clean-transcript",
        action="store_true",
        default=None,
        help="Emit a clean transcript with profanity masked.",
    )

    parser.add_argument(
        "--emit-subtitles",
        action="store_true",
        default=None,
        help="Emit subtitles (SRT) marking censored segments.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON logs and transcripts (default: ./out).",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=None,
        help="Enable verbose logging.",
    )

    parser.add_argument(
        "--debug-dump-audio",
        action="store_true",
        default=None,
        help=(
            "Copy extracted audio.wav and all chunk_XXX.wav files into "
            "the output directory under ./audio_debug for inspection."
        ),
    )

    parser.add_argument(
        "--whisper-model",
        default=None,
        help=(
            "Whisper model name. For local_whisper: tiny/base/small/medium/large. "
            "For openai_api: whisper-1 (or other OpenAI transcription model). "
            "Can also be set via WHISPER_MODEL env var."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=None,
        help=(
            "Force processing even if the input already contains a 'Clean' audio track marker. "
            "Without --force, inputs that appear already processed are skipped to prevent clean-track accumulation."
        ),
    )

    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


def run_pipeline(config: AppConfig) -> None:
    """
    Execute the full profanity-stripping pipeline end-to-end.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Prevent re-processing an already-cleaned output (which would otherwise accumulate extra
    # "Clean" tracks over repeated runs).
    if not getattr(config, "force", False) and input_has_clean_track_marker(config.input_path):
        logger.info(
            "Input already appears to contain a clean-track marker; skipping processing. "
            "Use --force (or FORCE=true) to override."
        )
        return

    transcript_json_path = config.output_dir / "transcript.json"
    censor_log_path = config.output_dir / "censor_log.json"
    clean_transcript_path = config.output_dir / "transcript_clean.txt"
    subtitles_path = config.output_dir / "censored_subtitles.srt"

    logger.info("Output video will be written to: %s", config.output_path)
    logger.info("Logs and transcripts will be written under: %s", config.output_dir)

    with TemporaryDirectory(prefix="profanity_pipeline_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        audio_path = tmpdir / "audio.wav"

        logger.info("Extracting mono PCM audio...")
        extract_mono_pcm_audio(config.input_path, audio_path, sample_rate=16000)

        logger.info("Chunking audio into %s-second segments...", config.chunk_length_seconds)
        chunks_dir = tmpdir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks = chunk_audio(audio_path, config.chunk_length_seconds, chunks_dir)

        # Optional debug: copy the extracted audio and chunks to output_dir/audio_debug
        if getattr(config, "debug_dump_audio", False):
            debug_dir = config.output_dir / "audio_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(audio_path, debug_dir / "audio.wav")
                for chunk in chunks:
                    shutil.copy2(chunk.path, debug_dir / chunk.path.name)
                logger.info("Debug audio dumped to %s", debug_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to copy debug audio files: %s", exc)

        logger.info(
            "Transcribing %d audio chunks with Whisper backend '%s'...",
            len(chunks),
            getattr(config, "whisper_backend", "unknown"),
        )
        transcription_result = transcribe_audio_chunks(chunks, config)

        logger.info("Detected language: %s", transcription_result.language)

        logger.info("Loading profanity terms...")
        profanity_terms = load_profanity_terms(config)

        logger.info("Detecting profanity in transcript...")
        hits = detect_profanity(transcription_result, profanity_terms, config)
        logger.info("Found %d raw profanity hits.", len(hits))

        logger.info(
            "Merging profanity hits into spans (max gap %d ms)...",
            config.max_gap_combine_ms,
        )
        spans = merge_profanity_spans(hits, config.max_gap_combine_ms)
        logger.info("Merged into %d profanity spans.", len(spans))

        logger.info("Writing transcript JSON to %s", transcript_json_path)
        save_transcript_json(transcription_result, transcript_json_path)

        logger.info("Writing censor log JSON to %s", censor_log_path)
        build_censor_log(spans, censor_log_path)

        if config.emit_clean_transcript:
            logger.info("Building clean transcript with profanity masked...")
            clean_text = build_clean_transcript(transcription_result, spans)
            clean_transcript_path.parent.mkdir(parents=True, exist_ok=True)
            clean_transcript_path.write_text(clean_text, encoding="utf-8")

        if config.emit_subtitles:
            logger.info("Generating subtitles for censored spans...")
            build_subtitles(spans, transcription_result, subtitles_path)

        logger.info("Applying audio filters and re-muxing video...")
        apply_audio_filters_and_mux(
            input_video=config.input_path,
            output_video=config.output_path,
            spans=spans,
            config=config,
        )

    logger.info("Finished processing. Cleaned video: %s", config.output_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Setup a baseline logger first; may be overridden once env/merged config is loaded.
    setup_logging(bool(args.verbose))

    try:
        config = load_config_from_args(args)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load configuration: %s", exc)
        return 1

    # Reconfigure logging using merged config (CLI > env > defaults).
    setup_logging(bool(getattr(config, "verbose", False)))

    try:
        run_pipeline(config)
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
