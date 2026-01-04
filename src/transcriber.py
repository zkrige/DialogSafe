from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .audio_tools import AudioChunk
from .config import AppConfig
from .domain import TranscriptWord, TranscriptSegment, TranscriptionResult

from .transcription_backends.local_whisper import (
    transcribe_audio as _local_whisper_transcribe_audio,
)
from .transcription_backends.openai_api import (
    transcribe_audio as _openai_api_transcribe_audio,
)

logger = logging.getLogger(__name__)


def _normalize_model_for_backend(model: str, backend: str) -> str:
    """Normalize user-facing model names to backend-specific identifiers.

    Goal: let users set a single WHISPER_MODEL value while keeping sensible
    defaults across backends.
    """

    m = (model or "").strip()
    if not m:
        return m

    if backend == "local_whisper":
        # If the caller provided OpenAI's default model name, translate it
        # into a local Whisper default.
        if m == "whisper-1":
            return "base"
        return m

    # backend == openai_api
    # OpenAI's transcription endpoint does not accept local-whisper size names.
    # If the user used a local size name, fall back to OpenAI's whisper-1.
    if m.lower() in {
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
    }:
        return "whisper-1"

    return m


def _parse_transcript_response(
    raw: Dict[str, Any],
    chunk: AudioChunk,
    chunk_index: int,
) -> tuple[str, List[TranscriptSegment]]:
    """
    Convert a verbose_json transcription into TranscriptSegment objects.

    Timestamps from the API are relative to the chunk; we convert to absolute
    by adding chunk.start_time.
    """
    language = str(raw.get("language") or "unknown")

    segments_out: List[TranscriptSegment] = []
    segments_data = raw.get("segments") or []

    for seg in segments_data:
        try:
            seg_start_rel = float(seg.get("start", 0.0))
        except Exception:
            seg_start_rel = 0.0
        try:
            seg_end_rel = float(seg.get("end", seg_start_rel))
        except Exception:
            seg_end_rel = seg_start_rel

        seg_start = seg_start_rel + chunk.start_time
        seg_end = seg_end_rel + chunk.start_time

        words_out: List[TranscriptWord] = []
        confidences: List[float] = []

        words_data = seg.get("words") or []
        for w in words_data:
            w_text = str(w.get("word", "")).strip()
            if not w_text:
                continue

            try:
                w_start_rel = float(w.get("start", seg_start_rel))
            except Exception:
                w_start_rel = seg_start_rel
            try:
                w_end_rel = float(w.get("end", w_start_rel))
            except Exception:
                w_end_rel = w_start_rel

            conf_val = w.get("confidence")
            if conf_val is None:
                # Some models do not expose confidences yet; default to 1.0.
                w_conf = 1.0
            else:
                try:
                    w_conf = float(conf_val)
                except Exception:
                    w_conf = 1.0

            confidences.append(w_conf)

            words_out.append(
                TranscriptWord(
                    word=w_text,
                    start=w_start_rel + chunk.start_time,
                    end=w_end_rel + chunk.start_time,
                    confidence=w_conf,
                )
            )

        if confidences:
            avg_conf = sum(confidences) / len(confidences)
        else:
            seg_conf = seg.get("confidence", 1.0)
            try:
                avg_conf = float(seg_conf)
            except Exception:
                avg_conf = 1.0

        segment = TranscriptSegment(
            id=int(seg.get("id", len(segments_out))),
            start=seg_start,
            end=seg_end,
            text=str(seg.get("text", "")).strip(),
            words=words_out,
            avg_confidence=avg_conf,
            chunk_index=chunk_index,
        )
        segments_out.append(segment)

    return language, segments_out


def transcribe_chunk(
    chunk: AudioChunk,
    config: AppConfig,
    primary_model: str | None = None,
    fallback_model: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """
    Transcribe a single audio chunk with retries and optional fallback model.

    NOTE:
    - We use "whisper-1" as the default model because it supports
      `response_format="verbose_json"` with segment/word timestamps.
    - "gpt-4o-mini-transcribe" currently does NOT support "verbose_json" and
      will return 400s if used with that response_format, so we avoid it here.

    Returns a dict with:
        {
          "chunk_index": int,
          "language": str,
          "segments": List[TranscriptSegment],
          "raw": Dict[str, Any],
        }
    """
    backend = getattr(config, "whisper_backend", "openai_api")

    # If not overridden per-call, use config (CLI > env > defaults).
    configured_model = str(getattr(config, "whisper_model", "base") or "base")
    effective_primary = primary_model if primary_model is not None else configured_model
    effective_fallback = fallback_model if fallback_model is not None else effective_primary

    effective_primary = _normalize_model_for_backend(effective_primary, backend)
    effective_fallback = _normalize_model_for_backend(effective_fallback, backend)

    models_to_try: Sequence[str] = (effective_primary, effective_fallback)

    last_err: Optional[Exception] = None
    for model in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Transcribing chunk %s with model %s (attempt %s/%s)",
                    chunk.index,
                    model,
                    attempt,
                    max_retries,
                )
                if backend == "local_whisper":
                    raw = _local_whisper_transcribe_audio(
                        chunk.path,
                        language=config.audio_language,
                        model=model,
                    )
                else:
                    raw = _openai_api_transcribe_audio(
                        chunk.path,
                        language=config.audio_language,
                        model=model,
                    )

                language, segments = _parse_transcript_response(raw, chunk, chunk.index)

                # If model reports unknown language, try fallback if available.
                if language == "unknown" and model != fallback_model:
                    logger.warning(
                        "Chunk %s returned unknown language with model %s; "
                        "will retry with fallback model.",
                        chunk.index,
                        model,
                    )
                    break

                return {
                    "chunk_index": chunk.index,
                    "language": language,
                    "segments": segments,
                    "raw": raw,
                }
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning(
                    "Transcription failed for chunk %s with model %s on attempt %s/%s: %s",
                    chunk.index,
                    model,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)

        # move to next model
        continue

    raise RuntimeError(
        f"Transcription failed for chunk {chunk.index} after retries: {last_err}"
    )


def transcribe_audio_chunks(
    chunks: List[AudioChunk],
    config: AppConfig,
    max_workers: Optional[int] = None,
) -> TranscriptionResult:
    """
    Transcribe all audio chunks in parallel and aggregate results.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_segments: List[TranscriptSegment] = []
    languages: List[str] = []
    raw_responses: List[Dict[str, Any]] = []

    # Simple heuristic for workers if not specified
    if max_workers is None:
        try:
            import multiprocessing

            max_workers = max(1, min(8, multiprocessing.cpu_count()))
        except Exception:
            max_workers = 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(transcribe_chunk, chunk, config): chunk for chunk in chunks
        }

        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to transcribe chunk %s: %s", chunk.index, exc)
                continue

            languages.append(result["language"])
            raw_responses.append(result["raw"])
            all_segments.extend(result["segments"])

    # Sort segments by time
    all_segments.sort(key=lambda s: (s.start, s.chunk_index, s.id))

    # Determine dominant language (first non-unknown)
    language = "unknown"
    for lang in languages:
        if lang and lang != "unknown":
            language = lang
            break

    return TranscriptionResult(
        segments=all_segments,
        language=language,
        raw_responses=raw_responses,
    )


def save_transcript_json(result: TranscriptionResult, path: Path) -> None:
    """
    Persist transcript (segments + words) to JSON for downstream analysis.
    """
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "language": result.language,
        "segments": [],
    }

    for seg in result.segments:
        seg_dict: Dict[str, Any] = asdict(seg)
        # asdict will already expand nested dataclasses like TranscriptWord
        data["segments"].append(seg_dict)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_clean_transcript(
    result: TranscriptionResult,
    profanity_spans: Iterable[Any],
    mask_token: str = "****",
) -> str:
    """
    Build a plain-text transcript with profanity masked.

    profanity_spans elements are expected to expose .start and .end attributes
    (or 'start'/'end' dict keys) representing absolute seconds.
    """
    spans: List[tuple[float, float]] = []
    for span in profanity_spans:
        if hasattr(span, "start") and hasattr(span, "end"):
            spans.append((float(span.start), float(span.end)))
        elif isinstance(span, dict) and "start" in span and "end" in span:
            spans.append((float(span["start"]), float(span["end"])))

    spans.sort()

    def is_in_span(t: float) -> bool:
        for s_start, s_end in spans:
            if s_start <= t <= s_end:
                return True
        return False

    lines: List[str] = []

    for seg in result.segments:
        if not seg.words:
            lines.append(seg.text)
            continue

        tokens: List[str] = []
        for w in seg.words:
            if is_in_span(w.start) or is_in_span(w.end):
                tokens.append(mask_token)
            else:
                tokens.append(w.word)

        lines.append(" ".join(tokens))

    return "\n".join(lines)
