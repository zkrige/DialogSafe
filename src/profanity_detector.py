from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import AppConfig
from .domain import (
    TranscriptSegment,
    TranscriptWord,
    TranscriptionResult,
    ProfanityTerm,
    ProfanityHit,
    ProfanitySpan,
)

logger = logging.getLogger(__name__)


def load_profanity_terms(config: AppConfig) -> List[ProfanityTerm]:
    """
    Convert config.profanity_terms list of strings into ProfanityTerm objects.
    """
    terms: List[ProfanityTerm] = []
    for w in config.profanity_terms:
        w = w.strip().lower()
        if not w:
            continue
        terms.append(ProfanityTerm(text=w))
    return terms


def _token_normalize(word: str) -> str:
    return re.sub(r"[^\w']+", "", word.lower())


def detect_profanity(
    transcription: TranscriptionResult,
    profanity_terms: Sequence[ProfanityTerm],
    config: AppConfig,
) -> List[ProfanityHit]:
    """
    Scan transcript segments/words for profanity.

    Rules:
    - Normalize to lowercase.
    - Enforce word boundaries: "ass" should not match "class".
    - Use word-level timestamps when available.
    - Apply min_confidence threshold (except: in --mode mute, keep matched
      profane word tokens even when confidence is low, so mute spans remain
      correctly time-aligned).
    """
    term_texts = {t.text: t for t in profanity_terms}
    hits: List[ProfanityHit] = []

    if not profanity_terms:
        logger.warning("Profanity detection called with empty term list.")
        return hits

    for seg in transcription.segments:
        seg_text_lower = seg.text.lower()
        context_text = seg.text.strip()

        # Build a map from normalized word -> list of (TranscriptWord, original word text)
        word_candidates: List[Tuple[TranscriptWord, str]] = []
        if seg.words:
            for w in seg.words:
                norm = _token_normalize(w.word)
                if not norm:
                    continue
                word_candidates.append((w, norm))

        # Primary path: use word-level alignment
        for w_obj, norm in word_candidates:
            if norm in term_texts:
                conf = float(w_obj.confidence)
                if conf < config.min_confidence:
                    # For mute mode, we must trust timestamps to build censor
                    # intervals, even when the ASR word confidence is low.
                    # Otherwise clear profanities can be missed entirely.
                    if config.mode != "mute":
                        logger.debug(
                            "Skipping profanity token %r (conf=%.3f < min=%.3f) in mode=%s",
                            norm,
                            conf,
                            config.min_confidence,
                            config.mode,
                        )
                        continue
                    logger.debug(
                        "Keeping low-confidence profanity token %r (conf=%.3f < min=%.3f) in mode=%s",
                        norm,
                        conf,
                        config.min_confidence,
                        config.mode,
                    )

                hits.append(
                    ProfanityHit(
                        word=norm,
                        start=float(w_obj.start),
                        end=float(w_obj.end),
                        confidence=conf,
                        context=context_text,
                        segment_id=seg.id,
                        chunk_index=seg.chunk_index,
                    )
                )

        # Fallback: if no word timing present, use regex on the segment text.
        if not seg.words:
            for term in profanity_terms:
                # Pylance sometimes loses track of the `pattern` property on ProfanityTerm
                # when imported from the shared domain module. The attribute does exist,
                # so we silence the spurious type error here.
                for match in term.pattern.finditer(seg_text_lower):  # type: ignore[attr-defined]
                    # Map match to whole segment timing; imperfect but better than nothing.
                    conf = float(seg.avg_confidence)
                    if conf < config.min_confidence:
                        logger.debug(
                            "Skipping segment-level profanity match %r (seg_avg_conf=%.3f < min=%.3f)",
                            term.text,
                            conf,
                            config.min_confidence,
                        )
                        continue

                    hits.append(
                        ProfanityHit(
                            word=term.text,
                            start=float(seg.start),
                            end=float(seg.end),
                            confidence=conf,
                            context=context_text,
                            segment_id=seg.id,
                            chunk_index=seg.chunk_index,
                        )
                    )

    # Sort hits by time for downstream merging
    hits.sort(key=lambda h: (h.start, h.end))
    return hits


def merge_profanity_spans(
    hits: Sequence[ProfanityHit],
    max_gap_ms: int,
) -> List[ProfanitySpan]:
    """
    Merge overlapping or near-adjacent profanity hits into spans.

    - Overlapping timestamps are merged.
    - Spans separated by <= max_gap_ms are merged to avoid choppy censoring.
    """
    if not hits:
        return []

    merged: List[ProfanitySpan] = []

    current_start = hits[0].start
    current_end = hits[0].end
    current_hits: List[ProfanityHit] = [hits[0]]

    def flush() -> None:
        if not current_hits:
            return
        span = ProfanitySpan(
            start=current_start,
            end=current_end,
            hits=list(current_hits),
            max_confidence=max(h.confidence for h in current_hits),
        )
        merged.append(span)

    max_gap_sec = max_gap_ms / 1000.0

    for h in hits[1:]:
        gap = h.start - current_end
        if gap <= max_gap_sec:
            # Extend current span
            current_end = max(current_end, h.end)
            current_hits.append(h)
        else:
            # Close current span and start a new one
            flush()
            current_start = h.start
            current_end = h.end
            current_hits = [h]

    flush()
    return merged


def build_censor_log(
    spans: Sequence[ProfanitySpan],
    path: Path,
) -> None:
    """
    Write censor_log.json with entries like:

    {
      "word": "fuck",
      "start": 123.45,
      "end": 124.02,
      "confidence": 0.93,
      "context": "what the fuck is this"
    }
    """
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    log_entries: List[Dict[str, Any]] = []

    for span in spans:
        # Use the most confident hit for representative context/word
        if span.hits:
            best_hit = max(span.hits, key=lambda h: h.confidence)
            word = best_hit.word
            context = best_hit.context
            confidence = best_hit.confidence
        else:
            word = ""
            context = ""
            confidence = 0.0

        log_entries.append(
            {
                "word": word,
                "start": float(span.start),
                "end": float(span.end),
                "confidence": float(confidence),
                "context": context,
            }
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump(log_entries, f, ensure_ascii=False, indent=2)


def _format_srt_timestamp(t: float) -> str:
    """
    Convert seconds to SRT timestamp: HH:MM:SS,mmm
    """
    if t < 0:
        t = 0.0
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    millis = int(round((t - int(t)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_subtitles(
    spans: Sequence[ProfanitySpan],
    transcript: TranscriptionResult,
    path: Path,
) -> None:
    """
    Generate an SRT subtitle file where each entry corresponds to a profanity span.

    The subtitle text will show a context snippet with profanity masked.
    """
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare for quick span lookup: we'll simply use the best_hit.context and mask all
    # profanity words present in that context.
    lines: List[str] = []
    index = 1

    for span in spans:
        if not span.hits:
            continue

        best_hit = max(span.hits, key=lambda h: h.confidence)
        context = best_hit.context
        if not context:
            continue

        # Mask any profanity term inside context text (simple approach).
        masked_context = context
        for h in span.hits:
            if not h.word:
                continue
            pattern = re.compile(rf"\b{re.escape(h.word)}\b", re.IGNORECASE)
            masked_context = pattern.sub("****", masked_context)

        start_ts = _format_srt_timestamp(span.start)
        end_ts = _format_srt_timestamp(span.end)

        lines.append(str(index))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(masked_context)
        lines.append("")  # blank line

        index += 1

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
