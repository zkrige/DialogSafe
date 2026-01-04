from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
import re


# -----------------------
# Transcription domain models
# -----------------------


@dataclass
class TranscriptWord:
    """Single word with timing and confidence."""

    word: str
    start: float  # absolute seconds
    end: float
    confidence: float


@dataclass
class TranscriptSegment:
    """A contiguous chunk of recognized speech."""

    id: int
    start: float  # absolute seconds
    end: float
    text: str
    words: List[TranscriptWord]
    avg_confidence: float
    chunk_index: int


@dataclass
class TranscriptionResult:
    """
    Aggregated transcription for a full audio source.

    raw_responses can carry provider-specific metadata for debugging.
    """

    segments: List[TranscriptSegment]
    language: str
    raw_responses: List[dict[str, Any]]


# -----------------------
# Profanity detection domain models
# -----------------------


@dataclass
class ProfanityTerm:
    """Canonical profanity term (normalized to lowercase)."""

    text: str  # canonical lowercase form

    @property
    def pattern(self) -> re.Pattern[str]:
        """
        Compiled regex for this term with word boundaries.

        Kept here so profanity-related behaviour (word-boundary matching)
        lives on the domain model rather than being duplicated in callers.
        """
        return re.compile(rf"\b{re.escape(self.text)}\b", re.IGNORECASE)


@dataclass
class ProfanityHit:
    """
    A single profanity occurrence detected in the transcript.
    """

    word: str
    start: float
    end: float
    confidence: float
    context: str
    segment_id: int
    chunk_index: int


@dataclass
class ProfanitySpan:
    """
    A merged profanity span, potentially representing several nearby hits.
    """

    start: float
    end: float
    hits: List[ProfanityHit]
    max_confidence: float

    @property
    def representative_word(self) -> str:
        if not self.hits:
            return ""
        # Highest-confidence word in the span
        return max(self.hits, key=lambda h: h.confidence).word