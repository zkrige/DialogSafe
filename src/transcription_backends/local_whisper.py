from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


# Cache local Whisper models (loaded via the `whisper` python package).
_local_whisper_models: Dict[str, Any] = {}

# NOTE:
# Chunk transcription runs in a ThreadPoolExecutor (see `transcribe_audio_chunks`).
# The upstream `whisper` model object is not guaranteed to be thread-safe for
# concurrent `model.transcribe(...)` calls. In practice, concurrent inference
# can trigger sporadic PyTorch shape mismatch / NoneType failures.
#
# We prevent that by serializing *per model name* calls to `transcribe`.
_local_whisper_models_lock = threading.Lock()
_local_whisper_transcribe_locks: Dict[str, threading.Lock] = {}


def _get_transcribe_lock(model_name: str) -> threading.Lock:
    """Return a per-model lock used to serialize model.transcribe calls."""

    model_name = (model_name or "base").strip()
    with _local_whisper_models_lock:
        lock = _local_whisper_transcribe_locks.get(model_name)
        if lock is None:
            lock = threading.Lock()
            _local_whisper_transcribe_locks[model_name] = lock
        return lock


def _get_local_whisper_model(model_name: str) -> Any:
    """Load and cache an OpenAI Whisper model (local inference)."""

    model_name = (model_name or "base").strip()

    # Protect cache initialization: chunking uses threads, and we don't want
    # multiple threads to race to load the same model simultaneously.
    with _local_whisper_models_lock:
        if model_name in _local_whisper_models:
            return _local_whisper_models[model_name]

        # Import lazily so the project can still run with the OpenAI API backend
        # even if local Whisper is not installed.
        import whisper  # type: ignore

        mdl = whisper.load_model(model_name)
        _local_whisper_models[model_name] = mdl
        return mdl


def _normalize_local_whisper_result_to_verbose_json(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert `whisper` result dict into an OpenAI-like `verbose_json` shape.

    This lets the rest of the pipeline reuse the same parsing logic.
    """

    language = str(raw.get("language") or "unknown")
    segments_in = raw.get("segments") or []

    segments_out: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments_in):
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end = float(seg.get("end", seg_start) or seg_start)
        seg_text = str(seg.get("text", "")).strip()

        words_in = seg.get("words") or []
        words_out: List[Dict[str, Any]] = []
        confidences: List[float] = []
        for w in words_in:
            w_text = str(w.get("word", "")).strip()
            if not w_text:
                continue

            w_start = float(w.get("start", seg_start) or seg_start)
            w_end = float(w.get("end", w_start) or w_start)

            # `whisper` uses `probability` for word-level confidence.
            prob = w.get("probability")
            try:
                w_conf = float(prob) if prob is not None else 1.0
            except Exception:
                w_conf = 1.0

            confidences.append(w_conf)
            words_out.append(
                {
                    "word": w_text,
                    "start": w_start,
                    "end": w_end,
                    "confidence": w_conf,
                }
            )

        if confidences:
            seg_conf = sum(confidences) / len(confidences)
        else:
            # Fall back to a conservative default when word-level confidences are missing.
            seg_conf = 1.0

        segments_out.append(
            {
                "id": int(seg.get("id", i)),
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "confidence": seg_conf,
                "words": words_out,
            }
        )

    return {
        "language": language,
        "segments": segments_out,
    }


def transcribe_audio(
    audio_path: Path,
    *,
    language: str,
    model: str,
    task: str = "transcribe",
    fp16: bool = False,
) -> Dict[str, Any]:
    """Transcribe via local Whisper and return a `verbose_json`-like dict."""

    mdl = _get_local_whisper_model(model)
    lock = _get_transcribe_lock(model)

    # Helpful diagnostics when running with --verbose / debug logging.
    # (The reported runtime failures are consistent with concurrent calls.)
    t0 = time.monotonic()
    logger.debug(
        "local_whisper: preparing to transcribe audio=%s model=%s model_id=%s thread=%s",
        str(audio_path),
        model,
        id(mdl),
        threading.current_thread().name,
    )

    # Upstream docs pattern:
    #   import whisper
    #   model = whisper.load_model(...)
    #   result = model.transcribe(...)
    with lock:
        waited_s = time.monotonic() - t0
        if waited_s > 0.01:
            logger.debug(
                "local_whisper: waited %.3fs for per-model transcribe lock model=%s thread=%s",
                waited_s,
                model,
                threading.current_thread().name,
            )

        result = mdl.transcribe(
            str(audio_path),
            task=task,
            language=language,
            word_timestamps=True,
            verbose=False,
            fp16=fp16,
        )

    if not isinstance(result, dict):
        raise RuntimeError("Unexpected transcription response type from local Whisper.")

    return _normalize_local_whisper_result_to_verbose_json(result)
