from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI


# Lazily created global OpenAI client.
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Construct (once) and return the OpenAI client.

    Loads environment variables from a .env file (if present) so that
    OPENAI_API_KEY set in .env is picked up automatically.
    """

    global _client
    if _client is None:
        load_dotenv()
        _client = OpenAI()
    return _client


def transcribe_audio(
    audio_path: Path,
    *,
    language: str,
    model: str,
) -> Dict:
    """Transcribe via OpenAI API and return a `verbose_json`-like dict."""

    cli = get_client()
    ts_arg: List[Literal["segment", "word"]] = ["segment", "word"]

    with audio_path.open("rb") as f:
        response = cli.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            temperature=0.0,
            language=language,
            timestamp_granularities=ts_arg,
        )

    # New OpenAI client returns pydantic-style objects; normalise to dict.
    if hasattr(response, "model_dump"):
        return response.model_dump()  # type: ignore[no-any-return]

    if isinstance(response, dict):
        return response

    # Fallback to generic JSON serialization
    try:
        return json.loads(response.json())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Unexpected transcription response type from OpenAI.") from exc

