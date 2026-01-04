from __future__ import annotations

from src.config import load_default_profanity_list


def test_load_default_profanity_list_en_language():
    words = load_default_profanity_list("en")
    assert isinstance(words, list)
    assert words, "Expected non-empty profanity list for English"
    assert all(isinstance(w, str) for w in words)
    assert all(w == w.lower() for w in words), "Expected all words to be lowercase"

