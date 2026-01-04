from __future__ import annotations

from src.domain import ProfanitySpan
from src.video_tools import build_mute_filter, build_bleep_filter


def test_build_mute_filter_generates_volume_filters():
    spans = [
        ProfanitySpan(start=1.0, end=2.0, hits=[], max_confidence=0.9),
        ProfanitySpan(start=3.5, end=3.9, hits=[], max_confidence=0.8),
    ]
    af = build_mute_filter(spans)
    assert af is not None
    # Mute mode pads the END of each span by 150ms to cover trailing audio.
    assert "volume=enable='between(t,1.000,2.150)'" in af
    assert "volume=enable='between(t,3.500,4.050)'" in af


def test_build_bleep_filter_contains_expected_components():
    spans = [
        ProfanitySpan(start=1.0, end=2.0, hits=[], max_confidence=0.9),
    ]
    fc = build_bleep_filter(spans, sample_rate=16000)
    assert fc is not None
    assert "aevalsrc=0.5*sin(2*PI*1000*t)" in fc
    assert "adelay=" in fc
    assert "amix=inputs=" in fc
