import threading
import time
from concurrent.futures import ThreadPoolExecutor


def test_local_whisper_transcribe_is_serialized_per_model(monkeypatch, tmp_path):
    """Regression test for sporadic local Whisper failures under chunk concurrency.

    The pipeline uses a ThreadPoolExecutor to transcribe chunks concurrently.
    The upstream `whisper` model object is not guaranteed to be thread-safe for
    concurrent `model.transcribe(...)` calls.

    This test monkeypatches the model to raise if `transcribe` is entered while
    another thread is still inside it. The backend must prevent that.
    """

    from src.transcription_backends import local_whisper as lw

    class FakeModel:
        def __init__(self):
            self._guard = threading.Lock()
            self._in_call = False
            self.concurrent_detected = 0

        def transcribe(self, *_args, **_kwargs):
            with self._guard:
                if self._in_call:
                    self.concurrent_detected += 1
                    raise RuntimeError("concurrent transcribe detected")
                self._in_call = True

            try:
                # Make overlap likely if the backend does not serialize calls.
                time.sleep(0.05)
                return {"language": "en", "segments": []}
            finally:
                with self._guard:
                    self._in_call = False

    fake_model = FakeModel()

    # Ensure we don't inherit state from other tests.
    lw._local_whisper_transcribe_locks.clear()

    monkeypatch.setattr(lw, "_get_local_whisper_model", lambda _model_name: fake_model)

    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"not a real wav")

    def do_call():
        return lw.transcribe_audio(audio_path, language="en", model="base")

    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(lambda _: do_call(), range(2)))

    assert fake_model.concurrent_detected == 0
    assert results == [
        {"language": "en", "segments": []},
        {"language": "en", "segments": []},
    ]

