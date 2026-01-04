# Movie Profanity Remover

Automated pipeline that transcribes movie audio with Whisper (local by default), detects profanity, and outputs a cleaned video with censored audio plus logs.

> **NOTE:** Transcription can run entirely on your machine (default: local Whisper). To use the OpenAI API instead, set `WHISPER_BACKEND=openai_api` and provide `OPENAI_API_KEY` (for example in your [`.env`](.env:1) file).

## Features

- Supports common containers (MP4, MKV, AVI) via FFmpeg.
- Modes: mute (silence profanity) or bleep (synthetic 1 kHz tone).
- Uses Whisper speech-to-text with word-level timestamps for precise censoring (configurable backend).
- Outputs `censor_log.json` with word, start, end, confidence, and context.
- Optional clean transcript (profanity masked) and subtitles for censored spans.

## Requirements

- macOS or Linux with FFmpeg in `PATH`.
- Python 3.10+ (you are currently using Python 3.14).
- Local Whisper is the default backend (installed via `pip install -r requirements.txt`).
- OpenAI API key is only required if you set `WHISPER_BACKEND=openai_api`.

## Installation

From the project root:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Note: Whisper is installed inside this project's virtual environment via `requirements.txt`, so no global `pip install -U openai-whisper` is required.

## Environment configuration (.env)

Copy the template and edit it:

```bash
cp .env.example .env
```

The application uses `python-dotenv` and will automatically load the repo-root [`.env`](.env:1) file if present.

Default (local Whisper):

```bash
WHISPER_BACKEND=local_whisper
```

Optional (use OpenAI API speech-to-text instead):

```bash
WHISPER_BACKEND=openai_api
OPENAI_API_KEY=sk-your-key-here
```

You do not need to export variables manually in your shell.

## Running the profanity remover

Basic usage:

```bash
python -m src.main \
  --input movie.mp4 \
  --output out/safe.mp4 \
  --mode mute
```

### Example

```bash
python -m src.main \
  --input "sample-with-profanity.mp4" \
  --output out/clean.mp4 \
  --mode mute \
  --audio-language en \
  --chunk-length-seconds 300 \
  --min-confidence 0.6 \
  --max-gap-combine-ms 500 \
  --emit-clean-transcript \
  --emit-subtitles \
  --verbose \
  --debug-dump-audio
```

Common options:

- `--mode` — `mute` (default) or `bleep`.
- `--audio-language` — spoken language code (default: `en`).
- `--profanity-list` — path to a JSON file with a list of banned words/phrases.
- `--chunk-length-seconds` — audio chunk size for transcription (default: 300).
- `--min-confidence` — minimum confidence for profanity hits (default: 0.6).
- `--max-gap-combine-ms` — merge hits closer than this gap (default: 500 ms).
- `--emit-clean-transcript` — also write a profanity-masked transcript.
- `--emit-subtitles` — write an SRT subtitle file for censored spans.
- `--output-dir` — directory for logs and transcripts (default: `out`).

To see all CLI options:

```bash
python -m src.main --help
```

## Outputs

- Cleaned video at the path given by `--output`.
- `transcript.json` in the output directory, with segments and word timestamps.
- `censor_log.json` in the output directory, listing all detected profanity spans.
- Optional `transcript_clean.txt` with profanity masked.
- Optional `censored_subtitles.srt` for censored spans.

## Running tests

With the virtual environment activated:

```bash
pytest
```

This runs unit tests in `tests/test_core.py` covering chunking, transcription parsing, profanity detection, span merging, log generation, and FFmpeg filter construction.

## Benchmarks

### Prometheus (sample.mkv)

This benchmark was captured by running the CLI with `--verbose` and using the timestamps in the logs.

#### Hardware / software

- Machine: MacBook Pro (Apple M3 Max, 64 GB RAM)
- OS: macOS 26.2
- Transcription: local Whisper model `base`

#### Runtime options

```bash
python -m src.main \
  --input "~/Downloads/sample.mkv" \
  --output out/clean.mkv \
  --mode mute \
  --audio-language en \
  --chunk-length-seconds 300 \
  --min-confidence 0.6 \
  --max-gap-combine-ms 500 \
  --emit-clean-transcript \
  --emit-subtitles \
  --verbose \
  --debug-dump-audio
```

Observed configuration from logs:

- Whisper backend: `local_whisper`
- Whisper model: `base`

#### Input media (relevant for decode/encode)

Input file: `~/Downloads/sample.mkv`

`ffprobe` summary:

- Movie runtime: 02:03:44
- Video: 1440p WEBRip (2560x1066), HEVC/H.265 10-bit, ~23.976 fps
- Audio: English 5.1 (E-AC-3)
- File size: ~8.7 GB

#### Results

- Detected language: `en`
- Profanity hits: 17 raw hits
- Profanity spans: 17 merged spans (max gap: 500 ms)

Timing summary (from log timestamps):

- Transcription: 3m 32s
- Profanity detect + emit logs/subtitles: < 1s
- FFmpeg re-mux (mute filter + AAC): 3m 10s
- End-to-end (observed window): 6m 48s

Notes:

- With the local Whisper backend, `model.transcribe(...)` is serialized per model name for stability, so additional transcription threads may spend time waiting on a per-model lock.

## Notes

- FFmpeg must be installed and available on your `PATH` (try `ffmpeg -version` to verify).
- The tool processes audio in chunks and calls OpenAI in parallel; for long movies, runtime will depend on network speed and OpenAI API performance.

Shell note: when copying multi-line commands, each line continuation backslash (`\`) must be the final character on the line (no trailing spaces).
