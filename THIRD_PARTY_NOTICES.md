# Third-Party Notices

This project includes or depends on third-party software. Licenses and attributions for major components are summarized below.

> Note: This file is intentionally brief and human-curated (not an auto-generated dependency dump).

## Major components

### OpenAI Whisper (local transcription)

- Source: https://github.com/openai/whisper
- License: MIT

This project installs Whisper via `requirements.txt` using a git dependency.

### FFmpeg (media processing)

- Website: https://ffmpeg.org/
- License: FFmpeg is typically distributed under LGPLv2.1+ or GPLv2+ depending on build configuration.

FFmpeg is **not bundled** with this repository; users install it separately and the tool calls it as an external program.

## Python dependencies (from requirements.txt)

The following are direct Python dependencies listed in [`requirements.txt`](requirements.txt:1) (license identifiers are provided where commonly known):

- `openai` — MIT (OpenAI Python SDK)
- `tqdm` — MPL-2.0
- `python-dotenv` — BSD-3-Clause
- `pytest` — MIT
- `whisper` (git dependency: `openai/whisper`) — MIT

If any license information above is incorrect for the exact version you are using, the dependency's own license files and metadata are authoritative.

