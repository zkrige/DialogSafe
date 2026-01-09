#!/usr/bin/env bash
set -euo pipefail

# Radarr Custom Script (On Download): censor profanities and append a "Clean" audio stream.
#
# This script is designed to be run by Radarr after a movie is imported.
# It runs the existing CLI pipeline in [`src/main.py:main()`](src/main.py:229),
# writes a temp output next to the original file, then atomically replaces the
# original file in-place (same directory => same filesystem => atomic `mv`).
#
# Notes:
# - Idempotent by default: inputs with an audio stream titled "Clean" are skipped
#   unless `--force` is used (see [`src/video_tools.py:input_has_clean_track_marker()`](src/video_tools.py:234)).
# - Radarr can pass the movie path as `{MovieFile.Path}` (recommended) OR rely on
#   Radarr's `radarr_moviefile_path` env var.

log() { printf '%s\n' "[profanity_remover][radarr] $*"; }
err() { printf '%s\n' "[profanity_remover][radarr][ERROR] $*" >&2; }

event_type="${radarr_eventtype:-}" # e.g. Download, Test, Rename, Health...

case "${event_type}" in
  ""|Download)
    ;; # proceed (some Radarr versions may not set radarr_eventtype)
  Test)
    log "Radarr test event: exiting 0"
    exit 0
    ;;
  *)
    log "Ignoring unsupported Radarr event type: '${event_type}'"
    exit 0
    ;;
esac

in_path="${1:-${radarr_moviefile_path:-}}"
if [[ -z "${in_path}" ]]; then
  err "Missing input path. Configure Radarr to pass {MovieFile.Path} as the first argument (recommended)."
  exit 1
fi

if [[ ! -f "${in_path}" ]]; then
  err "Input file not found: ${in_path}"
  exit 1
fi

# Resolve repo root based on this script's location.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Choose Python:
# - allow override via PROFANITY_REMOVER_PYTHON
# - otherwise prefer repo-local venv if present
# - else fall back to python3 on PATH
python_bin="${PROFANITY_REMOVER_PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
  if [[ -x "${repo_dir}/.venv/bin/python" ]]; then
    python_bin="${repo_dir}/.venv/bin/python"
  else
    python_bin="python3"
  fi
fi

# Temp output in same directory => mv is atomic on same filesystem.
in_dir="$(cd "$(dirname "${in_path}")" && pwd)"
in_file="$(basename "${in_path}")"
base="${in_file%.*}"
ext="${in_file##*.}"

tmp_out="$(mktemp "${in_dir}/.${base}.clean.XXXXXX.${ext}")"

cleanup() {
  if [[ -n "${tmp_out:-}" && -f "${tmp_out}" ]]; then
    rm -f "${tmp_out}" || true
  fi
}
trap cleanup EXIT

log "Input : ${in_path}"
log "Temp  : ${tmp_out}"
log "Repo  : ${repo_dir}"
log "Python: ${python_bin}"

(
  cd "${repo_dir}"
  # Do NOT pass --force here; idempotence is handled internally by
  # [`input_has_clean_track_marker()`](src/video_tools.py:234).
  "${python_bin}" -m src.main --input "${in_path}" --output "${tmp_out}"
)

if [[ ! -s "${tmp_out}" ]]; then
  err "Output missing/empty: ${tmp_out}"
  exit 1
fi

log "Replacing original in-place"
mv -f "${tmp_out}" "${in_path}"

tmp_out="" # prevent trap deletion of the moved file
log "Done"

