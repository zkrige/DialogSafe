# Changelog

## 0.0.1 - 2026-01-09

- Appends a `Clean` audio track instead of replacing the original.
- Skips processing when a clean-track marker exists unless `--force` / `FORCE=true`.
- MKV outputs preserve streams via `-map 0`.
