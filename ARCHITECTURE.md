# transcribe Architecture

## skills/transcribe/SKILL.md

Agent-facing instructions; Agent Skills spec frontmatter only. The command it shows must stay unquoted with the script path first, because that is the form its `allowed-tools` rule matches.

## skills/transcribe/scripts/transcribe.py

The whole pipeline, in one file.

- **Flow:** `main` → `classify_or_fail` → `preflight` (tools, then `resolve_model`/`download_model`) → per input `process(source, Options, index, total)` → `probe_url`/`probe_file` → SKIP check → `fetch_subtitles` | `fetch_audio` → `to_wav` → `transcribe_wav` → `parse_srt` → `render_txt` → `publish` (`.txt` is written last, as the completion marker).
- **Errors:** `InputError` fails one input, `PreflightError` exits 2. Expected external failures are translated at each tool boundary (`run`, artifact checks, `process` for `OSError`); everything else propagates.
- **Identity:** `stem_for` is the title slug plus sha1(canonical URL or absolute path)[:8], never the site's id.
- **Parsing:** `parse_srt` is positional. Only a fully valid timing line opens a cue; a number is an index only at a block start with a valid timing line next; only known formatting tags (`_FORMAT_TAG`) are stripped.
- **Boundaries:** `YT_DLP` holds the shared yt-dlp isolation flags; `redact` masks URL credentials and secret-looking query params in every output; `format_result`/`format_provenance`/`render_txt` are the only stringifiers; `log`/`progress` are the only stderr writers.

## tests/

- `fakes.py` writes fake `yt-dlp`/`ffmpeg`/`ffprobe`/`whisper-cli`/`curl` into a hermetic PATH. `FAKE_*` env vars set their behaviour, and invocations are logged to `$FAKE_LOG`.
- `test_transcribe.py` holds units, a crash-injected `publish`, CLI runs against the fakes, and one real whisper.cpp integration test (skipped when unavailable).

## .claude-plugin/

`plugin.json` and `marketplace.json`. The repo is its own marketplace, with source `./`.

## pyproject.toml

Metadata and ruff config only; no dependencies.
