# transcribe

[![CI](https://github.com/holarchic/transcribe/actions/workflows/ci.yml/badge.svg)](https://github.com/holarchic/transcribe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/holarchic/transcribe)](LICENSE)

Local video transcription skill for Claude Code and Agent Skills: human subtitles via yt-dlp, else whisper.cpp.

> **Status:** v0.1.0, actively maintained and in use since September 2026. 39 tests, including a real whisper.cpp integration test (local only; CI uses fakes).

```console
$ skills/transcribe/scripts/transcribe.py --lang en 'https://www.youtube.com/watch?v=jNQXAC9IVRw'
[1/1] subtitles:en me-at-the-zoo-14fc8477
OK	subtitles:en	transcripts/me-at-the-zoo-14fc8477.txt
$ cat transcripts/me-at-the-zoo-14fc8477.txt
title: Me at the zoo
channel: jawed
date: 2005-04-24
duration: 0m19s
source: https://www.youtube.com/watch?v=jNQXAC9IVRw
transcript: subtitles:en
---
All right, so here we are, in front of the elephants
the cool thing about these guys is that they have really...
really really long trunks
and that's cool
(baaaaaaaaaaahhh!!)
and that's pretty much all there is to say
```

## Why

For anyone with more video and podcast audio to get through than time to watch it. Hand your agent a batch of links or files and it makes the transcripts on your own machine, then summarises them or answers your questions. It needs no API keys and no Python packages.

Each input gets the site's human-made subtitles when they exist and a local [whisper.cpp](https://github.com/ggml-org/whisper.cpp) transcription otherwise. Inputs are YouTube links (or anything [yt-dlp](https://github.com/yt-dlp/yt-dlp) supports) and local media files. You get a `.txt` and an `.srt` per input, and the agent (Claude Code, or any agent that reads [Agent Skills](https://agentskills.io)) works from the `.txt`.

```
> summarise these: https://www.youtube.com/watch?v=… https://www.youtube.com/watch?v=… ~/Downloads/call.m4a
```

**When not to use it:**

- You only need YouTube's captions. YouTube transcript MCP servers fetch them without downloading media or running a model, so they're quicker, but most videos only have auto-generated captions (no punctuation, repeated lines).
- You have hours of audio on a CPU-only machine. whisper.cpp is much slower without a GPU, and a cloud speech-to-text API is the faster option there (at a per-minute price).
- You need speaker labels (who said what). They aren't included here, but [WhisperX](https://github.com/m-bain/whisperX) adds diarisation on top of Whisper.

Also not included: translation, playlists, cloud transcription, and summarising inside the script (the agent does that).

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org) (includes ffprobe)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp): the `whisper-cli` binary
- [yt-dlp](https://github.com/yt-dlp/yt-dlp), for URLs. YouTube also needs a JavaScript runtime ([deno](https://deno.com)); see yt-dlp's [EJS guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
- curl, for the one-time model download (macOS ships with it)

macOS:

```bash
brew install ffmpeg whisper-cpp yt-dlp    # Homebrew's yt-dlp pulls in deno
```

Linux: install ffmpeg from your package manager, `pipx install "yt-dlp[default]"` plus deno, and whisper.cpp from Homebrew or [built from source](https://github.com/ggml-org/whisper.cpp#quick-start) with `whisper-cli` on your `PATH`.

The first run downloads the default model, `large-v3-turbo-q5_0` (574 MB), to `~/.cache/transcribe/models/` (or `$XDG_CACHE_HOME/transcribe/models/` when that's set).

Tested on macOS with Apple Silicon. CI runs the test suite on Linux and macOS, with fakes in place of the external tools. Windows isn't supported.

## Install the skill

**Claude Code plugin**

```
/plugin marketplace add holarchic/transcribe
/plugin install transcribe@transcribe
```

**Manual** (Claude Code or any Agent Skills–compatible agent): copy `skills/transcribe/` into your agent's skills directory, for example `~/.claude/skills/transcribe/`. The skill runs `scripts/transcribe.py` relative to its own directory.

## Use

Ask the agent to transcribe or summarise videos, or call `/transcribe` (`/transcribe:transcribe` when installed as a plugin). When you type the slash command, Claude Code runs the script without a permission prompt. When Claude picks the skill on its own, it asks once before running the script.

You can also run the script directly:

```
skills/transcribe/scripts/transcribe.py [-o DIR] [--model NAME|PATH] [--lang CODE] [--force] INPUT...
```

| Argument | Meaning | Default |
|---|---|---|
| `INPUT` | http(s) URL or local media file | |
| `-o`, `--out DIR` | output directory | `./transcripts` |
| `--model NAME\|PATH` | whisper.cpp model name, or path to a ggml `.bin` | `large-v3-turbo-q5_0` |
| `--lang CODE` | spoken language code, or `auto` | `auto` |
| `--force` | redo inputs that already have a transcript | off |

Quote URLs: `?` and `&` are special to the shell, and zsh stops with "no matches found" on an unquoted YouTube link.

It prints one line per input, `OK|SKIP|FAIL <tab> how <tab> path-or-error`, and sets the exit code:

| Exit | Meaning |
|---|---|
| 0 | nothing failed |
| 1 | an input failed |
| 2 | it couldn't start (a missing dependency, a model that failed to download, an output directory it can't create) |

Each `.txt` starts with a header (title, channel, upload date, duration, source, and how it was transcribed), then `---`, then the text, as in the example at the top. Channel and date only appear for URLs.

## Behaviour worth knowing

- **Speed.** On an Apple Silicon Mac a 23-minute video takes about two and a half minutes end to end, including the download (measured on an M5). CPU-only machines are much slower.
- **Re-runs resume.** Inputs that already have a transcript are skipped, so an interrupted batch picks up where it stopped. A transcript is written only once it's complete.
- **Human subtitles only.** YouTube's auto-generated captions are ignored (no punctuation, repeated lines), so those videos go through whisper.cpp. Subtitles are used when their language matches the video's (or `--lang`). If the site doesn't report the video's language, a lone subtitle track is used, but with several it falls back to whisper.cpp unless you pass `--lang` (the example at the top needs `--lang en` for this reason).
- **One video per URL.** Playlist URLs are rejected; a watch URL with a `list=` parameter takes just that video.
- **Your yt-dlp config file is ignored** (`--ignore-config`), so outputs are predictable. Cookies and proxies set there don't apply.
- **One run at a time** per output directory. Two runs writing the same directory (or downloading the model) at once aren't supported.
- **Update yt-dlp first** if YouTube downloads start failing.

## Contributing

Bug reports and fixes are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers setup and pull requests. [ARCHITECTURE.md](ARCHITECTURE.md) maps the code: the pipeline, its error types and the test fakes.

## Attribution

For those who care about such things, the majority of the code in this repo was produced by Claude Opus 5.5 (Anthropic), with david guiding Claude, providing insights, direction and critical review throughout the process.

## License

[MIT](LICENSE)
