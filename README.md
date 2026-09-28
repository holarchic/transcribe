# transcribe

An agent skill that turns a batch of videos into transcripts on your own machine, then lets the agent read and summarise them.

Give it YouTube links (or anything [yt-dlp](https://github.com/yt-dlp/yt-dlp) supports) and local media files. For each one it uses the site's human-made subtitles when they exist, and otherwise transcribes the audio locally with [whisper.cpp](https://github.com/ggml-org/whisper.cpp). You get a `.txt` and `.srt` per input. The agent (Claude Code, or any agent that reads [Agent Skills](https://agentskills.io)) reads the `.txt` and does what you asked: summarise, pull out claims, compare videos.

```
> summarise these: https://www.youtube.com/watch?v=… https://www.youtube.com/watch?v=… ~/Downloads/call.m4a
```

No API keys, no cloud transcription, and no Python packages to install.

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org) (includes ffprobe)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp): the `whisper-cli` binary
- [yt-dlp](https://github.com/yt-dlp/yt-dlp), for URLs. YouTube also needs a JavaScript runtime ([deno](https://deno.com)); see yt-dlp's [EJS guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).

macOS:

```bash
brew install ffmpeg whisper-cpp yt-dlp    # Homebrew's yt-dlp pulls in deno
```

Linux: install ffmpeg from your package manager, `pipx install "yt-dlp[default]"` plus deno, and whisper.cpp from Homebrew or [built from source](https://github.com/ggml-org/whisper.cpp#quick-start) with `whisper-cli` on your `PATH`.

The first run downloads the default model, `large-v3-turbo-q5_0` (574 MB), to `~/.cache/transcribe/models/`.

## Install the skill

**Claude Code plugin**

```
/plugin marketplace add holarchic/transcribe
/plugin install transcribe@transcribe
```

**Manual** (Claude Code or any Agent Skills–compatible agent): copy `skills/transcribe/` into your agent's skills directory, e.g. `~/.claude/skills/transcribe/`. The skill runs `scripts/transcribe.py` relative to its own directory.

## Use

Ask the agent to transcribe or summarise videos, or call `/transcribe` (`/transcribe:transcribe` when installed as a plugin). When you type the slash command, Claude Code runs the script without a permission prompt; when Claude picks the skill on its own, it asks once before running the script.

You can also run the script directly:

```
skills/transcribe/scripts/transcribe.py [-o DIR] [--model NAME|PATH] [--lang CODE] [--force] INPUT...

  INPUT      http(s) URL or local media file
  -o         output directory (default ./transcripts)
  --model    whisper.cpp model name or path to a ggml .bin (default large-v3-turbo-q5_0)
  --lang     spoken language code, or auto (default)
  --force    redo inputs that already have a transcript
```

It prints one line per input, `OK|SKIP|FAIL <tab> how <tab> path-or-error`, and exits 0 when nothing failed, 1 when an input failed, and 2 when it couldn't start (a missing dependency, a model that failed to download, an output directory it can't create).

Each `.txt` starts with a header, then the text:

```
title: Me at the zoo
channel: jawed
date: 2005-04-24
duration: 0m19s
source: https://www.youtube.com/watch?v=jNQXAC9IVRw
transcript: subtitles:en
---
All right, so here we are, in front of the elephants
```

## Behaviour worth knowing

- **Speed.** On an Apple Silicon Mac a 23-minute video takes about two and a half minutes end to end, including the download. CPU-only machines are much slower.
- **Re-runs are free.** Inputs that already have a transcript are skipped, so an interrupted batch resumes where it stopped. A transcript is written only once it's complete.
- **Human subtitles only.** YouTube's auto-generated captions are ignored (no punctuation, repeated lines); those videos go through whisper.cpp. Subtitles are used when their language matches the video's (or `--lang`).
- **One video per URL.** Playlist URLs are rejected; a watch URL with a `list=` parameter takes just that video.
- **Your yt-dlp config file is ignored** (`--ignore-config`), so outputs are predictable. Cookies and proxies set there don't apply.
- **One run at a time** per output directory. Two runs writing the same directory (or downloading the model) at once aren't supported.
- If YouTube downloads start failing, update yt-dlp first.

## Not included

Speaker labels, translation, playlists, cloud transcription, and summarising inside the script (the agent does that).

## Development

```bash
python3 -m unittest discover -s tests -v    # fakes for every external tool; one real whisper.cpp test when available
uvx ruff check .
claude plugin validate .
```

`SPEC.md` is the design contract: decisions, acceptance criteria, and failure modes.

## License

MIT
