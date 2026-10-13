# Contributing

Bug reports and fixes are welcome. For anything bigger than a small fix, open an issue first so we can agree the approach before you build it. [SPEC.md](SPEC.md) is the design contract (non-goals and decisions), so a change that crosses it starts there.

## Setup

You need Python 3.9+ and [uv](https://github.com/astral-sh/uv) for ruff. The tests put fakes for yt-dlp, ffmpeg, ffprobe, whisper-cli and curl on a private `PATH`, so they need neither the real tools nor a network. One integration test runs the real whisper.cpp when it's installed from Homebrew (it uses the bundled `jfk.wav` sample) and the default model is cached. Otherwise it's skipped.

```bash
python3 -m unittest discover -s tests -v
uvx ruff check .
claude plugin validate --strict .    # needs Claude Code
```

## Rules

- The script stays stdlib-only Python 3.9+. No pip dependencies and no Python bindings: shell out to the CLIs and parse their output.
- Transcription stays local. No cloud speech-to-text and no LLM or API calls from the script.
- The script never installs anything. It detects a missing tool and prints the install command.
- `SKILL.md` frontmatter uses Agent Skills spec fields only. The command it shows keeps the script path first and unquoted, because that's the form its `allowed-tools` rule matches.
- Keep shell strings ASCII.

## Pull requests

- One concern per PR, with tests. `tests/fakes.py` sets each fake tool's behaviour through `FAKE_*` environment variables.
- The tests and ruff pass.
- A behaviour change updates README.md and SPEC.md in the same PR.
- Say what changed and how you verified it.

## Bug reports

Include the command you ran, its stdout and stderr, your OS, and the output of `yt-dlp --version`.

## License

By contributing, you agree your contributions are licensed under the MIT License.
