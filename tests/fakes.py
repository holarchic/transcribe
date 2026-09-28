"""Fake yt-dlp / ffmpeg / ffprobe / whisper-cli / curl for driving transcribe.py end to end.

Each fake appends {"tool", "argv"} as one JSON line to $FAKE_LOG and takes its behaviour
from environment variables, so a test controls the whole pipeline without network or GPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

GOOD_SRT = """1
00:00:00,000 --> 00:00:02,000
Hello from the fake.

2
00:00:02,000 --> 00:00:04,000
Second line.
"""

_PRELUDE = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
log = os.environ.get("FAKE_LOG")
prior = []
if log and Path(log).exists():
    prior = [json.loads(line) for line in Path(log).read_text().splitlines() if line]
if log:
    with open(log, "a") as f:
        f.write(json.dumps({"tool": TOOL, "argv": argv}) + "\n")
def opt(flag):
    return argv[argv.index(flag) + 1] if flag in argv else None
GOOD_SRT = GOOD_SRT_VALUE
'''

_BODIES = {
    "yt-dlp": r'''
if os.environ.get("FAKE_YTDLP_FAIL"):
    sys.stderr.write("ERROR: HTTP Error 403: Forbidden\n"); sys.exit(1)
if "-J" in argv:
    meta = os.environ.get("FAKE_META")
    sys.stdout.write(Path(meta).read_text() if meta else json.dumps(
        {"_type": "video", "id": "vid123", "title": "Fake Video", "channel": "Fake Channel",
         "upload_date": "20260102", "duration": 125, "language": "en",
         "webpage_url": argv[-1], "subtitles": {}}))
elif "--write-subs" in argv:
    Path(opt("-o") + "." + opt("--sub-langs") + ".srt").write_text(
        os.environ.get("FAKE_SUBS_SRT", GOOD_SRT))
else:
    target = opt("-o").replace("%(ext)s", os.environ.get("FAKE_AUDIO_EXT", "webm"))
    Path(target).write_bytes(b"fake-media")
''',
    "ffmpeg": r'''
src, dst = opt("-i"), argv[-1]
if os.path.abspath(src) == os.path.abspath(dst):
    sys.stderr.write("Output same as Input #0 - exiting\n"); sys.exit(1)
Path(dst).write_bytes(b"RIFF-fake-wav")
''',
    "ffprobe": r'''
sys.stdout.write(json.dumps({"format": {"duration": "11.0"}}))
''',
    "whisper-cli": r'''
modes = os.environ.get("FAKE_WHISPER_MODES", "ok").split(",")
calls = sum(1 for p in prior if p["tool"] == "whisper-cli")
mode = modes[min(calls, len(modes) - 1)]
out = Path(opt("-of") + ".srt")
if mode == "fail":
    sys.stderr.write("error: failed to read audio file\n"); sys.exit(1)
elif mode == "exit0-no-output":
    sys.exit(0)
elif mode == "empty":
    out.write_text("")
elif mode == "malformed":
    out.write_text("this is not\nan srt file\n")
else:
    out.write_text(GOOD_SRT)
''',
    "curl": r'''
if os.environ.get("FAKE_CURL_FAIL"):
    sys.stderr.write("curl: (22) The requested URL returned error: 404\n"); sys.exit(22)
Path(opt("-o")).write_bytes(b"fake-model")
''',
}

TOOLS = tuple(_BODIES)


def make_bin(bin_dir: Path, tools: tuple = TOOLS) -> Path:
    """Write the requested fakes plus a python3 link to this interpreter into bin_dir."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(sys.executable, bin_dir / "python3")
    for tool in tools:
        path = bin_dir / tool
        prelude = _PRELUDE.replace("TOOL", repr(tool)).replace("GOOD_SRT_VALUE", repr(GOOD_SRT))
        path.write_text(prelude + _BODIES[tool])
        path.chmod(0o755)
    return bin_dir
