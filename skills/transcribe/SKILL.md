---
name: transcribe
description: Transcribe videos and audio to text locally, from YouTube or any yt-dlp-supported URL or from local media files, then read the transcripts to summarise or answer questions. Use when the user shares video, podcast, or audio links or files and wants a transcript, summary, notes, or key points.
license: MIT
---

# Transcribe

Run `scripts/transcribe.py` from this skill's directory (`${CLAUDE_SKILL_DIR}` in Claude Code) with the user's URLs and file paths:

```bash
"${CLAUDE_SKILL_DIR}/scripts/transcribe.py" -o transcripts <url-or-file> ...
```

Each input prints one result line: `OK|SKIP|FAIL <tab> <how> <tab> <.txt path or error>`. Read each `.txt` and do what the user asked; with no other instruction, give a short summary per video.
