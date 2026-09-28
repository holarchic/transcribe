#!/usr/bin/env python3
"""Transcribe video/audio URLs or local files to .txt + .srt, locally.

Human subtitles when the site has them (yt-dlp), otherwise ffmpeg -> whisper.cpp.
Stdlib only; shells out to yt-dlp, ffmpeg, ffprobe and whisper-cli.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "large-v3-turbo-q5_0"


class InputError(Exception):
    """A failure confined to one input: reported as FAIL, the batch continues."""


class PreflightError(Exception):
    """A failure that stops the run before any input is processed (exit 2)."""


@dataclass(frozen=True)
class Source:
    raw: str
    kind: str  # "url" | "file"


@dataclass(frozen=True)
class Meta:
    title: str
    channel: Optional[str]
    date: Optional[str]
    duration_s: Optional[float]
    source: str
    stem_id: str
    language: Optional[str]
    subtitle_langs: tuple


@dataclass(frozen=True)
class Cue:
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class Result:
    input: str
    status: str  # "OK" | "SKIP" | "FAIL"
    how: str
    detail: str


def run(cmd: list) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        raise InputError(f"{cmd[0]} failed: {lines[-1] if lines else f'exit {proc.returncode}'}")
    return proc


def classify(raw: str) -> Source:
    if raw.startswith(("http://", "https://")):
        return Source(raw, "url")
    if Path(raw).expanduser().is_file():
        return Source(raw, "file")
    raise InputError("not found")


def preflight(sources: list, model_arg: str) -> Path:
    needed = ["ffmpeg", "ffprobe", "whisper-cli"]
    if any(s.kind == "url" for s in sources):
        needed.append("yt-dlp")
    missing = [tool for tool in needed if shutil.which(tool) is None]
    if missing:
        raise PreflightError(f"missing: {', '.join(missing)}")
    return resolve_model(model_arg)


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "transcribe" / "models"


def resolve_model(arg: str) -> Path:
    if "/" in arg or arg.endswith(".bin"):
        path = Path(arg).expanduser()
    else:
        path = cache_dir() / f"ggml-{arg}.bin"
    if not path.is_file():
        raise PreflightError(f"model not found: {path}")
    return path


def probe_url(url: str) -> Meta:
    proc = run(["yt-dlp", "--ignore-config", "-J", "--flat-playlist", "--no-playlist",
                "--no-warnings", url])
    info = json.loads(proc.stdout)
    langs = tuple(sorted(k for k in (info.get("subtitles") or {}) if k != "live_chat"))
    upload = info.get("upload_date")
    return Meta(
        title=info.get("title") or info.get("id") or "untitled",
        channel=info.get("channel") or info.get("uploader"),
        date=f"{upload[:4]}-{upload[4:6]}-{upload[6:]}" if upload else None,
        duration_s=info.get("duration"),
        source=info.get("webpage_url") or url,
        stem_id=str(info.get("id") or hashlib.sha1(url.encode()).hexdigest()[:8]),
        language=info.get("language"),
        subtitle_langs=langs,
    )


def probe_file(path: Path) -> Meta:
    proc = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json",
                str(path)])
    duration = json.loads(proc.stdout).get("format", {}).get("duration")
    return Meta(
        title=path.stem,
        channel=None,
        date=None,
        duration_s=float(duration) if duration else None,
        source=str(path),
        stem_id=hashlib.sha1(str(path).encode()).hexdigest()[:8],
        language=None,
        subtitle_langs=(),
    )


def stem_for(meta: Meta) -> str:
    ascii_title = unicodedata.normalize("NFKD", meta.title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")[:60].rstrip("-")
    return f"{slug}-{meta.stem_id}" if slug else meta.stem_id


def pick_subtitle_lang(meta: Meta, lang_arg: str) -> Optional[str]:
    target = meta.language if lang_arg == "auto" else lang_arg
    if target is None:
        return meta.subtitle_langs[0] if len(meta.subtitle_langs) == 1 else None
    if target in meta.subtitle_langs:
        return target
    base = target.split("-")[0]
    matches = [k for k in meta.subtitle_langs if k.split("-")[0] == base]
    return matches[0] if matches else None


def fetch_subtitles(url: str, lang: str, tmp: Path) -> str:
    run(["yt-dlp", "--ignore-config", "--skip-download", "--write-subs", "--sub-langs", lang,
         "--convert-subs", "srt", "--no-playlist", "--no-warnings", "-o", str(tmp / "sub"), url])
    found = sorted(tmp.glob("sub*.srt"))
    return found[0].read_text(encoding="utf-8", errors="replace")


def fetch_audio(url: str, tmp: Path) -> Path:
    run(["yt-dlp", "--ignore-config", "-f", "bestaudio/best", "--no-playlist", "--no-warnings",
         "-o", str(tmp / "source.%(ext)s"), url])
    return sorted(tmp.glob("source.*"))[0]


def to_wav(src: Path, tmp: Path) -> Path:
    wav = tmp / "audio.wav"
    run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
    return wav


def transcribe_wav(wav: Path, model: Path, lang: str, tmp: Path) -> str:
    run(["whisper-cli", "-m", str(model), "-f", str(wav), "-l", lang, "-osrt",
         "-of", str(tmp / "w"), "-np"])
    return (tmp / "w.srt").read_text(encoding="utf-8", errors="replace")


_TIMING = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)")


def parse_srt(text: str) -> list:
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.splitlines()
        timing_at = next((i for i, line in enumerate(lines) if _TIMING.match(line)), None)
        if timing_at is None:
            continue
        start, end = _TIMING.match(lines[timing_at]).groups()
        body = " ".join(lines[timing_at + 1:])
        body = re.sub(r"<[^>]+>", "", body)
        body = " ".join(body.split())
        if body:
            cues.append(Cue(start, end, body))
    return cues


def render_txt(meta: Meta, how: str, cues: list) -> str:
    header = [f"title: {meta.title}", f"source: {meta.source}", f"transcript: {how}", "---"]
    return "\n".join(header + [c.text for c in cues]) + "\n"


def process(source: Source, args: argparse.Namespace, model: Path) -> Result:
    if source.kind == "url":
        meta = probe_url(source.raw)
    else:
        meta = probe_file(Path(source.raw).expanduser().resolve())
    stem = stem_for(meta)
    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp_name:
        tmp = Path(tmp_name)
        lang = pick_subtitle_lang(meta, args.lang) if source.kind == "url" else None
        if lang:
            srt, how = fetch_subtitles(source.raw, lang, tmp), f"subtitles:{lang}"
        else:
            media = fetch_audio(source.raw, tmp) if source.kind == "url" else Path(meta.source)
            srt = transcribe_wav(to_wav(media, tmp), model, args.lang, tmp)
            how = f"whisper:{model.stem.replace('ggml-', '')}"
    txt_path = args.out / f"{stem}.txt"
    (args.out / f"{stem}.srt").write_text(srt, encoding="utf-8")
    txt_path.write_text(render_txt(meta, how, parse_srt(srt)), encoding="utf-8")
    return Result(source.raw, "OK", how, str(txt_path))


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", metavar="INPUT", help="URL or local media file")
    parser.add_argument("-o", "--out", type=Path, default=Path("transcripts"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name or path to a .bin")
    parser.add_argument("--lang", default="auto", help="spoken language, or auto")
    return parser.parse_args(argv)


def main(argv: list) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [classify(raw) for raw in args.inputs]
    try:
        model = preflight(sources, args.model)
    except PreflightError as e:
        print(f"transcribe: {e}", file=sys.stderr)
        return 2
    results = []
    for n, source in enumerate(sources, 1):
        print(f"[{n}/{len(sources)}] {source.raw}", file=sys.stderr)
        try:
            results.append(process(source, args, model))
        except InputError as e:
            results.append(Result(source.raw, "FAIL", "-", str(e)))
    for r in results:
        print(f"{r.status}\t{r.how}\t{r.detail}")
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
