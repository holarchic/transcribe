#!/usr/bin/env python3
"""Transcribe video/audio URLs or local files to .txt + .srt, locally.

Human subtitles when the site has them (yt-dlp), otherwise ffmpeg -> whisper.cpp.
Stdlib only; shells out to yt-dlp, ffmpeg, ffprobe, whisper-cli and (for the
one-time model download) curl.
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

DEFAULT_MODEL = "large-v3-turbo-q5_0"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"

INSTALL_HINTS = {
    "ffmpeg": {"darwin": "brew install ffmpeg", "other": "sudo apt install ffmpeg"},
    "ffprobe": {"darwin": "brew install ffmpeg", "other": "sudo apt install ffmpeg"},
    "whisper-cli": {
        "darwin": "brew install whisper-cpp",
        "other": "brew install whisper-cpp, or build https://github.com/ggml-org/whisper.cpp",
    },
    "yt-dlp": {
        "darwin": "brew install yt-dlp",
        "other": 'pipx install "yt-dlp[default]" (YouTube also needs deno)',
    },
    "curl": {"darwin": "curl ships with macOS", "other": "sudo apt install curl"},
}


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
    channel: str | None
    date: str | None
    duration_s: float | None
    source: str
    stem_id: str
    language: str | None
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


def install_hint(tool: str) -> str:
    platform = "darwin" if sys.platform == "darwin" else "other"
    return INSTALL_HINTS[tool][platform]


def run(cmd: list) -> subprocess.CompletedProcess:
    tool = cmd[0]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
    except OSError as e:
        raise InputError(f"{tool} could not run: {e}") from e
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        reason = lines[-1] if lines else f"exit {proc.returncode}"
        suffix = " (if the site changed, update yt-dlp)" if tool == "yt-dlp" else ""
        raise InputError(f"{tool} failed: {reason}{suffix}")
    return proc


def classify(raw: str) -> Source:
    if raw.startswith(("http://", "https://")):
        return Source(raw, "url")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw):
        raise InputError("unsupported scheme (only http and https URLs)")
    if Path(raw).expanduser().is_file():
        return Source(raw, "file")
    raise InputError("not found")


def preflight(sources: list, model_arg: str) -> Path:
    needed = ["ffmpeg", "ffprobe", "whisper-cli"]
    if any(s.kind == "url" for s in sources):
        needed.append("yt-dlp")
    missing = [tool for tool in needed if shutil.which(tool) is None]
    if missing:
        lines = [f"  {tool}: {install_hint(tool)}" for tool in missing]
        raise PreflightError("missing dependencies:\n" + "\n".join(lines))
    return resolve_model(model_arg)


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "transcribe" / "models"


def resolve_model(arg: str) -> Path:
    if "/" in arg or arg.endswith(".bin"):
        path = Path(arg).expanduser()
        if not path.is_file():
            raise PreflightError(f"model file not found: {path}")
        return path
    path = cache_dir() / f"ggml-{arg}.bin"
    if not path.is_file():
        download_model(arg, path)
    return path


def download_model(name: str, dest: Path) -> None:
    if shutil.which("curl") is None:
        hint = install_hint("curl")
        raise PreflightError(f"model {name} is not cached and curl is missing: {hint}")
    url = MODEL_URL.format(name=name)
    part = dest.with_name(dest.name + ".part")
    print(f"downloading model {name} (one-time) to {dest}", file=sys.stderr)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["curl", "-fL", "--retry", "3", "-#", "-o", str(part), url]
        code = subprocess.run(cmd, check=False).returncode
        if code != 0:
            raise PreflightError(f"model download failed (curl exit {code}): {url}")
        os.replace(part, dest)
    except OSError as e:
        raise PreflightError(f"model cache error: {e}") from e
    finally:
        remove_quietly(part)


def remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def probe_url(url: str) -> Meta:
    proc = run(["yt-dlp", "--ignore-config", "-J", "--flat-playlist", "--no-playlist",
                "--no-warnings", url])
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise InputError("yt-dlp returned unreadable metadata") from e
    if info.get("_type") == "playlist":
        raise InputError("playlist URLs are not supported; pass video URLs")
    langs = tuple(sorted(k for k in (info.get("subtitles") or {}) if k != "live_chat"))
    upload = info.get("upload_date")
    return Meta(
        title=info.get("title") or info.get("id") or "untitled",
        channel=info.get("channel") or info.get("uploader"),
        date=f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}" if upload and len(upload) == 8 else None,
        duration_s=_as_float(info.get("duration")),
        source=info.get("webpage_url") or url,
        stem_id=str(info.get("id") or hashlib.sha1(url.encode()).hexdigest()[:8]),
        language=info.get("language"),
        subtitle_langs=langs,
    )


def probe_file(path: Path) -> Meta:
    proc = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json",
                str(path)])
    try:
        duration = json.loads(proc.stdout).get("format", {}).get("duration")
    except (json.JSONDecodeError, AttributeError):
        duration = None
    return Meta(
        title=path.stem,
        channel=None,
        date=None,
        duration_s=_as_float(duration),
        source=str(path),
        stem_id=hashlib.sha1(str(path).encode()).hexdigest()[:8],
        language=None,
        subtitle_langs=(),
    )


def _as_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def stem_for(meta: Meta) -> str:
    ascii_title = unicodedata.normalize("NFKD", meta.title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")[:60].rstrip("-")
    return f"{slug}-{meta.stem_id}" if slug else meta.stem_id


def pick_subtitle_lang(meta: Meta, lang_arg: str) -> str | None:
    target = meta.language if lang_arg == "auto" else lang_arg
    if target is None:
        return meta.subtitle_langs[0] if len(meta.subtitle_langs) == 1 else None
    if target in meta.subtitle_langs:
        return target
    base = target.split("-")[0]
    matches = [k for k in meta.subtitle_langs if k.split("-")[0] == base]
    return matches[0] if matches else None


def only_artifact(tmp: Path, pattern: str, producer: str) -> Path:
    found = sorted(tmp.glob(pattern))
    if len(found) != 1:
        raise InputError(f"{producer} produced {len(found)} files matching {pattern}, expected 1")
    return found[0]


def fetch_subtitles(url: str, lang: str, tmp: Path) -> str:
    run(["yt-dlp", "--ignore-config", "--skip-download", "--write-subs", "--sub-langs", lang,
         "--convert-subs", "srt", "--no-playlist", "--no-warnings", "-o", str(tmp / "sub"), url])
    return only_artifact(tmp, "sub*.srt", "yt-dlp").read_text(encoding="utf-8", errors="replace")


def fetch_audio(url: str, tmp: Path) -> Path:
    # Named source.*, never audio.*: a WAV download would collide with the conversion target.
    run(["yt-dlp", "--ignore-config", "-f", "bestaudio/best", "--no-playlist", "--no-warnings",
         "-o", str(tmp / "source.%(ext)s"), url])
    return only_artifact(tmp, "source.*", "yt-dlp")


def to_wav(src: Path, tmp: Path) -> Path:
    wav = tmp / "audio.wav"
    run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
    if not wav.is_file() or wav.stat().st_size == 0:
        raise InputError("ffmpeg produced no audio")
    return wav


def transcribe_wav(wav: Path, model: Path, lang: str, tmp: Path) -> str:
    run(["whisper-cli", "-m", str(model), "-f", str(wav), "-l", lang, "-osrt",
         "-of", str(tmp / "w"), "-np"])
    # whisper-cli can exit 0 without writing anything (e.g. unreadable audio).
    srt = tmp / "w.srt"
    if not srt.is_file():
        raise InputError("whisper-cli produced no transcript")
    return srt.read_text(encoding="utf-8", errors="replace")


_TIMING = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)")


def parse_srt(text: str) -> list:
    cues = []
    normalized = text.lstrip("﻿").replace("\r\n", "\n").strip()
    if not normalized:
        return cues
    for block in re.split(r"\n\s*\n", normalized):
        lines = block.splitlines()
        timing_at = next((i for i, line in enumerate(lines) if _TIMING.match(line)), None)
        if timing_at is None:
            raise InputError("malformed subtitles: block without a timing line")
        start, end = _TIMING.match(lines[timing_at]).groups()
        body = re.sub(r"<[^>]+>", "", " ".join(lines[timing_at + 1:]))
        body = " ".join(body.split())
        if body:
            cues.append(Cue(start, end, body))
    return cues


def cue_lines(cues: list) -> list:
    lines = []
    for cue in cues:
        if not lines or lines[-1] != cue.text:
            lines.append(cue.text)
    return lines


def format_duration(seconds: float) -> str:
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def render_txt(meta: Meta, how: str, lines: list) -> str:
    header = [f"title: {meta.title}"]
    if meta.channel:
        header.append(f"channel: {meta.channel}")
    if meta.date:
        header.append(f"date: {meta.date}")
    if meta.duration_s is not None:
        header.append(f"duration: {format_duration(meta.duration_s)}")
    header += [f"source: {meta.source}", f"transcript: {how}", "---"]
    return "\n".join(header + lines) + "\n"


def publish(out: Path, stem: str, srt: str, txt: str) -> None:
    """Replace <stem>.srt/.txt so that a present .txt always means a complete, matching pair.

    Both files are staged first; the old .txt (the completion marker) is removed before
    the .srt is replaced, and the new .txt lands last.
    """
    srt_tmp, txt_tmp = out / f".{stem}.srt.tmp", out / f".{stem}.txt.tmp"
    try:
        srt_tmp.write_text(srt, encoding="utf-8")
        txt_tmp.write_text(txt, encoding="utf-8")
        (out / f"{stem}.txt").unlink(missing_ok=True)
        os.replace(srt_tmp, out / f"{stem}.srt")
        os.replace(txt_tmp, out / f"{stem}.txt")
    finally:
        remove_quietly(srt_tmp)
        remove_quietly(txt_tmp)


def process(source: Source, args: argparse.Namespace, model: Path, tag: str) -> Result:
    try:
        return _process(source, args, model, tag)
    except OSError as e:
        raise InputError(f"filesystem error: {e}") from e


def _process(source: Source, args: argparse.Namespace, model: Path, tag: str) -> Result:
    is_url = source.kind == "url"
    meta = probe_url(source.raw) if is_url else probe_file(Path(source.raw).expanduser().resolve())
    stem = stem_for(meta)
    txt_path = args.out / f"{stem}.txt"
    if txt_path.exists() and not args.force:
        return Result(source.raw, "SKIP", "-", str(txt_path))
    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp_name:
        tmp = Path(tmp_name)
        lang = pick_subtitle_lang(meta, args.lang) if is_url else None
        if lang:
            print(f"{tag} subtitles:{lang} {stem}", file=sys.stderr)
            srt, how = fetch_subtitles(source.raw, lang, tmp), f"subtitles:{lang}"
        else:
            if is_url:
                print(f"{tag} download {stem}", file=sys.stderr)
            media = fetch_audio(source.raw, tmp) if is_url else Path(meta.source)
            print(f"{tag} whisper {stem}", file=sys.stderr)
            srt = transcribe_wav(to_wav(media, tmp), model, args.lang, tmp)
            how = f"whisper:{model.stem.replace('ggml-', '', 1)}"
    lines = cue_lines(parse_srt(srt))
    if not lines:
        raise InputError("empty transcript (no speech detected)")
    publish(args.out, stem, srt, render_txt(meta, how, lines))
    return Result(source.raw, "OK", how, str(txt_path))


def classify_or_fail(raw: str) -> Source | Result:
    try:
        return classify(raw)
    except InputError as e:
        return Result(raw, "FAIL", "-", str(e))


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", metavar="INPUT", help="URL or local media file")
    parser.add_argument("-o", "--out", type=Path, default=Path("transcripts"),
                        help="output directory (default: ./transcripts)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"whisper.cpp model name or path to a .bin (default: {DEFAULT_MODEL})")
    parser.add_argument("--lang", default="auto", help="spoken language code, or auto (default)")
    parser.add_argument("--force", action="store_true", help="re-transcribe even if outputs exist")
    return parser.parse_args(argv)


def main(argv: list) -> int:
    args = parse_args(argv)
    items = [classify_or_fail(raw) for raw in args.inputs]
    try:
        args.out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"transcribe: cannot create output directory: {e}", file=sys.stderr)
        return 2
    try:
        model = preflight([i for i in items if isinstance(i, Source)], args.model)
    except PreflightError as e:
        print(f"transcribe: {e}", file=sys.stderr)
        return 2
    results = []
    for n, item in enumerate(items, 1):
        if isinstance(item, Result):
            results.append(item)
            continue
        try:
            results.append(process(item, args, model, f"[{n}/{len(items)}]"))
        except InputError as e:
            results.append(Result(item.raw, "FAIL", "-", str(e)))
    for r in results:
        print(f"{r.status}\t{r.how}\t{r.detail}")
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
