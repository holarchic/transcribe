from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import fakes

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "skills" / "transcribe" / "scripts" / "transcribe.py"
_spec = importlib.util.spec_from_file_location("transcribe", SCRIPT)
tx = importlib.util.module_from_spec(_spec)
sys.modules["transcribe"] = tx  # dataclasses resolve string annotations via sys.modules
_spec.loader.exec_module(tx)

URL = "https://www.youtube.com/watch?v=vid123"


def meta(**overrides):
    base = {"title": "Fake Video", "channel": "Fake Channel", "date": "2026-01-02",
            "duration_s": 125.0, "source": URL, "stem_id": "vid123", "language": "en",
            "subtitle_langs": ()}
    base.update(overrides)
    return tx.Meta(**base)


class Units(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(tx.classify("https://x.test/v").kind, "url")
        self.assertEqual(tx.classify("http://x.test/v").kind, "url")
        with self.assertRaisesRegex(tx.InputError, "unsupported scheme"):
            tx.classify("ftp://x.test/v")
        with self.assertRaisesRegex(tx.InputError, "not found"):
            tx.classify("youtu.be/abc")
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            self.assertEqual(tx.classify(f.name).kind, "file")

    def test_stem_for(self):
        self.assertEqual(tx.stem_for(meta(title="Héllo, Wörld!")), "hello-world-vid123")
        self.assertEqual(tx.stem_for(meta(title="日本語")), "vid123")
        long = tx.stem_for(meta(title="a " * 100))
        self.assertLessEqual(len(long), 60 + len("-vid123"))
        self.assertFalse(long.startswith("-") or "--" in long)

    def test_pick_subtitle_lang(self):
        pick = tx.pick_subtitle_lang
        self.assertEqual(pick(meta(subtitle_langs=("de", "en")), "auto"), "en")
        self.assertEqual(pick(meta(language="en-US", subtitle_langs=("en-GB",)), "auto"), "en-GB")
        self.assertEqual(pick(meta(language="en", subtitle_langs=("de", "en")), "de"), "de")
        self.assertIsNone(pick(meta(language="fr", subtitle_langs=("de", "en")), "auto"))
        self.assertEqual(pick(meta(language=None, subtitle_langs=("de",)), "auto"), "de")
        self.assertIsNone(pick(meta(language=None, subtitle_langs=("de", "en")), "auto"))
        self.assertIsNone(pick(meta(subtitle_langs=()), "auto"))

    def test_parse_srt(self):
        text = ("﻿1\r\n00:00:01.200 --> 00:00:03.360\r\n<i>All right,</i> so here\r\n"
                "we are\r\n\r\n00:00:04,000 --> 00:00:05,000\r\nno index\r\n")
        cues = tx.parse_srt(text)
        self.assertEqual([c.text for c in cues], ["All right, so here we are", "no index"])
        self.assertEqual(cues[0].start, "00:00:01.200")
        self.assertEqual(tx.parse_srt("  \n"), [])
        with self.assertRaisesRegex(tx.InputError, "malformed"):
            tx.parse_srt("this is not\nan srt file\n")

    def test_cue_lines_collapses_consecutive_duplicates(self):
        cues = [tx.Cue("a", "b", t) for t in ("x", "x", "y", "x")]
        self.assertEqual(tx.cue_lines(cues), ["x", "y", "x"])

    def test_render_txt(self):
        url_txt = tx.render_txt(meta(), "subtitles:en", ["line"])
        self.assertEqual(url_txt.splitlines()[:7], [
            "title: Fake Video", "channel: Fake Channel", "date: 2026-01-02", "duration: 2m05s",
            f"source: {URL}", "transcript: subtitles:en", "---"])
        file_txt = tx.render_txt(meta(channel=None, date=None, duration_s=3725.0), "whisper:m", [])
        self.assertIn("duration: 1h02m05s", file_txt)
        self.assertNotIn("channel:", file_txt)
        self.assertNotIn("date:", file_txt)


class Publish(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out)

    def replace_failing_on(self, nth: int):
        real = os.replace
        calls = {"n": 0}

        def fake(src, dst):
            calls["n"] += 1
            if calls["n"] == nth:
                raise OSError("injected rename failure")
            return real(src, dst)
        return mock.patch.object(tx.os, "replace", fake)

    def test_fresh_publish_interrupted_leaves_no_txt(self):
        with self.replace_failing_on(2), self.assertRaisesRegex(OSError, "injected"):
            tx.publish(self.out, "s", "SRT", "TXT")
        self.assertFalse((self.out / "s.txt").exists())
        self.assertEqual([p.name for p in self.out.iterdir() if p.name.endswith(".tmp")], [])

    def test_forced_publish_interrupted_removes_stale_marker(self):
        (self.out / "s.srt").write_text("OLD SRT")
        (self.out / "s.txt").write_text("OLD TXT")
        with self.replace_failing_on(2), self.assertRaisesRegex(OSError, "injected"):
            tx.publish(self.out, "s", "NEW SRT", "NEW TXT")
        self.assertFalse((self.out / "s.txt").exists())
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["s.srt"])

    def test_publish_writes_pair(self):
        tx.publish(self.out, "s", "SRT", "TXT")
        self.assertEqual((self.out / "s.srt").read_text(), "SRT")
        self.assertEqual((self.out / "s.txt").read_text(), "TXT")


class CLI(unittest.TestCase):
    """Runs the real script as a subprocess against fakes on a hermetic PATH."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.log = self.tmp / "fake.log"
        self.cache = self.tmp / "cache"
        model = self.cache / "transcribe" / "models" / f"ggml-{tx.DEFAULT_MODEL}.bin"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"fake-model")
        self.media = self.tmp / "clip.mp3"
        self.media.write_bytes(b"fake-media")
        self.out = self.tmp / "out"

    def env(self, tools=fakes.TOOLS, **extra):
        bin_dir = fakes.make_bin(Path(tempfile.mkdtemp(dir=self.tmp)), tools)
        env = {"PATH": str(bin_dir), "FAKE_LOG": str(self.log), "XDG_CACHE_HOME": str(self.cache),
               "HOME": str(self.tmp)}
        env.update(extra)
        return env

    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(SCRIPT), "-o", str(self.out), *args],
                              env=env or self.env(), cwd=self.tmp, capture_output=True, text=True,
                              check=False)

    def calls(self, tool=None):
        if not self.log.exists():
            return []
        entries = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [e for e in entries if tool is None or e["tool"] == tool]

    def results(self, proc):
        return [line.split("\t") for line in proc.stdout.splitlines()]

    def write_meta(self, **fields):
        info = {"_type": "video", "id": "vid123", "title": "Fake Video", "channel": "Fake Channel",
                "upload_date": "20260102", "duration": 125, "language": "en",
                "webpage_url": URL, "subtitles": {}}
        info.update(fields)
        path = self.tmp / "meta.json"
        path.write_text(json.dumps(info))
        return str(path)

    def test_url_with_human_subtitles_skips_whisper(self):
        env = self.env(FAKE_META=self.write_meta(subtitles={"en": [{}], "live_chat": [{}]}))
        proc = self.run_cli(URL, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.results(proc)[0][:2], ["OK", "subtitles:en"])
        self.assertEqual(self.calls("whisper-cli"), [])
        txt = (self.out / "fake-video-vid123.txt").read_text()
        self.assertIn("channel: Fake Channel", txt)
        self.assertIn("date: 2026-01-02", txt)
        self.assertIn("Hello from the fake.", txt)
        for call in self.calls("yt-dlp"):
            self.assertIn("--ignore-config", call["argv"])

    def test_url_without_subtitles_uses_whisper(self):
        proc = self.run_cli(URL)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.results(proc)[0][:2], ["OK", f"whisper:{tx.DEFAULT_MODEL}"])
        self.assertEqual(len(self.calls("whisper-cli")), 1)
        whisper_argv = self.calls("whisper-cli")[0]["argv"]
        self.assertEqual(whisper_argv[whisper_argv.index("-l") + 1], "auto")
        for call in self.calls("yt-dlp"):
            self.assertIn("--ignore-config", call["argv"])

    def test_wav_download_does_not_collide_with_conversion_target(self):
        proc = self.run_cli(URL, env=self.env(FAKE_AUDIO_EXT="wav"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.results(proc)[0][0], "OK")

    def test_local_file(self):
        proc = self.run_cli(str(self.media))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        txt_path = Path(self.results(proc)[0][2])
        self.assertTrue(txt_path.name.startswith("clip-"))
        txt = txt_path.read_text()
        self.assertIn("duration: 0m11s", txt)
        self.assertNotIn("channel:", txt)
        self.assertEqual(self.calls("yt-dlp"), [])

    def test_batch_isolates_missing_input(self):
        proc = self.run_cli(URL, str(self.tmp / "missing.mp4"), str(self.media))
        self.assertEqual(proc.returncode, 1)
        self.assertEqual([r[0] for r in self.results(proc)], ["OK", "FAIL", "OK"])
        self.assertEqual(self.results(proc)[1][2], "not found")

    def test_second_run_skips_without_work(self):
        self.run_cli(URL, str(self.media))
        self.log.unlink()
        proc = self.run_cli(URL, str(self.media))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([r[0] for r in self.results(proc)], ["SKIP", "SKIP"])
        self.assertEqual({c["tool"] for c in self.calls()}, {"yt-dlp", "ffprobe"})
        self.assertTrue(all("-J" in c["argv"] for c in self.calls("yt-dlp")))

    def test_force_rewrites(self):
        self.run_cli(URL)
        txt = self.out / "fake-video-vid123.txt"
        txt.write_text("stale")
        proc = self.run_cli("--force", URL)
        self.assertEqual(self.results(proc)[0][0], "OK")
        self.assertIn("Hello from the fake.", txt.read_text())

    def test_forced_run_failing_before_publish_keeps_previous_pair(self):
        self.run_cli(URL)
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        proc = self.run_cli("--force", URL, env=self.env(FAKE_WHISPER_MODES="fail"))
        self.assertEqual(self.results(proc)[0][0], "FAIL")
        self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, before)

    def test_whisper_failures_are_per_input(self):
        for mode, reason in [("fail", "whisper-cli failed"),
                             ("exit0-no-output", "produced no transcript"),
                             ("empty", "empty transcript"),
                             ("malformed", "malformed subtitles")]:
            with self.subTest(mode=mode):
                shutil.rmtree(self.out, ignore_errors=True)
                if self.log.exists():
                    self.log.unlink()
                other = self.tmp / f"other-{mode}.mp3"
                other.write_bytes(b"fake-media")
                proc = self.run_cli(str(self.media), str(other),
                                    env=self.env(FAKE_WHISPER_MODES=f"{mode},ok"))
                rows = self.results(proc)
                self.assertEqual(proc.returncode, 1)
                self.assertEqual([r[0] for r in rows], ["FAIL", "OK"])
                self.assertIn(reason, rows[0][2])
                self.assertEqual(len(list(self.out.glob("clip-*"))), 0)

    def test_yt_dlp_failure_suggests_update(self):
        proc = self.run_cli(URL, env=self.env(FAKE_YTDLP_FAIL="1"))
        row = self.results(proc)[0]
        self.assertEqual(row[0], "FAIL")
        self.assertIn("update yt-dlp", row[2])

    def test_missing_dependency_exits_before_work(self):
        tools = tuple(t for t in fakes.TOOLS if t != "whisper-cli")
        proc = self.run_cli(URL, env=self.env(tools=tools))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("whisper-cli", proc.stderr)
        self.assertIn("whisper-cpp", proc.stderr)
        self.assertEqual(self.calls(), [])

    def test_playlist_and_unsupported_scheme_fail(self):
        env = self.env(FAKE_META=self.write_meta(_type="playlist"))
        proc = self.run_cli(URL, "ftp://example.test/v.mp4", env=env)
        rows = self.results(proc)
        self.assertEqual([r[0] for r in rows], ["FAIL", "FAIL"])
        self.assertIn("pass video URLs", rows[0][2])
        self.assertIn("unsupported scheme", rows[1][2])

    def test_creates_nested_output_dir(self):
        self.out = self.tmp / "a" / "b" / "c"
        proc = self.run_cli(str(self.media))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(list(self.out.glob("*.txt"))), 1)

    def test_model_download(self):
        shutil.rmtree(self.cache)
        model = self.cache / "transcribe" / "models" / f"ggml-{tx.DEFAULT_MODEL}.bin"
        proc = self.run_cli(str(self.media))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(model.is_file())
        self.assertEqual(list(model.parent.glob("*.part")), [])

    def test_model_download_failure(self):
        shutil.rmtree(self.cache)
        proc = self.run_cli(str(self.media), env=self.env(FAKE_CURL_FAIL="1"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("model download failed", proc.stderr)
        self.assertEqual(list((self.cache / "transcribe" / "models").glob("*")), [])

    def test_cached_model_needs_no_curl(self):
        tools = tuple(t for t in fakes.TOOLS if t != "curl")
        proc = self.run_cli(str(self.media), env=self.env(tools=tools))
        self.assertEqual(proc.returncode, 0, proc.stderr)


class InProcess(unittest.TestCase):
    """Runs main() in-process so a filesystem failure can be injected mid-batch."""

    def test_publication_failure_is_per_input(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        bin_dir = fakes.make_bin(tmp / "bin")
        model = tmp / "cache" / "transcribe" / "models" / f"ggml-{tx.DEFAULT_MODEL}.bin"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"fake-model")
        first, second = tmp / "first.mp3", tmp / "second.mp3"
        first.write_bytes(b"x")
        second.write_bytes(b"x")
        real = os.replace
        calls = {"n": 0}

        def failing_once(src, dst):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("injected rename failure")
            return real(src, dst)

        env = {"PATH": str(bin_dir), "XDG_CACHE_HOME": str(tmp / "cache")}
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, env), mock.patch.object(tx.os, "replace", failing_once), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = tx.main(["-o", str(tmp / "out"), str(first), str(second)])
        rows = [line.split("\t") for line in stdout.getvalue().splitlines()]
        self.assertEqual(code, 1)
        self.assertEqual([r[0] for r in rows], ["FAIL", "OK"])
        self.assertIn("filesystem error", rows[0][2])


JFK = Path("/opt/homebrew/share/whisper-cpp/jfk.wav")
REAL_MODEL = tx.cache_dir() / f"ggml-{tx.DEFAULT_MODEL}.bin"


@unittest.skipUnless(shutil.which("whisper-cli") and shutil.which("ffmpeg") and JFK.is_file()
                     and REAL_MODEL.is_file(), "needs whisper-cli, ffmpeg, jfk.wav, cached model")
class Integration(unittest.TestCase):
    def test_real_transcription_from_unrelated_cwd(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        proc = subprocess.run([sys.executable, str(SCRIPT), "-o", "out", str(JFK)],
                              cwd=tmp, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        txt = Path(tmp / proc.stdout.split("\t")[2].strip()).read_text()
        self.assertIn("ask not what your country can do for you", txt.lower())


if __name__ == "__main__":
    unittest.main()
