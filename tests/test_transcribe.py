from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
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
STEM = re.compile(r"^[a-z0-9-]*[0-9a-f]{8}$")

# Verbatim ffmpeg 8.1 output for an ASS cue "First\N\NSecond": a blank line inside a cue.
FFMPEG_ASS_SRT = ('1\n00:00:01,000 --> 00:00:02,500\n<font size="20">First\n\nSecond</font>\n\n'
                  '2\n00:00:03,000 --> 00:00:04,000\n<font size="20">Third line</font>\n\n')
# Verbatim ffmpeg 8.1 SRT->SRT output: dialogue containing an arrow and a bare number.
FFMPEG_ARROW_SRT = ("1\n00:00:01,000 --> 00:00:02,000\nprice goes A --> B\n\n"
                    "2\n00:00:03,000 --> 00:00:04,000\n2026\n\n"
                    "3\n00:00:05,000 --> 00:00:06,000\nlast\n\n")
# Verbatim ffmpeg 8.1 output for an ASS cue "First\N\N2026\NA --> B".
FFMPEG_ASS_NUMERIC_SRT = ('1\n00:00:01,000 --> 00:00:02,500\n'
                          '<font size="20">First\n\n2026\nA --> B</font>\n\n')
# Verbatim ffmpeg 8.1 SRT->SRT output: comparison operators next to real formatting tags.
FFMPEG_COMPARISON_SRT = ("1\n00:00:01,000 --> 00:00:02,000\nif x < 5 and y > 3\n\n"
                         "2\n00:00:03,000 --> 00:00:04,000\n<i>styled</i> <b>text</b>\n\n")


@dataclass(frozen=True)
class Row:
    status: str
    how: str
    detail: str


def rows(stdout: str) -> list[Row]:
    return [Row(*line.split("\t")) for line in stdout.splitlines()]


def meta(**overrides):
    base = {"title": "Fake Video", "channel": "Fake Channel", "date": "2026-01-02",
            "duration_s": 125.0, "source": URL, "language": "en", "subtitle_langs": ()}
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

    def test_classify_inaccessible_path_is_an_input_error(self):
        with mock.patch.object(tx.Path, "is_file", side_effect=PermissionError("denied")), \
                self.assertRaisesRegex(tx.InputError, "cannot access"):
            tx.classify("/locked/clip.mp3")

    def test_model_lookup_filesystem_error_is_preflight(self):
        with mock.patch.object(tx.shutil, "which", return_value="/bin/x"), \
                mock.patch.object(tx.Path, "is_file", side_effect=PermissionError("denied")), \
                self.assertRaisesRegex(tx.PreflightError, "model cache error"):
            tx.preflight([], tx.DEFAULT_MODEL)

    def test_stem_is_safe_bounded_and_unique_per_source(self):
        self.assertRegex(tx.stem_for(meta(title="Héllo, Wörld!")), r"^hello-world-[0-9a-f]{8}$")
        self.assertRegex(tx.stem_for(meta(title="日本語")), r"^[0-9a-f]{8}$")
        for title in ("../other", ".hidden", "a/b", "x" * 200, "a " * 100):
            stem = tx.stem_for(meta(title=title))
            self.assertRegex(stem, STEM)
            self.assertLessEqual(len(stem), 69)
        a = tx.stem_for(meta(title="episode", source="https://a.test/episode.mp3"))
        b = tx.stem_for(meta(title="episode", source="https://b.test/episode.mp3"))
        self.assertNotEqual(a, b)

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
                "we are\r\n\r\n00:00:04,000 --> 00:00:05,000 X1:10\r\nno index\r\n")
        cues = tx.parse_srt(text)
        self.assertEqual([c.text for c in cues], ["All right, so here we are", "no index"])
        self.assertEqual((cues[0].start, cues[0].end), ("00:00:01.200", "00:00:03.360"))
        self.assertEqual(tx.parse_srt("  \n"), [])

    def test_parse_srt_keeps_blank_lines_inside_a_cue(self):
        cues = tx.parse_srt(FFMPEG_ASS_SRT)
        self.assertEqual([c.text for c in cues], ["First Second", "Third line"])

    def test_parse_srt_keeps_arrows_and_numbers_in_dialogue(self):
        self.assertEqual([c.text for c in tx.parse_srt(FFMPEG_ARROW_SRT)],
                         ["price goes A --> B", "2026", "last"])
        unindexed = "00:00:01,000 --> 00:00:02,000\n2026\n00:00:03,000 --> 00:00:04,000\nnext\n"
        self.assertEqual([c.text for c in tx.parse_srt(unindexed)], ["2026", "next"])
        bare_arrow = "1\n00:00:01,000 --> 00:00:02,000\nA --> B\n"
        self.assertEqual([c.text for c in tx.parse_srt(bare_arrow)], ["A --> B"])
        self.assertEqual([c.text for c in tx.parse_srt(FFMPEG_ASS_NUMERIC_SRT)],
                         ["First 2026 A --> B"])

    def test_parse_srt_strips_only_formatting_tags(self):
        self.assertEqual([c.text for c in tx.parse_srt(FFMPEG_COMPARISON_SRT)],
                         ["if x < 5 and y > 3", "styled text"])
        vtt_style = "1\n00:00:01,000 --> 00:00:02,000\n<c.yellow>hi</c> <v Bob>there</v>\n"
        self.assertEqual([c.text for c in tx.parse_srt(vtt_style)], ["hi there"])

    def test_parse_srt_rejects_malformed(self):
        for text, reason in [("this is not\nan srt file\n", "before the first timing line"),
                             ("1\nfoo --> bar\ntext\n", "malformed subtitles"),
                             ("stray\n1\n00:00:01,000 --> 00:00:02,000\nx\n", "before the first"),
                             ("1\n00:00:61,000 --> 00:01:30,000\nx\n", "malformed subtitles"),
                             ("1\n00:00:05,000 --> 00:00:01,000\nx\n", "malformed subtitles"),
                             ("foo --> bar\nx\n", "bad timing line")]:
            with self.subTest(text=text), self.assertRaisesRegex(tx.InputError, reason):
                tx.parse_srt(text)

    def test_cue_lines_collapses_consecutive_duplicates(self):
        cues = [tx.Cue("a", "b", t) for t in ("x", "x", "y", "x")]
        self.assertEqual(tx.cue_lines(cues), ["x", "y", "x"])

    def test_render_txt(self):
        url_txt = tx.render_txt(meta(), tx.Provenance("subtitles", "en"), ["line"])
        self.assertEqual(url_txt.splitlines()[:7], [
            "title: Fake Video", "channel: Fake Channel", "date: 2026-01-02", "duration: 2m05s",
            f"source: {URL}", "transcript: subtitles:en", "---"])
        file_txt = tx.render_txt(meta(channel=None, date=None, duration_s=3725.0),
                                 tx.Provenance("whisper", "m"), [])
        self.assertIn("duration: 1h02m05s", file_txt)
        self.assertNotIn("channel:", file_txt)
        self.assertNotIn("date:", file_txt)

    def test_redact(self):
        self.assertEqual(tx.redact("GET https://user:pw@host.test/x failed"),
                         "GET https://***@host.test/x failed")
        self.assertEqual(tx.redact("https://h.test/v.mp3?token=SECRET&id=7&X-Amz-Signature=ab"),
                         "https://h.test/v.mp3?token=***&id=7&X-Amz-Signature=***")
        self.assertEqual(tx.redact(URL), URL)
        prov = tx.Provenance("whisper", "m")
        creds = tx.render_txt(meta(source="https://u:p@h.test/v"), prov, [])
        self.assertIn("source: https://***@h.test/v", creds)

    def test_as_duration(self):
        self.assertEqual(tx.as_duration("11.0"), 11.0)
        for bad in ("NaN", "inf", -1, None, "x", [1], 10**400):
            self.assertIsNone(tx.as_duration(bad))

    def test_result_line_is_always_three_fields(self):
        line = tx.format_result(tx.Result("x", "FAIL", error="tool failed:\tbad\nthing"))
        self.assertEqual(line.split("\t"), ["FAIL", "-", "tool failed: bad thing"])
        self.assertNotIn("\n", line)


class Publish(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out)

    def replace_failing_on(self, nth: int):
        real = os.replace
        calls = {"n": 0}

        def fake(src, dst):
            if Path(dst).parent == self.out:
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

    def write_meta(self, raw=None, **fields):
        info = {"_type": "video", "id": "vid123", "title": "Fake Video", "channel": "Fake Channel",
                "upload_date": "20260102", "duration": 125, "language": "en",
                "webpage_url": URL, "subtitles": {}}
        info.update(fields)
        path = self.tmp / "meta.json"
        path.write_text(raw if raw is not None else json.dumps(info))
        return str(path)

    def test_url_with_human_subtitles_skips_whisper(self):
        env = self.env(FAKE_META=self.write_meta(subtitles={"en": [{}], "live_chat": [{}]}))
        proc = self.run_cli(URL, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = rows(proc.stdout)[0]
        self.assertEqual((row.status, row.how), ("OK", "subtitles:en"))
        self.assertEqual(self.calls("whisper-cli"), [])
        txt = Path(row.detail).read_text()
        self.assertIn("channel: Fake Channel", txt)
        self.assertIn("date: 2026-01-02", txt)
        self.assertIn("Hello from the fake.", txt)
        for call in self.calls("yt-dlp"):
            self.assertIn("--ignore-config", call["argv"])

    def test_url_without_subtitles_uses_whisper(self):
        proc = self.run_cli(URL)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = rows(proc.stdout)[0]
        self.assertEqual((row.status, row.how), ("OK", f"whisper:{tx.DEFAULT_MODEL}"))
        self.assertRegex(Path(row.detail).name, r"^fake-video-[0-9a-f]{8}\.txt$")
        whisper_argv = self.calls("whisper-cli")[0]["argv"]
        self.assertEqual(whisper_argv[whisper_argv.index("-l") + 1], "auto")
        for call in self.calls("yt-dlp"):
            self.assertIn("--ignore-config", call["argv"])

    def test_same_site_id_on_different_hosts_does_not_collide(self):
        proc = self.run_cli("https://a.test/episode.mp3", "https://b.test/episode.mp3")
        self.assertEqual([r.status for r in rows(proc.stdout)], ["OK", "OK"])
        self.assertEqual(len(list(self.out.glob("*.txt"))), 2)

    def test_wav_download_does_not_collide_with_conversion_target(self):
        proc = self.run_cli(URL, env=self.env(FAKE_AUDIO_EXT="wav"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(rows(proc.stdout)[0].status, "OK")

    def test_local_file(self):
        proc = self.run_cli(str(self.media))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        txt_path = Path(rows(proc.stdout)[0].detail)
        self.assertRegex(txt_path.name, r"^clip-[0-9a-f]{8}\.txt$")
        txt = txt_path.read_text()
        self.assertIn("duration: 0m11s", txt)
        self.assertNotIn("channel:", txt)
        self.assertEqual(self.calls("yt-dlp"), [])

    def test_malformed_metadata_fails_only_that_input(self):
        proc = self.run_cli(URL, str(self.media), env=self.env(FAKE_META=self.write_meta(raw="[]")))
        result = rows(proc.stdout)
        self.assertEqual([r.status for r in result], ["FAIL", "OK"])
        self.assertIn("unexpected metadata", result[0].detail)

    def test_invalid_duration_is_omitted(self):
        for duration in ("NaN", 10**400):
            with self.subTest(duration=str(duration)[:10]):
                shutil.rmtree(self.out, ignore_errors=True)
                env = self.env(FAKE_META=self.write_meta(duration=duration))
                proc = self.run_cli(URL, str(self.media), env=env)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual([r.status for r in rows(proc.stdout)], ["OK", "OK"])
                self.assertNotIn("duration:", Path(rows(proc.stdout)[0].detail).read_text())

    def test_batch_isolates_missing_input(self):
        proc = self.run_cli(URL, str(self.tmp / "missing.mp4"), str(self.media))
        result = rows(proc.stdout)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual([r.status for r in result], ["OK", "FAIL", "OK"])
        self.assertEqual(result[1].detail, "not found")

    def test_second_run_skips_without_work(self):
        self.run_cli(URL, str(self.media))
        self.log.unlink()
        proc = self.run_cli(URL, str(self.media))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([r.status for r in rows(proc.stdout)], ["SKIP", "SKIP"])
        self.assertEqual({c["tool"] for c in self.calls()}, {"yt-dlp", "ffprobe"})
        self.assertTrue(all("-J" in c["argv"] for c in self.calls("yt-dlp")))

    def test_force_rewrites(self):
        txt = Path(rows(self.run_cli(URL).stdout)[0].detail)
        txt.write_text("stale")
        proc = self.run_cli("--force", URL)
        self.assertEqual(rows(proc.stdout)[0].status, "OK")
        self.assertIn("Hello from the fake.", txt.read_text())

    def test_forced_run_failing_before_publish_keeps_previous_pair(self):
        self.run_cli(URL)
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        proc = self.run_cli("--force", URL, env=self.env(FAKE_WHISPER_MODES="fail"))
        self.assertEqual(rows(proc.stdout)[0].status, "FAIL")
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
                result = rows(proc.stdout)
                self.assertEqual(proc.returncode, 1)
                self.assertEqual([r.status for r in result], ["FAIL", "OK"])
                self.assertIn(reason, result[0].detail)
                self.assertEqual(len(list(self.out.glob("clip-*"))), 0)

    def test_yt_dlp_failure_suggests_update(self):
        proc = self.run_cli(URL, env=self.env(FAKE_YTDLP_FAIL="1"))
        row = rows(proc.stdout)[0]
        self.assertEqual(row.status, "FAIL")
        self.assertIn("update yt-dlp", row.detail)

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
        result = rows(proc.stdout)
        self.assertEqual([r.status for r in result], ["FAIL", "FAIL"])
        self.assertIn("pass video URLs", result[0].detail)
        self.assertIn("unsupported scheme", result[1].detail)

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

    def test_model_download_failure_removes_partial_file(self):
        shutil.rmtree(self.cache)
        proc = self.run_cli(str(self.media), env=self.env(FAKE_CURL_FAIL="1"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("model download failed", proc.stderr)
        curl_argv = self.calls("curl")[0]["argv"]
        curl_out = Path(curl_argv[curl_argv.index("-o") + 1])
        self.assertTrue(curl_out.name.endswith(".part"))
        self.assertFalse(curl_out.exists())
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
        out = tmp / "out"
        first, second = tmp / "first.mp3", tmp / "second.mp3"
        first.write_bytes(b"x")
        second.write_bytes(b"x")
        real = os.replace
        failed = []

        def failing_once(src, dst):
            if Path(dst).parent == out and not failed:
                failed.append(Path(dst).name)
                raise OSError("injected rename failure")
            return real(src, dst)

        env = {"PATH": str(bin_dir), "XDG_CACHE_HOME": str(tmp / "cache"), "HOME": str(tmp),
               "TMPDIR": tempfile.gettempdir()}
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(tx.os, "replace", failing_once), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = tx.main(["-o", str(out), str(first), str(second)])
        result = rows(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0].startswith("first-") and failed[0].endswith(".srt"))
        self.assertEqual([r.status for r in result], ["FAIL", "OK"])
        self.assertIn("filesystem error", result[0].detail)


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
        txt = (tmp / rows(proc.stdout)[0].detail).read_text()
        self.assertIn("ask not what your country can do for you", txt.lower())


if __name__ == "__main__":
    unittest.main()
