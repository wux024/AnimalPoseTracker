"""The inferencer's own configuration file: template, defaults, overrides.

The inferencer reads no project configuration; this file (any path passed as
``--config``) is a plain two-layer default set between the command line and
the built-in defaults. These tests pin the template, the merge order and the
refusals.
"""

import io
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml

from animalposetracker.cfg import CFG_DIR
from animalposetracker.inference.cli import (
    _INFER_FILE_KEYS,
    _infer_file_values,
    _make_argument_parser,
    _write_template,
)


def _template_text() -> str:
    return (CFG_DIR / "infer.example.yaml").read_text(encoding="utf-8")


class TemplateTests(unittest.TestCase):
    def test_the_template_lists_every_configurable_key(self):
        text = _template_text()
        missing = [key for key in sorted(_INFER_FILE_KEYS) if f"# {key}:" not in text]
        self.assertEqual(missing, [])

    def test_the_template_values_match_the_parser_defaults(self):
        # The template is hand-written; this keeps its documented defaults in
        # step with the parser's.
        text = _template_text()
        args = _make_argument_parser().parse_args([])
        for key, expected in (
            ("conf", "0.25"),
            ("iou", "0.45"),
            ("detector_conf", "0.5"),
            ("imgsz", "640"),
            ("line_width", "2"),
            ("point_radius", "4"),
            ("bits", "fp32"),
            ("agnostic_nms", "true"),
        ):
            with self.subTest(key=key):
                self.assertEqual(str(getattr(args, key)).lower(), expected)
                match = re.search(rf"^# {key}: (\S+)", text, flags=re.MULTILINE)
                self.assertIsNotNone(match, f"template carries no default for {key}")
                self.assertEqual(match.group(1).lower(), expected)


class FileValueTests(unittest.TestCase):
    def _write_file(self, temporary, payload) -> Path:
        path = Path(temporary) / "infer.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_file_values_become_defaults_and_cli_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_file(temporary, {"conf": 0.7, "line_width": 9, "device": "cpu"})
            values = _infer_file_values(path)

        self.assertEqual(values, {"conf": 0.7, "line_width": 9, "device": "cpu"})
        parser = _make_argument_parser()
        parser.set_defaults(**values)
        parsed = parser.parse_args([])
        self.assertEqual(parsed.conf, 0.7)
        self.assertEqual(parsed.line_width, 9)
        self.assertEqual(parsed.device, "cpu")
        # An explicit flag still beats the file.
        overridden = parser.parse_args(["--conf", "0.9"])
        self.assertEqual(overridden.conf, 0.9)
        self.assertEqual(overridden.line_width, 9)

    def test_data_description_keys_are_ignored_here(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_file(
                temporary,
                {"names": {0: "animal"}, "kpt_shape": [17, 3], "skeleton": []},
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                values = _infer_file_values(path)

        self.assertEqual(values, {})
        self.assertEqual(stderr.getvalue(), "")

    def test_unknown_key_warns_and_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_file(temporary, {"konf": 0.5, "conf": 0.3})
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                values = _infer_file_values(path)

        self.assertEqual(values, {"conf": 0.3})
        self.assertIn("konf", stderr.getvalue())

    def test_missing_file_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                _infer_file_values(Path(temporary) / "nope.yaml")

    def test_no_path_means_no_values(self):
        self.assertEqual(_infer_file_values(None), {})


class WriteTemplateTests(unittest.TestCase):
    def test_write_template_roundtrips_the_packaged_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "infer.example.yaml"
            output = io.StringIO()
            with redirect_stdout(output):
                result = _write_template(target)

            self.assertEqual(result, 0)
            self.assertEqual(target.read_text(encoding="utf-8"), _template_text())
            self.assertIn(str(target), output.getvalue())


if __name__ == "__main__":
    unittest.main()
