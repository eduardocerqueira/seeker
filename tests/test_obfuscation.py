import os
import tempfile
import unittest
from pathlib import Path

from seeker import util


class ObfuscationTests(unittest.TestCase):
    def test_get_config_is_independent_of_cwd(self):
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                os.chdir(tmp_dir)
                self.assertIn("Python", util.get_config("gists", "language"))
        finally:
            os.chdir(original_cwd)

    def test_obfuscate_masks_sensitive_value(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snippet_dir = Path(tmp_dir)
            snippet = snippet_dir / "sample.py"
            snippet.write_text('password = "supersecret"\n', encoding="utf-8")

            original_snippet_dir = util.SNIPPET_DIR
            util.SNIPPET_DIR = snippet_dir
            try:
                util.obfuscate()
            finally:
                util.SNIPPET_DIR = original_snippet_dir

            content = snippet.read_text(encoding="utf-8")
            self.assertIn('password = "**********"', content)
            self.assertNotIn("supersecret", content)

    def test_obfuscate_keeps_pre_masked_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snippet_dir = Path(tmp_dir)
            snippet = snippet_dir / "sample.py"
            masked_content = 'token = "**********"\n'
            snippet.write_text(masked_content, encoding="utf-8")

            original_snippet_dir = util.SNIPPET_DIR
            util.SNIPPET_DIR = snippet_dir
            try:
                util.obfuscate()
            finally:
                util.SNIPPET_DIR = original_snippet_dir

            self.assertEqual(masked_content, snippet.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
