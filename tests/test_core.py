import tempfile
import unittest
from pathlib import Path

from agentcord.core import PointsManager, SearchAllowlist, WorkspaceError, WorkspaceManager


class WorkspaceManagerTests(unittest.TestCase):
    def test_enforces_5mb_limit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(Path(d), max_bytes=10)
            mgr.write_text(1, "ok.txt", "12345")
            with self.assertRaises(WorkspaceError):
                mgr.write_text(1, "big.txt", "x" * 6)

    def test_blocks_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(Path(d), max_bytes=100)
            with self.assertRaises(WorkspaceError):
                mgr.write_text(1, "../hack.txt", "no")


class PointsManagerTests(unittest.TestCase):
    def test_charges_and_rejects_when_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            pm = PointsManager(Path(d) / "state.json", default_points=2)
            with self.assertRaises(WorkspaceError):
                pm.charge(1, "openai-large", "abcd" * 100, "efgh" * 100)

            pm.set_points(1, 100)
            cost = pm.charge(1, "llama", "hello", "world")
            self.assertGreaterEqual(cost, 1)
            self.assertEqual(pm.get_points(1), 100 - cost)


class SearchAllowlistTests(unittest.TestCase):
    def test_only_registered_url_is_allowed(self) -> None:
        sa = SearchAllowlist()
        result = "See https://example.com/a and https://example.org/b?x=1"
        sa.register_urls(1, sa.extract_urls(result))
        self.assertTrue(sa.is_allowed(1, "https://example.com/a"))
        self.assertFalse(sa.is_allowed(1, "https://example.com/other"))

    def test_extract_urls_strips_trailing_punctuation(self) -> None:
        sa = SearchAllowlist()
        result = "Read https://example.com/path, then https://example.org/a.)"
        urls = sa.extract_urls(result)
        self.assertIn("https://example.com/path", urls)
        self.assertIn("https://example.org/a", urls)


if __name__ == "__main__":
    unittest.main()
