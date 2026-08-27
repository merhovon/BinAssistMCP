import sys
import types
import unittest
from unittest.mock import patch

from src.binassist_mcp.context import BinAssistMCPBinaryContextManager


class FakeBinaryView:
    def __init__(self, filename):
        self.file = types.SimpleNamespace(filename=filename)
        self.functions = []


class FakeTab:
    def __init__(self, binary_view):
        self.binary_view = binary_view

    def getCurrentBinaryView(self):
        return self.binary_view


class FakeUIWindow:
    def __init__(self, binary_views):
        self.tabs = [FakeTab(binary_view) for binary_view in binary_views]

    def getTabs(self):
        return self.tabs


class ContextSyncTests(unittest.TestCase):
    def setUp(self):
        self.binary_view = FakeBinaryView("/tmp/session-independent.bndb")
        window = FakeUIWindow([self.binary_view])
        ui_context = type("UIContext", (), {"allContexts": staticmethod(lambda: [window])})
        self.ui_module = types.SimpleNamespace(UIContext=ui_context)

    def _patch_binary_ninja_ui(self):
        return patch.dict(sys.modules, {"binaryninjaui": self.ui_module})

    def test_get_binary_syncs_a_fresh_session_on_cache_miss(self):
        manager = BinAssistMCPBinaryContextManager()

        with patch("src.binassist_mcp.context.BINJA_AVAILABLE", True), self._patch_binary_ninja_ui():
            result = manager.get_binary("session-independent.bndb")

        self.assertIs(result, self.binary_view)
        self.assertEqual(manager.list_binaries(), ["session-independent.bndb"])

    def test_get_binary_info_does_not_require_list_binaries_first(self):
        manager = BinAssistMCPBinaryContextManager()

        with patch("src.binassist_mcp.context.BINJA_AVAILABLE", True), self._patch_binary_ninja_ui():
            info = manager.get_binary_info("session-independent.bndb")

        self.assertIs(info.view, self.binary_view)

    def test_analysis_progress_refreshes_before_reporting_not_found(self):
        manager = BinAssistMCPBinaryContextManager()

        with patch("src.binassist_mcp.context.BINJA_AVAILABLE", True), self._patch_binary_ninja_ui():
            progress = manager.get_analysis_progress("session-independent.bndb")

        self.assertNotEqual(progress["state"], "not_found")
        self.assertIn("session-independent.bndb", manager)

    def test_unknown_binary_still_reports_available_names(self):
        manager = BinAssistMCPBinaryContextManager()

        with patch("src.binassist_mcp.context.BINJA_AVAILABLE", True), self._patch_binary_ninja_ui():
            with self.assertRaisesRegex(KeyError, "session-independent.bndb"):
                manager.get_binary("missing.bndb")


if __name__ == "__main__":
    unittest.main()
